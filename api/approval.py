"""
Human Approval Policy Layer -- API surface.

`POST /agent/execute` is the new, gated primary entry point: it plans a
turn with sensitive actions *held* (see `app.agent.run_agent(...,
hold_sensitive=True)`), scores it with the existing, unmodified detector,
and asks `app.policy.evaluate` (deterministic, no LLM) what to do next.

`POST /score` (api/score.py) is untouched and still executes immediately
and unconditionally -- it remains the raw scoring endpoint used by the
demo/eval harness. `POST /agent/execute` is additive, not a replacement.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_db
from api.schemas import (
    AgentExecuteRequest,
    AgentExecuteResult,
    ApprovalActionResult,
    ApprovalRequiredOut,
    ApprovalStatusOut,
    ApproveRequest,
    RejectRequest,
)
from app import policy
from app.agent import finalize_pending_call, run_agent
from app.detector import score_and_explain
from app.repository import (
    create_approval_request,
    get_active_baseline_profile,
    get_approval_request_by_token,
    get_or_create_agent,
    get_session_by_id,
    load_baseline,
    record_anomaly_result,
    record_session,
    record_tool_execution_event,
    resolve_approval,
)

router = APIRouter(tags=["approval"])

_REJECTED_RESPONSE = (
    "This request was reviewed by the security policy layer and the sensitive action(s) it "
    "would have performed were not executed. If you believe this is an error, please contact support."
)


def _compose_approved_response(prompt: str, executed_calls: list) -> str:
    for call in executed_calls:
        if call.tool_name == "disclose_to_user":
            # Now-authorized disclosure -- this is the only place the
            # underlying content is ever returned to the caller.
            disclosed = call.output.get("disclosed", "")
            return f"[Approved by reviewer] {disclosed}"
    if not executed_calls:
        return f"Handled: {prompt[:60]}"
    summary = ", ".join(f"{c.tool_name} ({c.output.get('status', 'done')})" for c in executed_calls)
    return f"Approved and executed: {summary}."


@router.post("/agent/execute")
async def agent_execute(req: AgentExecuteRequest, db: Session = Depends(get_db)):
    """Plan -> score -> policy decision -> execute-now or pause-for-approval."""
    return _execute_one(db, req.agent_name, req.prompt, None, req.use_real_llm)


@router.post("/agent/execute/injection-demo")
async def agent_execute_injection_demo(agent_name: str = "support-agent-v1", use_real_llm: bool = False, db: Session = Depends(get_db)):
    """Runs all 3 crafted injection scenarios through the gated pipeline
    (rather than the raw /score/injection-demo path) so every one can be
    seen pausing for approval instead of just being scored."""
    from app.scenarios import INJECTION_SCENARIOS

    return {
        "results": [
            _execute_one(db, agent_name, s.prompt, s.poison_payload, use_real_llm)
            for s in INJECTION_SCENARIOS
        ]
    }


def _execute_one(db: Session, agent_name: str, prompt: str, poison_payload: str | None, use_real_llm: bool):
    agent = get_or_create_agent(db, agent_name)
    profile = get_active_baseline_profile(db, agent)
    if profile is None:
        raise HTTPException(status_code=400, detail="No baseline found for this agent. Call /baseline/build first.")
    baseline = load_baseline(profile)

    # Plan the turn with sensitive tool calls HELD -- nothing sensitive is
    # dispatched until the policy engine has evaluated the behavioral score.
    trace = run_agent(prompt, poison_payload=poison_payload, use_real_llm=use_real_llm, hold_sensitive=True)
    result = score_and_explain(trace, baseline)

    session = record_session(db, agent, trace)
    anomaly = record_anomaly_result(db, session, profile, result)

    sensitive_tools = sorted({c.tool_name for c in trace.tool_calls if c.sensitive})
    decision = policy.evaluate(result, baseline.threshold, trace.tool_sequence, sensitive_tools)

    if not decision.approval_required:
        # Safe: execute any held sensitive calls immediately (most normal
        # runs have none at all -- read-only tools were already dispatched
        # for real during planning).
        pending = [c for c in trace.tool_calls if c.pending]
        executed = [finalize_pending_call(c.tool_name, c.params) for c in pending]
        for call in executed:
            record_tool_execution_event(db, session, call, source=trace.provider)
        final_response = _compose_approved_response(trace.prompt, executed) if pending else trace.response_text

        return AgentExecuteResult(
            approval_required=False,
            session_id=session.id,
            trace_id=trace.trace_id,
            prompt=trace.prompt,
            response=final_response,
            tool_sequence=trace.tool_sequence,
            score=result.score,
            flagged=result.flagged,
            risk_level=decision.risk_level.value,
            intent_label=result.intent_label,
            expected_tools=result.expected_tools,
        ).model_dump()

    # Suspicious: pause. Persist the full pending plan + audit context and
    # hand back an approval token; nothing sensitive has executed.
    pending_plan = [{"tool_name": c.tool_name, "params": c.params} for c in trace.tool_calls if c.pending]
    approval = create_approval_request(
        db,
        session=session,
        anomaly=anomaly,
        agent_name=agent_name,
        trace=trace,
        pending_plan=pending_plan,
        decision=decision,
        score=result.score,
        threshold=baseline.threshold,
    )

    return ApprovalRequiredOut(
        approval_required=True,
        risk_level=decision.risk_level.value,
        session_id=session.id,
        trace_id=trace.trace_id,
        approval_token=approval.approval_token,
        prompt=trace.prompt,
        score=result.score,
        threshold=round(baseline.threshold, 2),
        planned_tools=decision.planned_tools,
        sensitive_tools=decision.sensitive_tools,
        reasons=decision.reasons,
        intent_label=result.intent_label,
        expected_tools=result.expected_tools,
    ).model_dump()


@router.post("/approve", response_model=ApprovalActionResult)
async def approve(req: ApproveRequest, db: Session = Depends(get_db)) -> ApprovalActionResult:
    request = get_approval_request_by_token(db, req.approval_token)
    if request is None:
        raise HTTPException(status_code=404, detail="Unknown approval_token.")
    if request.status != "pending":
        raise HTTPException(status_code=409, detail=f"Approval already resolved (status={request.status}).")

    executed = [finalize_pending_call(item["tool_name"], item["params"]) for item in request.pending_plan]
    session = get_session_by_id(db, request.session_id)
    if session is not None:
        for call in executed:
            record_tool_execution_event(db, session, call, source=request.provider)

    final_response = _compose_approved_response(request.prompt, executed)
    resolve_approval(db, request, approved=True, actor=req.approved_by, final_response=final_response)

    return ApprovalActionResult(
        status="approved",
        approval_token=request.approval_token,
        session_id=request.session_id,
        response=final_response,
    )


@router.post("/reject", response_model=ApprovalActionResult)
async def reject(req: RejectRequest, db: Session = Depends(get_db)) -> ApprovalActionResult:
    request = get_approval_request_by_token(db, req.approval_token)
    if request is None:
        raise HTTPException(status_code=404, detail="Unknown approval_token.")
    if request.status != "pending":
        raise HTTPException(status_code=409, detail=f"Approval already resolved (status={request.status}).")

    resolve_approval(db, request, approved=False, actor=req.rejected_by, final_response=_REJECTED_RESPONSE)

    return ApprovalActionResult(
        status="rejected",
        approval_token=request.approval_token,
        session_id=request.session_id,
        response=_REJECTED_RESPONSE,
    )


@router.get("/approvals/{approval_token}", response_model=ApprovalStatusOut)
async def approval_status(approval_token: str, db: Session = Depends(get_db)) -> ApprovalStatusOut:
    request = get_approval_request_by_token(db, approval_token)
    if request is None:
        raise HTTPException(status_code=404, detail="Unknown approval_token.")
    return ApprovalStatusOut(
        approval_token=request.approval_token,
        status=request.status,
        risk_level=request.risk_level,
        session_id=request.session_id,
        planned_tools=request.planned_tools,
        sensitive_tools=request.sensitive_tools,
        reasons=request.reasons,
        requested_at=request.requested_at.isoformat(),
        resolved_at=request.resolved_at.isoformat() if request.resolved_at else None,
        approved_by=request.approved_by,
    )
