from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_db
from api.schemas import InjectionDemoResponse, LayerScoreOut, ScoreRequest, ScoreResponse
from app.agent import run_agent
from app.detector import SessionAccumulator, score_and_explain
from app.repository import (
    get_active_baseline_profile,
    get_or_create_agent,
    load_baseline,
    record_anomaly_result,
    record_session,
    record_session_accumulator_alert,
)
from app.scenarios import INJECTION_SCENARIOS

router = APIRouter(tags=["scoring"])

# In-memory per-session decaying accumulator (bonus extension). Dev-scope;
# production hardening swaps this for a DynamoDB-backed store. Durable
# *outcomes* (the alert itself, once the session crosses its cumulative
# threshold) are still persisted via the existing AnomalyResult/Alert
# tables, so the accumulator's in-memory nature only affects the running
# total surviving a process restart -- not whether an exceeded threshold
# gets recorded.
_ACCUMULATORS: dict[str, SessionAccumulator] = {}


def _score_one(db: Session, agent_name: str, prompt: str, poison_payload: str | None, use_real_llm: bool, session_key: str | None) -> ScoreResponse:
    agent = get_or_create_agent(db, agent_name)
    profile = get_active_baseline_profile(db, agent)
    if profile is None:
        raise HTTPException(status_code=400, detail="No baseline found for this agent. Call /baseline/build first.")
    baseline = load_baseline(profile)

    trace = run_agent(prompt, poison_payload=poison_payload, use_real_llm=use_real_llm)
    result = score_and_explain(trace, baseline)

    session = record_session(db, agent, trace)
    record_anomaly_result(db, session, profile, result)

    session_total = None
    session_turns = None
    session_flagged = None
    if session_key:
        acc = _ACCUMULATORS.setdefault(session_key, SessionAccumulator(session_id=session_key))
        was_flagged = acc.flagged
        session_total = round(acc.add(result.score, per_turn_threshold=baseline.threshold), 2)
        session_turns = acc.turns
        session_flagged = acc.flagged

        # Trigger the session-level alert the moment cumulative suspicion
        # crosses the threshold, even if this (or any prior) individual
        # turn was itself below the per-turn threshold. Fire once per
        # session (edge-triggered), not on every subsequent turn.
        if acc.flagged and not was_flagged:
            record_session_accumulator_alert(
                db, session, profile, session_key, acc.total, acc.turns, baseline.threshold,
            )

    return ScoreResponse(
        trace_id=trace.trace_id,
        score=result.score,
        anomaly_score=result.anomaly_score,
        injection_score=result.injection_score,
        flagged=result.flagged,
        threshold=round(baseline.threshold, 2),
        z_component=result.z_component,
        novelty_component=result.novelty_component,
        sensitive_bonus=result.sensitive_bonus,
        param_pattern_component=result.param_pattern_component,
        novel_bigrams=[list(b) for b in result.novel_bigrams],
        novel_param_signatures=result.novel_param_signatures,
        explanation=result.explanation,
        tool_sequence=trace.tool_sequence,
        session_total=session_total,
        layers=[
            LayerScoreOut(name=l.name, score=l.score, weight=l.weight, reasons=l.reasons)
            for l in result.layers
        ],
        intent_label=result.intent_label,
        expected_tools=result.expected_tools,
        session_turns=session_turns,
        session_flagged=session_flagged,
    )


@router.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest, db: Session = Depends(get_db)) -> ScoreResponse:
    return _score_one(db, req.agent_name, req.prompt, None, req.use_real_llm, req.session_key)


@router.post("/score/injection-demo", response_model=InjectionDemoResponse)
async def score_injection_demo(agent_name: str = "support-agent-v1", use_real_llm: bool = False, db: Session = Depends(get_db)) -> InjectionDemoResponse:
    """Runs all 3 crafted injection scenarios and returns their scores, for the submission demo."""
    results = [
        _score_one(db, agent_name, s.prompt, s.poison_payload, use_real_llm, session_key=None)
        for s in INJECTION_SCENARIOS
    ]
    return InjectionDemoResponse(results=results)
