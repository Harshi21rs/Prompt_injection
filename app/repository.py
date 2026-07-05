"""
Persistence layer: maps detector domain objects (Trace, Baseline,
ScoreResult) onto the existing SQLAlchemy models (Agent, Session,
ToolEvent, BaselineProfile, AnomalyResult, Alert).
"""

from __future__ import annotations

import pickle
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session as DBSession

from app.agent import Trace
from app.detector import Baseline, ScoreResult
from app.policy import PolicyDecision
from models import Agent, AnomalyResult, Alert, ApprovalRequest, BaselineProfile
from models.session import Session as SessionModel
from models.tool_event import ToolEvent


def get_or_create_agent(db: DBSession, name: str, agent_type: str = "customer_support", version: str = "0.1.0") -> Agent:
    agent = db.query(Agent).filter(Agent.name == name).one_or_none()
    if agent:
        return agent
    agent = Agent(name=name, agent_type=agent_type, version=version, status="active")
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


def record_session(db: DBSession, agent: Agent, trace: Trace) -> SessionModel:
    from core.logger import get_logger
    logger = get_logger(__name__)
    logger.debug(f"Saving trace {trace.trace_id} to MySQL...")
    now = datetime.now(timezone.utc)
    session = SessionModel(
        agent_id=agent.id,
        session_name=trace.trace_id,
        started_at=now,
        ended_at=now,
        status="completed",
        metadata_payload={
            "provider": trace.provider,
            "prompt": trace.prompt[:200],
            "response_text": trace.response_text[:2000] if trace.response_text else "",
        },
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    for call in trace.tool_calls:
        event = ToolEvent(
            session_id=session.id,
            tool_name=call.tool_name,
            event_type="sensitive_call" if call.sensitive else "tool_call",
            occurred_at=now,
            payload={"params": call.params, "output": call.output},
            source=trace.provider,
        )
        db.add(event)
    db.commit()
    logger.debug(f"Saved trace {trace.trace_id} to MySQL successfully (session id: {session.id}).")
    return session


def persist_baseline(db: DBSession, agent: Agent, profile_name: str, baseline: Baseline) -> BaselineProfile:
    serialized = pickle.dumps(baseline).hex()
    existing = (
        db.query(BaselineProfile)
        .filter(BaselineProfile.agent_id == agent.id, BaselineProfile.profile_name == profile_name)
        .one_or_none()
    )
    parameters = {
        "means": baseline.means,
        "stds": baseline.stds,
        "known_bigrams": [list(b) for b in baseline.known_bigrams],
        "threshold": baseline.threshold,
        "n_runs": baseline.n_runs,
        "self_scores": baseline.self_scores,
        "pickle_hex": serialized,
        "tool_frequency": baseline.tool_frequency,
        "top_sequences": baseline.top_sequences,
        "transitions": baseline.transitions,
        "avg_trace_length": baseline.avg_trace_length,
        # Behavioral fingerprint enhancement: per-tool parameter-shape sets,
        # exposed read-only via the API/DB. The pickle_hex blob remains the
        # source of truth loaded back into a live Baseline (sets included).
        "known_param_signatures": {
            tool: sorted(sigs) for tool, sigs in baseline.known_param_signatures.items()
        },
        # Multi-layer behavioral fingerprint additions, exposed read-only.
        # The pickle_hex blob remains the source of truth for scoring.
        "avg_tool_frequency": baseline.avg_tool_frequency,
        "response_profile": (
            {
                "avg_len": baseline.response_profile.avg_len,
                "std_len": baseline.response_profile.std_len,
                "vocab_size": len(baseline.response_profile.vocab),
            }
            if baseline.response_profile
            else None
        ),
    }
    if existing:
        existing.parameters = parameters
        existing.is_active = True
        db.commit()
        db.refresh(existing)
        return existing

    profile = BaselineProfile(
        agent_id=agent.id,
        profile_name=profile_name,
        description="Auto-built behavioral baseline",
        window_size=baseline.n_runs,
        parameters=parameters,
        is_active=True,
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


def load_baseline(profile: BaselineProfile) -> Baseline:
    hex_blob = profile.parameters["pickle_hex"]
    return pickle.loads(bytes.fromhex(hex_blob))


def get_active_baseline_profile(db: DBSession, agent: Agent) -> BaselineProfile | None:
    return (
        db.query(BaselineProfile)
        .filter(BaselineProfile.agent_id == agent.id, BaselineProfile.is_active.is_(True))
        .order_by(BaselineProfile.updated_at.desc())
        .first()
    )


def record_anomaly_result(
    db: DBSession,
    session: SessionModel,
    baseline_profile: BaselineProfile | None,
    result: ScoreResult,
) -> AnomalyResult:
    severity = "high" if result.score >= 80 else "medium" if result.flagged else "low"
    anomaly = AnomalyResult(
        session_id=session.id,
        baseline_profile_id=baseline_profile.id if baseline_profile else None,
        detected_at=datetime.now(timezone.utc),
        severity=severity,
        score=result.score,
        anomaly_type="prompt_injection_behavioral",
        summary=result.explanation[:250],
        details={
            "anomaly_score": result.anomaly_score,
            "injection_score": result.injection_score,
            "z_component": result.z_component,
            "novelty_component": result.novelty_component,
            "sensitive_bonus": result.sensitive_bonus,
            "param_pattern_component": result.param_pattern_component,
            "novel_bigrams": [list(b) for b in result.novel_bigrams],
            "novel_param_signatures": result.novel_param_signatures,
        },
        is_resolved=not result.flagged,
    )
    db.add(anomaly)
    db.commit()
    db.refresh(anomaly)

    if result.flagged:
        alert = Alert(
            anomaly_result_id=anomaly.id,
            alert_type="behavioral_anomaly",
            destination="audit_log",
            payload=result.explanation,
            sent_at=datetime.now(timezone.utc),
            status="sent",
        )
        db.add(alert)
        db.commit()

    return anomaly


def record_session_accumulator_alert(
    db: DBSession,
    session: SessionModel,
    baseline_profile: BaselineProfile | None,
    session_key: str,
    session_total: float,
    turns: int,
    per_turn_threshold: float,
) -> AnomalyResult:
    """Bonus requirement: cumulative suspicion across a session's turns can
    trigger an alert even when no single turn individually crossed the
    per-turn threshold. Reuses the existing AnomalyResult/Alert tables
    (anomaly_type distinguishes it from a per-turn behavioral anomaly) --
    no schema change required.
    """
    summary = (
        f"Session '{session_key}' cumulative suspicion {round(session_total, 2)} exceeded the "
        f"session threshold after {turns} turn(s); no single turn individually crossed "
        f"the per-turn threshold ({round(per_turn_threshold, 2)})."
    )
    anomaly = AnomalyResult(
        session_id=session.id,
        baseline_profile_id=baseline_profile.id if baseline_profile else None,
        detected_at=datetime.now(timezone.utc),
        severity="high",
        score=min(session_total, 9999.9999),
        anomaly_type="session_suspicion_accumulation",
        summary=summary[:250],
        details={"session_key": session_key, "session_total": session_total, "turns": turns},
        is_resolved=False,
    )
    db.add(anomaly)
    db.commit()
    db.refresh(anomaly)

    alert = Alert(
        anomaly_result_id=anomaly.id,
        alert_type="session_suspicion_accumulation",
        destination="audit_log",
        payload=summary,
        sent_at=datetime.now(timezone.utc),
        status="sent",
    )
    db.add(alert)
    db.commit()

    return anomaly


# ---------------------------------------------------------------------------
# Human Approval Policy Layer -- persistence
# ---------------------------------------------------------------------------

def get_session_by_id(db: DBSession, session_id: int) -> SessionModel | None:
    return db.query(SessionModel).filter(SessionModel.id == session_id).one_or_none()


def record_tool_execution_event(db: DBSession, session: SessionModel, call, source: str) -> None:
    """Logs the moment a previously-*held* sensitive call actually runs
    (post-approval), distinct from the original `tool_call`/`sensitive_call`
    event already written when the plan was made (see `record_session`) --
    part of the audit trail requirement ("execution resumed")."""
    event = ToolEvent(
        session_id=session.id,
        tool_name=call.tool_name,
        event_type="sensitive_call_executed",
        occurred_at=datetime.now(timezone.utc),
        payload={"params": call.params, "output": call.output},
        source=source,
    )
    db.add(event)
    db.commit()


def create_approval_request(
    db: DBSession,
    session: SessionModel,
    anomaly: AnomalyResult | None,
    agent_name: str,
    trace: Trace,
    pending_plan: list[dict],
    decision: PolicyDecision,
    score: float,
    threshold: float,
) -> ApprovalRequest:
    """Persists a paused, approval-gated run. This is the durable record
    the audit trail requirement asks for -- "approval requested" is logged
    both as the row itself and as a linked Alert."""
    now = datetime.now(timezone.utc)
    request = ApprovalRequest(
        approval_token=uuid.uuid4().hex,
        session_id=session.id,
        anomaly_result_id=anomaly.id if anomaly else None,
        agent_name=agent_name,
        prompt=trace.prompt,
        provider=trace.provider,
        poison_payload=trace.poison_payload,
        risk_level=decision.risk_level.value,
        score=score,
        threshold=threshold,
        planned_tools=decision.planned_tools,
        sensitive_tools=decision.sensitive_tools,
        reasons=decision.reasons,
        pending_plan=pending_plan,
        partial_response_text=trace.response_text,
        status="pending",
        requested_at=now,
    )
    db.add(request)
    db.commit()
    db.refresh(request)

    if request.anomaly_result_id:
        db.add(
            Alert(
                anomaly_result_id=request.anomaly_result_id,
                alert_type="approval_requested",
                destination="audit_log",
                payload=f"Approval requested (token={request.approval_token}, risk={decision.risk_level.value}, "
                        f"sensitive_tools={decision.sensitive_tools}).",
                sent_at=now,
                status="sent",
            )
        )
        db.commit()

    return request


def get_approval_request_by_token(db: DBSession, approval_token: str) -> ApprovalRequest | None:
    return db.query(ApprovalRequest).filter(ApprovalRequest.approval_token == approval_token).one_or_none()


def resolve_approval(
    db: DBSession,
    request: ApprovalRequest,
    approved: bool,
    actor: str | None,
    final_response: str,
) -> ApprovalRequest:
    """Marks an ApprovalRequest resolved and appends the corresponding
    audit-trail Alert ("execution resumed" / "blocked execution"). If
    rejected, also blocks the originating session and files an incident
    record (a durable AnomalyResult), per the spec."""
    now = datetime.now(timezone.utc)
    request.status = "approved" if approved else "rejected"
    request.resolved_at = now
    request.approved_by = actor
    request.final_response = final_response
    db.commit()
    db.refresh(request)

    if request.anomaly_result_id:
        db.add(
            Alert(
                anomaly_result_id=request.anomaly_result_id,
                alert_type="execution_resumed" if approved else "blocked_execution",
                destination="audit_log",
                payload=f"Approval {request.status} (token={request.approval_token}, actor={actor or 'unspecified'}).",
                sent_at=now,
                status="sent",
            )
        )
        db.commit()

    session = get_session_by_id(db, request.session_id)
    if session is not None:
        if approved:
            session.status = "completed"
        else:
            session.status = "blocked"
            incident = AnomalyResult(
                session_id=session.id,
                baseline_profile_id=None,
                detected_at=now,
                severity="high",
                score=float(request.score),
                anomaly_type="approval_rejected_incident",
                summary=(
                    f"Sensitive action(s) {', '.join(request.sensitive_tools) or 'n/a'} were blocked by human "
                    f"reviewer for session {session.id} (approval_token={request.approval_token})."
                )[:250],
                details={
                    "approval_token": request.approval_token,
                    "planned_tools": request.planned_tools,
                    "sensitive_tools": request.sensitive_tools,
                    "reasons": request.reasons,
                    "rejected_by": actor,
                },
                is_resolved=True,
            )
            db.add(incident)
        db.commit()

    return request
