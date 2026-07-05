from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.deps import get_db
from api.schemas import BaselineBuildRequest, BaselineBuildResponse
from app.agent import run_agent
from app.detector import build_baseline
from app.repository import get_or_create_agent, persist_baseline
from app.scenarios import NORMAL_SCENARIOS

router = APIRouter(prefix="/baseline", tags=["baseline"])
@router.get("")
async def get_baseline(agent_name: str = "support-agent-v1", db: Session = Depends(get_db)):
    from app.repository import get_active_baseline_profile, get_or_create_agent
    agent = get_or_create_agent(db, agent_name)
    profile = get_active_baseline_profile(db, agent)
    if not profile or not profile.parameters:
        return {
            "exists": False,
            "n_runs": 0,
            "threshold": None,
            "tool_frequency": {},
            "top_sequences": [],
            "transitions": {},
            "avg_trace_length": 0.0,
            "known_param_signatures": {},
            "avg_tool_frequency": {},
            "response_profile": None,
        }
    return {
        "exists": True,
        "n_runs": profile.parameters.get("n_runs", 0),
        "threshold": profile.parameters.get("threshold"),
        "tool_frequency": profile.parameters.get("tool_frequency", {}),
        "top_sequences": profile.parameters.get("top_sequences", []),
        "transitions": profile.parameters.get("transitions", {}),
        "avg_trace_length": profile.parameters.get("avg_trace_length", 0.0),
        "known_param_signatures": profile.parameters.get("known_param_signatures", {}),
        "avg_tool_frequency": profile.parameters.get("avg_tool_frequency", {}),
        "response_profile": profile.parameters.get("response_profile"),
    }

@router.post("/build", response_model=BaselineBuildResponse)
async def build(req: BaselineBuildRequest, db: Session = Depends(get_db)) -> BaselineBuildResponse:
    from app.repository import record_session
    from core.logger import get_logger

    logger = get_logger(__name__)
    agent = get_or_create_agent(db, req.agent_name)

    # Use the FULL normal-scenario bank, deterministically, every time.
    # Randomly subsetting scenarios (as this endpoint used to do) makes the
    # baseline non-reproducible: whichever combinations of tools happened
    # not to be sampled on a given build become "novel" and get flagged the
    # next time a legitimate prompt exercises them, producing inconsistent,
    # seemingly-random scores across builds. A behavioral baseline should
    # represent the *whole* known-normal population every time it's built.
    logger.info(f"Building baseline from all {len(NORMAL_SCENARIOS)} normal scenarios")

    traces = []
    for s in NORMAL_SCENARIOS:
        trace = run_agent(s.prompt, poison_payload=None, use_real_llm=req.use_real_llm)
        # Still persisted for audit/history, but the baseline itself is
        # built directly from these freshly generated, fully-populated
        # traces below -- not from a lossy DB round-trip reconstruction
        # (which previously dropped response_text entirely, corrupting the
        # response-consistency baseline for every subsequent live score).
        record_session(db, agent, trace)
        traces.append(trace)

    logger.info(f"Generated {len(traces)} fresh traces for baseline aggregation")

    baseline = build_baseline(traces)
    persist_baseline(db, agent, req.profile_name, baseline)

    return BaselineBuildResponse(
        agent_name=req.agent_name,
        profile_name=req.profile_name,
        n_runs=baseline.n_runs,
        threshold=round(baseline.threshold, 2),
        self_scores=[round(s, 2) for s in baseline.self_scores],
        means=baseline.means,
        stds=baseline.stds,
        known_bigram_count=len(baseline.known_bigrams),
    )
