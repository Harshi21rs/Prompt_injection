from __future__ import annotations

from pydantic import BaseModel


class BaselineBuildRequest(BaseModel):
    agent_name: str = "support-agent-v1"
    use_real_llm: bool = False
    profile_name: str = "default"


class BaselineBuildResponse(BaseModel):
    agent_name: str
    profile_name: str
    n_runs: int
    threshold: float
    self_scores: list[float]
    means: dict[str, float]
    stds: dict[str, float]
    known_bigram_count: int


class ScoreRequest(BaseModel):
    agent_name: str = "support-agent-v1"
    prompt: str
    use_real_llm: bool = False
    session_key: str | None = None  # for the bonus suspicion-accumulator


class LayerScoreOut(BaseModel):
    name: str
    score: float
    weight: float
    reasons: list[str]


class ScoreResponse(BaseModel):
    trace_id: str
    score: float
    anomaly_score: float | None = None
    injection_score: float | None = None
    flagged: bool
    threshold: float
    z_component: float
    novelty_component: float
    sensitive_bonus: float
    param_pattern_component: float | None = None
    novel_bigrams: list[list[str]]
    novel_param_signatures: list[str] | None = None
    explanation: str
    tool_sequence: list[str]
    session_total: float | None = None
    # Multi-layer behavioral analysis breakdown (ranked, highest first).
    layers: list[LayerScoreOut] | None = None
    # Context-aware detection: inferred task and the tools expected for it.
    intent_label: str | None = None
    expected_tools: list[str] | None = None
    # Session-level suspicion accumulator (bonus requirement).
    session_turns: int | None = None
    session_flagged: bool | None = None


class InjectionDemoResponse(BaseModel):
    results: list[ScoreResponse]


class RunSummary(BaseModel):
    session_id: int
    session_name: str
    prompt: str | None = None
    tool_sequence: list[str]
    score: float | None
    flagged: bool | None
    severity: str | None
    started_at: str | None = None


# ---------------------------------------------------------------------------
# Human Approval Policy Layer
# ---------------------------------------------------------------------------

class AgentExecuteRequest(BaseModel):
    agent_name: str = "support-agent-v1"
    prompt: str
    use_real_llm: bool = False
    session_key: str | None = None


class ApprovalRequiredOut(BaseModel):
    approval_required: bool = True
    risk_level: str
    session_id: int
    trace_id: str
    approval_token: str
    prompt: str
    score: float
    threshold: float
    planned_tools: list[str]
    sensitive_tools: list[str]
    reasons: list[str]
    intent_label: str | None = None
    expected_tools: list[str] | None = None


class AgentExecuteResult(BaseModel):
    approval_required: bool = False
    session_id: int
    trace_id: str
    prompt: str
    response: str
    tool_sequence: list[str]
    score: float
    flagged: bool
    risk_level: str
    intent_label: str | None = None
    expected_tools: list[str] | None = None


class ApproveRequest(BaseModel):
    approval_token: str
    approved_by: str | None = None


class RejectRequest(BaseModel):
    approval_token: str
    rejected_by: str | None = None


class ApprovalActionResult(BaseModel):
    status: str  # "approved" | "rejected"
    approval_token: str
    session_id: int
    response: str


class ApprovalStatusOut(BaseModel):
    approval_token: str
    status: str
    risk_level: str
    session_id: int
    planned_tools: list[str]
    sensitive_tools: list[str]
    reasons: list[str]
    requested_at: str
    resolved_at: str | None = None
    approved_by: str | None = None
