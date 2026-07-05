"""
Human Approval Policy Layer.

A deterministic policy layer, not a second LLM agent. It sits strictly
between the (unmodified) behavioral anomaly detector and tool execution:
it only ever reads a `ScoreResult` that `app.detector.score_and_explain`
already produced and turns it into an approval decision. It never calls
Groq, OpenAI, or any other model.

    - LOW (< 15) -> approval_required=False. Executes immediately.
    - MEDIUM (>= 15, < threshold) -> approval_required=False. Executes immediately, logged with warning.
    - HIGH (>= threshold) -> approval_required=True. Paused for approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.detector import ScoreResult


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# A flagged run scoring this many times its own threshold, or one that
# specifically tripped the sensitive-tool-usage layer, is treated as HIGH
# risk; every other flagged run is MEDIUM. LOW is only ever used for runs
# that weren't flagged at all (and therefore don't require approval).
_HIGH_RISK_SCORE_RATIO = 1.6

# Internal detector layer name -> short, user-safe phrase. Deliberately
# generic: no z-scores, weights, or layer internals leak into what the
# person approving/rejecting a run sees.
_LAYER_EXPLANATIONS: dict[str, str] = {
    "sensitive_tool_usage": "Unexpected sensitive tool invocation",
    "goal_deviation": "Goal deviation from the requested task",
    "param_pattern_deviation": "Unusual parameter pattern",
    "sequence_deviation": "Novel tool transition",
    "transition_deviation": "Novel tool transition",
    "unexpected_tool_insertion": "New, previously unseen tool used",
    "tool_omission": "Expected step was skipped",
    "tool_frequency_deviation": "Unusual tool usage frequency",
    "response_consistency": "Unusual response content",
    "trace_shape": "High anomaly score",
}
_DEFAULT_REASON = "Anomalous agent behavior detected"


@dataclass
class PolicyDecision:
    approval_required: bool
    risk_level: RiskLevel
    reasons: list[str] = field(default_factory=list)
    sensitive_tools: list[str] = field(default_factory=list)
    planned_tools: list[str] = field(default_factory=list)


def _risk_level(result: ScoreResult, threshold: float) -> RiskLevel:
    if result.score < 15:
        return RiskLevel.LOW
    elif result.score >= threshold:
        return RiskLevel.HIGH
    return RiskLevel.MEDIUM


def _friendly_reasons(result: ScoreResult, limit: int = 5) -> list[str]:
    """Ranked, deduplicated, user-facing reasons -- the compact form of the
    detector's own ranked explanation, translated out of internal layer
    names into phrases safe to show an end user or approver."""
    reasons: list[str] = []
    for layer in result.layers:  # already sorted by contribution, highest first
        if layer.score <= 0:
            continue
        phrase = _LAYER_EXPLANATIONS.get(layer.name, _DEFAULT_REASON)
        if phrase not in reasons:
            reasons.append(phrase)
        if len(reasons) >= limit:
            break
    return reasons or [_DEFAULT_REASON]


def evaluate(
    result: ScoreResult,
    threshold: float,
    tool_sequence: list[str],
    sensitive_tools: list[str],
) -> PolicyDecision:
    """Pure function: detector output -> approval decision. Never calls an
    LLM; never mutates anything."""
    risk = _risk_level(result, threshold)
    
    return PolicyDecision(
        approval_required=(risk == RiskLevel.HIGH),
        risk_level=risk,
        reasons=_friendly_reasons(result) if risk != RiskLevel.LOW else [],
        sensitive_tools=sensitive_tools,
        planned_tools=tool_sequence,
    )
