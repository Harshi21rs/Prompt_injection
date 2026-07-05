"""
Intent inference -- context-aware detection support.

Before scoring a trace, the detector needs a cheap, deterministic guess at
*what the user actually asked for*, so it can tell "the agent used a tool
that has nothing to do with the request" apart from "the agent used a
slightly-unusual-but-plausible tool for this request".

This module is intentionally tiny and rule-based (keyword -> expected tool
set), mirroring the same keyword table the offline simulator in
`app.agent` already uses to *plan* tool calls -- here it is reused to
*evaluate* tool calls instead. No new dependency, no ML model, no change
to the agent harness's public behavior.

Two things come out of intent inference for a trace:
  - `expected_tools`: the set of tools a normal, on-task agent would be
    expected to call for this prompt.
  - `intent_label`: a short human-readable tag for explanations/audit logs.

An empty `expected_tools` set means "no strong expectation" (the prompt
didn't match any known task keyword) -- in that case goal-deviation /
tool-omission scoring is skipped for that trace, and detection falls back
to the other behavioral layers (sequence, transition, parameter, sensitive
usage), so we never penalize legitimate long-tail phrasing just because it
didn't match a keyword.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERN_CACHE: dict[str, "re.Pattern[str]"] = {}


def _keyword_matches(keyword: str, lower_prompt: str) -> bool:
    """Whole-word match (mirrors app.agent._keyword_matches) so intent
    inference and the offline planner never disagree about whether a
    keyword "hit" -- a naive substring check would match "order" inside
    "orders"/"reorder" and so on, which is fine on its own, but any
    divergence between this module and the planner's own matching would
    make the goal-deviation layer penalize prompts for a mismatch that
    doesn't actually exist."""
    pattern = _PATTERN_CACHE.get(keyword)
    if pattern is None:
        suffix = r"(?:s|es|ed|ing)?"
        if keyword == "email":
            pattern = re.compile(r"\bemail" + suffix + r"\b(?!\s+address)")
        else:
            pattern = re.compile(r"\b" + re.escape(keyword) + suffix + r"\b")
        _PATTERN_CACHE[keyword] = pattern
    return pattern.search(lower_prompt) is not None

# Keyword -> (intent label, expected tool(s)). Order matters only for the
# label chosen when multiple keywords match; expected tools are unioned
# across every keyword that matches the prompt, matching how the offline
# simulator in app.agent plans multiple actions for a compound request.
_INTENT_RULES: list[tuple[str, str, tuple[str, ...]]] = [
    ("order", "order_status_lookup", ("get_order_status",)),
    ("password", "password_help", ("search_kb",)),
    ("refund", "refund_policy_lookup", ("search_kb",)),
    ("shipping", "shipping_info_lookup", ("search_kb",)),
    ("runbook", "internal_runbook_lookup", ("read_internal_file",)),
    ("email", "send_email_request", ("send_email",)),
    ("update", "record_update_request", ("update_customer_record",)),
    ("warranty", "warranty_policy_lookup", ("search_kb",)),
    ("subscription", "subscription_help", ("search_kb",)),
]

# Fallback expectation when no keyword matches: a bare informational
# lookup, consistent with the offline simulator's own default plan.
_FALLBACK_LABEL = "general_kb_lookup"
_FALLBACK_TOOLS: tuple[str, ...] = ("search_kb",)


@dataclass
class Intent:
    label: str
    expected_tools: set[str]
    matched_keywords: list[str]


def infer_intent(prompt: str) -> Intent:
    """Infer the expected-tool footprint for a user prompt.

    Deterministic and side-effect-free so it can be called both while
    scoring a live trace and while building/validating the baseline.
    """
    lower = prompt.lower()
    matched: list[str] = []
    expected: set[str] = set()
    label = None

    for keyword, intent_label, tools in _INTENT_RULES:
        if _keyword_matches(keyword, lower):
            matched.append(keyword)
            expected |= set(tools)
            if label is None:
                label = intent_label

    if not matched:
        return Intent(label=_FALLBACK_LABEL, expected_tools=set(_FALLBACK_TOOLS), matched_keywords=[])

    return Intent(label=label or _FALLBACK_LABEL, expected_tools=expected, matched_keywords=matched)
