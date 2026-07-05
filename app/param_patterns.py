"""
Parameter-pattern fingerprinting.

Extends the behavioral fingerprint beyond "which tools, in what order"
to also cover "with what kind of parameters". This module is intentionally
separate from `app/tools.py` (tool implementations / schemas are untouched)
and from `app/detector.py` (sequence-level scoring is untouched) -- it is a
pure addition the detector composes with, not a redesign of either.

Design choice: we fingerprint the *shape* of a parameter, not its literal
value. E.g. for `send_email` we record the recipient's domain
("mailbox.example"), not the full address; for `update_customer_record`
we record which *field* was touched, not the value written. This lets a
baseline built from a modest number of runs generalize to legitimate
variation (different customer IDs, different order numbers) while still
flagging the kind of shift a hijacked agent introduces: a new outbound
domain, a new record field being written, a new file-path namespace, etc.

Nothing here inspects tool-output *text* for injection phrases -- it only
looks at the parameters the agent *chose to call a tool with*, which is
part of the agent's observable behavior (its tool trace), consistent with
the detector's "behavior, not prompt syntax" design.
"""

from __future__ import annotations

from typing import Any


def _domain_of(value: Any) -> str:
    if not isinstance(value, str) or "@" not in value:
        return "no-domain"
    return value.rsplit("@", 1)[-1].strip().lower() or "no-domain"


def _path_prefix(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "no-path"
    value = value.strip("/")
    return value.split("/", 1)[0].lower() if value else "no-path"


def _bucket_len(value: Any) -> str:
    n = len(str(value))
    if n <= 20:
        return "short"
    if n <= 60:
        return "medium"
    return "long"


# One pluggable rule per tool: maps a tool's raw params to a small set of
# discrete "shape" tokens. Unknown tools fall back to a generic rule below.
_SIGNATURE_RULES = {
    "send_email": lambda p: f"to_domain={_domain_of(p.get('to'))}",
    "update_customer_record": lambda p: f"field={str(p.get('field', 'unknown')).lower()}",
    "read_internal_file": lambda p: f"path_prefix={_path_prefix(p.get('file_path'))}",
    "get_order_status": lambda p: "order_lookup",
    "search_kb": lambda p: f"query_len={_bucket_len(p.get('query', ''))}",
}


def param_signature(tool_name: str, params: dict[str, Any]) -> str:
    """Collapse a tool call's parameters into a small, comparable token.

    Same idea as a tool-call bigram, but for *parameter shape* instead of
    *call order*: a baseline accumulates the set of signatures seen per
    tool, and a new call whose signature was never seen for that tool is a
    parameter-pattern anomaly.
    """
    rule = _SIGNATURE_RULES.get(tool_name)
    if rule is not None:
        try:
            return rule(params or {})
        except Exception:
            pass
    # Generic fallback for tools with no bespoke rule: the sorted set of
    # parameter keys used (catches a tool being called with an unexpected
    # / extra parameter shape).
    keys = ",".join(sorted((params or {}).keys()))
    return f"keys={keys or 'none'}"


def build_param_signature_index(traces: list) -> dict[str, set[str]]:
    """Aggregate {tool_name -> {signatures seen in these traces}}."""
    index: dict[str, set[str]] = {}
    for trace in traces:
        for call in trace.tool_calls:
            sig = param_signature(call.tool_name, call.params)
            index.setdefault(call.tool_name, set()).add(sig)
    return index
