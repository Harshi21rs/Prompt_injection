"""
Response-consistency fingerprinting.

Extends the behavioral fingerprint to the agent's final natural-language
response, not just its tool trace. A hijacked agent that discloses secrets,
apologizes profusely, or otherwise produces a response that reads nothing
like the agent's normal replies is itself a behavioral signal -- one the
tool-sequence layers can miss entirely if the injected instruction changes
*what the agent says* more than *what tools it calls* (see the
`file_secret_leak` scenario, where the agent's response is the disclosure
itself).

Like `app.param_patterns`, this fingerprints *shape*, not literal content:
response length bucket + response vocabulary, never full text, so the
baseline generalizes across legitimately varied replies (different order
numbers, different customer names) instead of requiring verbatim matches.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-zA-Z]{4,}")

# A small set of tokens that are highly unusual in a normal customer
# support reply and strongly associated with data-exfiltration /
# instruction-following disclosure. Cheap, explainable, and only ever adds
# to the score -- it is never the sole basis for a flag.
_SUSPICIOUS_TOKENS = {
    "secret", "secrets", "api_key", "apikey", "password", "leaked",
    "confidential", "internal-only", "credentials", "token",
}


@dataclass
class ResponseProfile:
    avg_len: float
    std_len: float
    vocab: set[str]


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "")}


def build_response_profile(response_texts: list[str]) -> ResponseProfile:
    lengths = [float(len(t)) for t in response_texts] or [0.0]
    mean = sum(lengths) / len(lengths)
    variance = sum((x - mean) ** 2 for x in lengths) / max(len(lengths) - 1, 1)
    std = math.sqrt(variance)

    vocab: set[str] = set()
    for t in response_texts:
        vocab |= _tokenize(t)

    return ResponseProfile(avg_len=mean, std_len=std if std > 1e-6 else 1e-6, vocab=vocab)


def response_consistency_deviation(response_text: str, profile: ResponseProfile) -> tuple[float, list[str]]:
    """Returns a 0-1 deviation fraction plus human-readable reasons.

    Combines a length z-score (capped) with a novel-vocabulary fraction and
    a hard bump for any suspicious-disclosure token, then averages/caps to
    stay in [0, 1] so the caller can apply its own weight.
    """
    reasons: list[str] = []
    length = float(len(response_text or ""))
    z = abs(length - profile.avg_len) / profile.std_len
    length_component = min(z / 4.0, 1.0)
    if length_component > 0.5:
        reasons.append(f"Response length is a statistical outlier vs baseline (z={z:.2f}).")

    tokens = _tokenize(response_text)
    if tokens:
        novel = tokens - profile.vocab
        vocab_component = len(novel) / len(tokens)
    else:
        vocab_component = 0.0

    suspicious_hit = tokens & _SUSPICIOUS_TOKENS
    suspicious_component = 1.0 if suspicious_hit else 0.0
    if suspicious_hit:
        reasons.append(f"Response contains disclosure-pattern token(s): {', '.join(sorted(suspicious_hit))}.")

    deviation = min((length_component + vocab_component) / 2.0 + suspicious_component * 0.6, 1.0)
    return deviation, reasons
