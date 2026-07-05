"""
Detector engine -- the core IP of the system.

Pure Python, no heavy ML / no GPU / no paid embedding API.

  1. Feature extraction    -- Trace -> numeric feature vector + bigram set
                               + per-tool parameter-pattern signatures
                               + inferred task intent
  2. Baseline construction  -- aggregate feature vectors/bigrams/parameter
                               signatures/response shape across normal runs
                               into per-feature mean/std, a "known
                               transitions" set (Markov-style), per-tool
                               call-frequency averages, a "known parameter
                               shapes per tool" set, and a response profile
  3. Self-calibrated threshold -- score the baseline against itself
                               (leave-one-out) and set the flagging
                               threshold from that distribution
                               (mean + k*std) instead of a hardcoded number
  4. Multi-layer anomaly scoring -- nine independent behavioral analyzers,
                               each scoring one dimension of behavior, are
                               combined into a single weighted 0-100 score:

       1. Tool frequency deviation        (per-tool call-rate vs baseline)
       2. Tool sequence deviation          (novel bigrams)
       3. Tool transition probabilities    (Markov-style transition rarity)
       4. Parameter pattern deviation      (param *shape* per tool)
       5. Sensitive tool usage             (novel path into a write/send tool)
       6. Goal / task deviation            (intent-inferred expected tools)
       7. Unexpected tool insertion        (a tool never seen in baseline at all)
       8. Tool omission                    (an intent-expected tool never called)
       9. Response consistency             (reply length/vocab/disclosure tokens)

     Every layer also contributes a ranked, human-readable reason so a
     flagged session can be explained, not just scored (see `explain`).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from app.agent import Trace
from app.intent import Intent, infer_intent
from app.param_patterns import build_param_signature_index, param_signature
from app.response_profile import ResponseProfile, build_response_profile, response_consistency_deviation

FEATURE_NAMES = ["call_count", "distinct_tool_count", "sensitive_count", "avg_param_size"]

# ---------------------------------------------------------------------------
# Multi-layer scoring weights. Each layer's *maximum possible* contribution
# to the final 0-100 score; they sum to 100 so the total score stays on the
# same 0-100 scale the rest of the system (thresholds, severities, the
# evaluation report) already assumes. Rebalancing these weights is how the
# detector's emphasis is tuned; the layer functions themselves never need
# to change to shift emphasis.
# ---------------------------------------------------------------------------
W_TRACE_SHAPE = 15.0        # legacy z-score layer (call_count/distinct/sensitive/param-size)
W_FREQ_DEVIATION = 10.0     # tool frequency deviation
W_SEQUENCE_DEVIATION = 10.0  # tool sequence deviation (novel bigrams)
W_TRANSITION_DEVIATION = 10.0  # Markov-style transition-probability deviation
W_PARAM_PATTERN = 12.0      # parameter pattern deviation
W_SENSITIVE_USAGE = 13.0    # sensitive tool usage
W_GOAL_DEVIATION = 12.0     # goal / task deviation (intent-aware)
W_TOOL_INSERTION = 10.0     # unexpected tool insertion (globally unseen tool)
W_TOOL_OMISSION = 5.0       # tool omission (expected tool never called)
W_RESPONSE_CONSISTENCY = 3.0  # response consistency

assert abs(
    (
        W_TRACE_SHAPE + W_FREQ_DEVIATION + W_SEQUENCE_DEVIATION + W_TRANSITION_DEVIATION
        + W_PARAM_PATTERN + W_SENSITIVE_USAGE + W_GOAL_DEVIATION + W_TOOL_INSERTION
        + W_TOOL_OMISSION + W_RESPONSE_CONSISTENCY
    )
    - 100.0
) < 1e-6

Z_CAP = 4.0  # cap individual |z| contributions so one wild feature can't dominate
TRANSITION_RARITY_FLOOR = 0.15  # a known transition rarer than this is "surprising"

DEFAULT_K = 1.8  # threshold = mean + K * std of baseline self-scores


@dataclass
class FeatureVector:
    call_count: float
    distinct_tool_count: float
    sensitive_count: float
    avg_param_size: float

    def as_dict(self) -> dict[str, float]:
        return {
            "call_count": self.call_count,
            "distinct_tool_count": self.distinct_tool_count,
            "sensitive_count": self.sensitive_count,
            "avg_param_size": self.avg_param_size,
        }


def extract_features(trace: Trace) -> FeatureVector:
    seq = trace.tool_sequence
    return FeatureVector(
        call_count=float(len(seq)),
        distinct_tool_count=float(len(set(seq))),
        sensitive_count=float(trace.sensitive_count),
        avg_param_size=float(trace.avg_param_size),
    )


@dataclass
class Baseline:
    means: dict[str, float]
    stds: dict[str, float]
    known_bigrams: set[tuple[str, str]]
    threshold: float
    self_scores: list[float] = field(default_factory=list)
    n_runs: int = 0
    tool_frequency: dict[str, int] = field(default_factory=dict)
    top_sequences: list[list[str]] = field(default_factory=list)
    transitions: dict[str, dict[str, float]] = field(default_factory=dict)
    avg_trace_length: float = 0.0
    known_param_signatures: dict[str, set[str]] = field(default_factory=dict)
    avg_tool_frequency: dict[str, float] = field(default_factory=dict)
    response_profile: ResponseProfile | None = None


def build_baseline(traces: list[Trace], k: float = DEFAULT_K) -> Baseline:
    if len(traces) < 2:
        raise ValueError("Baseline requires at least 2 runs")

    vectors = [extract_features(t) for t in traces]
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in FEATURE_NAMES:
        values = [getattr(v, name) for v in vectors]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / max(len(values) - 1, 1)
        std = math.sqrt(variance)
        means[name] = mean
        stds[name] = std if std > 1e-6 else 1e-6

    known_bigrams: set[tuple[str, str]] = set()
    for t in traces:
        known_bigrams |= t.bigrams

    tool_frequency: dict[str, int] = {}
    transitions_counts: dict[str, dict[str, int]] = {}
    total_length = 0
    sequences = []

    for t in traces:
        seq = t.tool_sequence
        sequences.append(tuple(seq))
        total_length += len(seq)

        for i, tool in enumerate(seq):
            tool_frequency[tool] = tool_frequency.get(tool, 0) + 1
            if i < len(seq) - 1:
                next_tool = seq[i + 1]
                transitions_counts.setdefault(tool, {})
                transitions_counts[tool][next_tool] = transitions_counts[tool].get(next_tool, 0) + 1

    transitions = {}
    for curr_tool, next_tools in transitions_counts.items():
        total_t = sum(next_tools.values())
        transitions[curr_tool] = {nt: count / total_t for nt, count in next_tools.items()}

    seq_counter = Counter(sequences)
    top_sequences = [list(seq) for seq, _ in seq_counter.most_common(10)]
    avg_trace_length = total_length / len(traces) if traces else 0.0
    known_param_signatures = build_param_signature_index(traces)
    avg_tool_frequency = {tool: count / len(traces) for tool, count in tool_frequency.items()}
    response_profile = build_response_profile([t.response_text for t in traces])

    baseline = Baseline(
        means=means,
        stds=stds,
        known_bigrams=known_bigrams,
        threshold=0.0,
        n_runs=len(traces),
        tool_frequency=tool_frequency,
        top_sequences=top_sequences,
        transitions=transitions,
        avg_trace_length=avg_trace_length,
        known_param_signatures=known_param_signatures,
        avg_tool_frequency=avg_tool_frequency,
        response_profile=response_profile,
    )

    self_scores = []
    for i, t in enumerate(traces):
        loo_traces = traces[:i] + traces[i + 1 :]
        loo_baseline = _build_loo_baseline(loo_traces)
        result = score_trace(t, loo_baseline)
        self_scores.append(result.score)

    mean_self = sum(self_scores) / len(self_scores)
    var_self = sum((s - mean_self) ** 2 for s in self_scores) / max(len(self_scores) - 1, 1)
    std_self = math.sqrt(var_self)

    baseline.self_scores = self_scores
    baseline.threshold = mean_self + k * std_self
    return baseline


def _build_loo_baseline(loo_traces: list[Trace]) -> Baseline:
    loo_vectors = [extract_features(x) for x in loo_traces]
    loo_means: dict[str, float] = {}
    loo_stds: dict[str, float] = {}
    for name in FEATURE_NAMES:
        values = [getattr(v, name) for v in loo_vectors]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / max(len(values) - 1, 1)
        std = math.sqrt(variance)
        loo_means[name] = mean
        loo_stds[name] = std if std > 1e-6 else 1e-6

    loo_bigrams: set[tuple[str, str]] = set()
    tool_frequency: dict[str, int] = {}
    transitions_counts: dict[str, dict[str, int]] = {}
    for x in loo_traces:
        loo_bigrams |= x.bigrams
        seq = x.tool_sequence
        for i, tool in enumerate(seq):
            tool_frequency[tool] = tool_frequency.get(tool, 0) + 1
            if i < len(seq) - 1:
                next_tool = seq[i + 1]
                transitions_counts.setdefault(tool, {})
                transitions_counts[tool][next_tool] = transitions_counts[tool].get(next_tool, 0) + 1
    transitions = {}
    for curr_tool, next_tools in transitions_counts.items():
        total_t = sum(next_tools.values())
        transitions[curr_tool] = {nt: count / total_t for nt, count in next_tools.items()}

    seq_counter = Counter(tuple(x.tool_sequence) for x in loo_traces)
    top_sequences = [list(seq) for seq, _ in seq_counter.most_common(10)]
    loo_param_signatures = build_param_signature_index(loo_traces)
    avg_tool_frequency = {tool: count / max(len(loo_traces), 1) for tool, count in tool_frequency.items()}
    response_profile = build_response_profile([x.response_text for x in loo_traces])

    return Baseline(
        means=loo_means,
        stds=loo_stds,
        known_bigrams=loo_bigrams,
        threshold=0.0,
        known_param_signatures=loo_param_signatures,
        tool_frequency=tool_frequency,
        top_sequences=top_sequences,
        transitions=transitions,
        avg_tool_frequency=avg_tool_frequency,
        response_profile=response_profile,
    )


@dataclass
class LayerScore:
    name: str
    score: float
    weight: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class ScoreResult:
    score: float
    anomaly_score: float
    injection_score: float
    z_component: float
    novelty_component: float
    sensitive_bonus: float
    param_pattern_component: float
    novel_bigrams: list[tuple[str, str]]
    novel_param_signatures: list[str] = field(default_factory=list)
    abnormal_behaviors: list[str] = field(default_factory=list)
    flagged: bool = False
    explanation: str = ""
    layers: list[LayerScore] = field(default_factory=list)
    intent_label: str = ""
    expected_tools: list[str] = field(default_factory=list)


def _trace_shape_layer(trace: Trace, baseline: Baseline) -> LayerScore:
    vector = extract_features(trace)
    z_values = []
    worst_feature = None
    worst_z = 0.0
    for name in FEATURE_NAMES:
        z = abs((getattr(vector, name) - baseline.means[name]) / baseline.stds[name])
        capped = min(z, Z_CAP)
        z_values.append(capped)
        if z > worst_z:
            worst_z = z
            worst_feature = name
    avg_z = sum(z_values) / len(z_values)
    score = min(avg_z / Z_CAP, 1.0) * W_TRACE_SHAPE
    reasons = []
    if score > W_TRACE_SHAPE * 0.6 and worst_feature:
        reasons.append(f"Numeric trace shape is a statistical outlier (feature={worst_feature}, z={worst_z:.2f}).")
    return LayerScore("trace_shape", round(score, 3), W_TRACE_SHAPE, reasons)


def _tool_frequency_layer(trace: Trace, baseline: Baseline) -> LayerScore:
    counts = Counter(trace.tool_sequence)
    if not counts or not baseline.avg_tool_frequency:
        return LayerScore("tool_frequency_deviation", 0.0, W_FREQ_DEVIATION, [])

    deviations = []
    worst = None
    worst_dev = 0.0
    for tool, count in counts.items():
        expected = baseline.avg_tool_frequency.get(tool, 0.0)
        dev = abs(count - expected) / (expected + 1.0)
        deviations.append(min(dev, 1.0))
        if dev > worst_dev:
            worst_dev = dev
            worst = tool
    fraction = sum(deviations) / len(deviations)
    score = fraction * W_FREQ_DEVIATION
    reasons = []
    if score > W_FREQ_DEVIATION * 0.5 and worst:
        expected = baseline.avg_tool_frequency.get(worst, 0.0)
        reasons.append(
            f"Call frequency for '{worst}' ({counts[worst]}x) deviates from baseline average ({expected:.2f}x)."
        )
    return LayerScore("tool_frequency_deviation", round(score, 3), W_FREQ_DEVIATION, reasons)


def _sequence_deviation_layer(trace: Trace, baseline: Baseline) -> tuple[LayerScore, list[tuple[str, str]]]:
    bigrams = trace.bigrams
    if not bigrams:
        return LayerScore("sequence_deviation", 0.0, W_SEQUENCE_DEVIATION, []), []

    novel = [b for b in bigrams if b not in baseline.known_bigrams]
    fraction_novel = len(novel) / len(bigrams)
    score = fraction_novel * W_SEQUENCE_DEVIATION
    reasons = []
    if novel:
        transitions_str = ", ".join(f"{a}->{b}" for a, b in novel)
        reasons.append(f"Novel tool-call transition(s) never seen in baseline: {transitions_str}.")
    return LayerScore("sequence_deviation", round(score, 3), W_SEQUENCE_DEVIATION, reasons), novel


def _transition_deviation_layer(trace: Trace, baseline: Baseline) -> LayerScore:
    bigrams = trace.bigrams
    if not bigrams:
        return LayerScore("transition_deviation", 0.0, W_TRANSITION_DEVIATION, [])

    penalty = 0.0
    reasons = []
    checked = 0
    for a, b in bigrams:
        if (a, b) in baseline.known_bigrams:
            checked += 1
            prob = baseline.transitions.get(a, {}).get(b, 0.0)
            if prob < TRANSITION_RARITY_FLOOR:
                rarity = (TRANSITION_RARITY_FLOOR - prob) / TRANSITION_RARITY_FLOOR
                penalty += rarity
                reasons.append(f"Abnormal transition probability ({a}->{b}, p={prob:.2f}).")
    fraction = penalty / checked if checked else 0.0
    score = min(fraction, 1.0) * W_TRANSITION_DEVIATION
    return LayerScore("transition_deviation", round(score, 3), W_TRANSITION_DEVIATION, reasons)


def _param_pattern_layer(trace: Trace, baseline: Baseline) -> tuple[LayerScore, list[str]]:
    known = baseline.known_param_signatures
    total_comparable = 0
    novel_count = 0
    novel_signatures: list[str] = []

    for call in trace.tool_calls:
        seen_for_tool = known.get(call.tool_name)
        if not seen_for_tool:
            continue
        total_comparable += 1
        sig = param_signature(call.tool_name, call.params)
        if sig not in seen_for_tool:
            novel_count += 1
            novel_signatures.append(f"{call.tool_name}:{sig}")

    if total_comparable == 0:
        return LayerScore("param_pattern_deviation", 0.0, W_PARAM_PATTERN, []), []
    fraction_novel = novel_count / total_comparable
    score = fraction_novel * W_PARAM_PATTERN
    reasons = []
    if novel_signatures:
        reasons.append(f"Parameter usage never seen in baseline for that tool: {', '.join(novel_signatures)}.")
    return LayerScore("param_pattern_deviation", round(score, 3), W_PARAM_PATTERN, reasons), novel_signatures


def _sensitive_usage_layer(trace: Trace, novel_bigrams: list[tuple[str, str]]) -> LayerScore:
    from app.tools import is_sensitive

    novel_tools_entered = {b[1] for b in novel_bigrams}
    sensitive_novel = {t for t in novel_tools_entered if is_sensitive(t)}
    if sensitive_novel:
        reasons = [
            "A novel transition led into a sensitive (write/send/disclose) tool call "
            f"({', '.join(sorted(sensitive_novel))}) -- high-signal hijack pattern."
        ]
        return LayerScore("sensitive_tool_usage", W_SENSITIVE_USAGE, W_SENSITIVE_USAGE, reasons)
    return LayerScore("sensitive_tool_usage", 0.0, W_SENSITIVE_USAGE, [])


def _goal_deviation_layer(trace: Trace, intent: Intent) -> LayerScore:
    if not intent.expected_tools:
        return LayerScore("goal_deviation", 0.0, W_GOAL_DEVIATION, [])

    actual = set(trace.tool_sequence)
    unexpected = actual - intent.expected_tools
    if not actual:
        return LayerScore("goal_deviation", 0.0, W_GOAL_DEVIATION, [])

    fraction = len(unexpected) / len(actual)
    score = fraction * W_GOAL_DEVIATION
    reasons = []
    if unexpected:
        reasons.append(
            f"Goal deviation: tool(s) {', '.join(sorted(unexpected))} do not match the inferred "
            f"task '{intent.label}' (expected: {', '.join(sorted(intent.expected_tools)) or 'none'})."
        )
    return LayerScore("goal_deviation", round(score, 3), W_GOAL_DEVIATION, reasons)


def _tool_insertion_layer(trace: Trace, baseline: Baseline) -> LayerScore:
    actual = set(trace.tool_sequence)
    if not actual:
        return LayerScore("unexpected_tool_insertion", 0.0, W_TOOL_INSERTION, [])

    unseen = {t for t in actual if baseline.tool_frequency.get(t, 0) == 0}
    if not unseen:
        return LayerScore("unexpected_tool_insertion", 0.0, W_TOOL_INSERTION, [])

    fraction = len(unseen) / len(actual)
    score = fraction * W_TOOL_INSERTION
    reasons = [f"New tool insertion: '{t}' was never called in any baseline run." for t in sorted(unseen)]
    return LayerScore("unexpected_tool_insertion", round(score, 3), W_TOOL_INSERTION, reasons)


def _tool_omission_layer(trace: Trace, intent: Intent) -> LayerScore:
    if not intent.expected_tools:
        return LayerScore("tool_omission", 0.0, W_TOOL_OMISSION, [])

    actual = set(trace.tool_sequence)
    missing = intent.expected_tools - actual
    if not missing:
        return LayerScore("tool_omission", 0.0, W_TOOL_OMISSION, [])

    fraction = len(missing) / len(intent.expected_tools)
    score = fraction * W_TOOL_OMISSION
    reasons = [f"Tool omission: expected tool(s) {', '.join(sorted(missing))} for task '{intent.label}' were never called."]
    return LayerScore("tool_omission", round(score, 3), W_TOOL_OMISSION, reasons)


def _response_consistency_layer(trace: Trace, baseline: Baseline) -> LayerScore:
    if baseline.response_profile is None:
        return LayerScore("response_consistency", 0.0, W_RESPONSE_CONSISTENCY, [])
    deviation, reasons = response_consistency_deviation(trace.response_text, baseline.response_profile)
    score = deviation * W_RESPONSE_CONSISTENCY
    return LayerScore("response_consistency", round(score, 3), W_RESPONSE_CONSISTENCY, reasons)


def score_trace(trace: Trace, baseline: Baseline) -> ScoreResult:
    intent = infer_intent(trace.prompt)

    trace_shape = _trace_shape_layer(trace, baseline)
    freq_layer = _tool_frequency_layer(trace, baseline)
    seq_layer, novel_bigrams = _sequence_deviation_layer(trace, baseline)
    transition_layer = _transition_deviation_layer(trace, baseline)
    param_layer, novel_param_signatures = _param_pattern_layer(trace, baseline)
    sensitive_layer = _sensitive_usage_layer(trace, novel_bigrams)
    goal_layer = _goal_deviation_layer(trace, intent)
    insertion_layer = _tool_insertion_layer(trace, baseline)
    omission_layer = _tool_omission_layer(trace, intent)
    response_layer = _response_consistency_layer(trace, baseline)

    layers = [
        trace_shape, freq_layer, seq_layer, transition_layer, param_layer,
        sensitive_layer, goal_layer, insertion_layer, omission_layer, response_layer,
    ]
    layers.sort(key=lambda l: l.score, reverse=True)

    total = sum(l.score for l in layers)
    score = min(total, 100.0)

    anomaly_score = trace_shape.score
    novelty_component = seq_layer.score + transition_layer.score
    sensitive_bonus = sensitive_layer.score
    param_pattern_component = param_layer.score
    injection_score = min(
        novelty_component + sensitive_bonus + param_pattern_component
        + goal_layer.score + insertion_layer.score + omission_layer.score + response_layer.score,
        100.0,
    )

    abnormal_behaviors: list[str] = []
    for l in layers:
        if l.name in ("goal_deviation",):
            abnormal_behaviors.extend(l.reasons)

    return ScoreResult(
        score=round(score, 2),
        anomaly_score=round(anomaly_score, 2),
        injection_score=round(injection_score, 2),
        z_component=round(trace_shape.score, 2),
        novelty_component=round(novelty_component, 2),
        sensitive_bonus=round(sensitive_bonus, 2),
        param_pattern_component=round(param_pattern_component, 2),
        novel_bigrams=novel_bigrams,
        novel_param_signatures=novel_param_signatures,
        abnormal_behaviors=abnormal_behaviors,
        layers=layers,
        intent_label=intent.label,
        expected_tools=sorted(intent.expected_tools),
    )


def explain(trace: Trace, result: ScoreResult, baseline: Baseline) -> str:
    if result.score < baseline.threshold:
        return f"Score {result.score} below threshold {round(baseline.threshold, 2)}; no anomaly."

    parts = [
        f"Flagged: score {result.score} >= threshold {round(baseline.threshold, 2)} "
        f"(inferred task: '{result.intent_label}')."
    ]
    ranked = [l for l in result.layers if l.score > 0]
    for rank, layer in enumerate(ranked, start=1):
        reason = " ".join(layer.reasons) if layer.reasons else "Contributing factor."
        parts.append(f"[{rank}] {layer.name} (+{round(layer.score, 2)}/{layer.weight}): {reason}")
    return " ".join(parts)


def score_and_explain(trace: Trace, baseline: Baseline) -> ScoreResult:
    result = score_trace(trace, baseline)
    result.flagged = result.score >= baseline.threshold
    result.explanation = explain(trace, result, baseline)
    return result


SESSION_THRESHOLD_MULTIPLIER = 1.3


@dataclass
class SessionAccumulator:
    session_id: str
    total: float = 0.0
    decay: float = 0.85
    turns: int = 0
    flagged: bool = False

    def add(self, score: float, per_turn_threshold: float | None = None) -> float:
        self.total = self.total * self.decay + score
        self.turns += 1
        if per_turn_threshold is not None:
            self.flagged = self.total >= per_turn_threshold * SESSION_THRESHOLD_MULTIPLIER
        return self.total
