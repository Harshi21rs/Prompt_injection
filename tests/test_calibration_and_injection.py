"""
Regression tests locking in the challenge's success criteria:

  - Baseline correctly established from >= 20 normal runs.
  - All 3 injected runs score significantly above the normal range.
  - >= 2 of the normal runs score moderately higher but stay below threshold
    (calibration, not a binary check).
  - Detection happens within one agent turn.
  - Parameter-pattern comparison (behavioral fingerprint enhancement)
    contributes to injection detection, not just tool sequence.

Run with:  pytest tests/test_calibration_and_injection.py -v
(or just: python -m pytest tests/ -v)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation import MIN_NORMAL_RUNS, run_evaluation
from app.scenarios import INJECTION_SCENARIOS, NORMAL_SCENARIOS


def test_scenario_bank_sizes():
    assert len(NORMAL_SCENARIOS) >= MIN_NORMAL_RUNS
    assert len(INJECTION_SCENARIOS) == 3


def test_calibration_and_detection():
    report = run_evaluation(use_real_llm=False)

    # Baseline correctly established from >= 20 normal runs.
    assert report.n_normal_runs >= MIN_NORMAL_RUNS

    # All 3 injected runs score significantly above the normal range.
    assert report.all_injections_detected
    assert report.separation_ok
    assert report.injection_score_min > report.normal_score_max

    # >= 2 normal runs moderately elevated but still below threshold.
    assert report.elevated_but_passing_count >= 2

    # No normal run should be flagged (zero false positives at threshold).
    assert all(not r.flagged for r in report.normal_results)

    # Detection within one agent turn for every injection scenario.
    assert all(r.agent_turns == 1 and r.detected_within_one_turn for r in report.injection_results)


def test_parameter_pattern_contributes_to_detection():
    report = run_evaluation(use_real_llm=False)
    # At least one injection scenario's detection should be partly explained
    # by a novel parameter *pattern* (not just tool sequence/order), proving
    # the behavioral fingerprint enhancement is load-bearing.
    flagged_with_param_signal = [
        r for r in report.injection_results if "Parameter usage never seen" in r.explanation
    ]
    assert len(flagged_with_param_signal) >= 1


if __name__ == "__main__":
    test_scenario_bank_sizes()
    test_calibration_and_detection()
    test_parameter_pattern_contributes_to_detection()
    print("All regression tests passed.")
