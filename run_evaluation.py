"""
Runs calibration validation + the full evaluation, prints a summary, and
writes the evaluation report (Markdown + JSON) to disk.

Run with:  python run_evaluation.py
Exit code is non-zero if any success criterion fails, so this doubles as a
CI-style calibration/regression check.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.evaluation import render_markdown, run_evaluation


def main() -> int:
    report = run_evaluation(use_real_llm=False)

    md = render_markdown(report)
    md_path = ROOT / "evaluation_report.md"
    json_path = ROOT / "evaluation_report.json"
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(report.as_json(), encoding="utf-8")

    print(md)
    print(f"\nWritten: {md_path}")
    print(f"Written: {json_path}")

    ok = report.calibration_ok and report.all_injections_detected and report.separation_ok
    if not ok:
        print("\nFAILURE: one or more success criteria did not pass.", file=sys.stderr)
        return 1
    print("\nAll success criteria PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
