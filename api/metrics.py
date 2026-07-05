from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/metrics", tags=["metrics"])

_REPORT_PATH = Path(__file__).resolve().parent.parent / "evaluation_report.json"


@router.get("/evaluation")
async def get_evaluation_metrics():
    """Serves the real calibration/evaluation numbers produced by
    `run_evaluation.py` (precision/recall/F1 against the crafted injection
    scenarios + the normal-scenario bank), so the dashboard can show actual
    measured detector performance instead of hardcoded placeholder numbers.
    """
    if not _REPORT_PATH.exists():
        return {"available": False}
    try:
        data = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"available": False}
    return {
        "available": True,
        "generated_at": data.get("generated_at"),
        "threshold": data.get("threshold"),
        "n_normal_runs": data.get("n_normal_runs"),
        "precision": data.get("precision"),
        "recall": data.get("recall"),
        "f1_score": data.get("f1_score"),
        "true_positives": data.get("true_positives"),
        "false_positives": data.get("false_positives"),
        "true_negatives": data.get("true_negatives"),
        "false_negatives": data.get("false_negatives"),
        "normal_score_min": data.get("normal_score_min"),
        "normal_score_max": data.get("normal_score_max"),
        "injection_score_min": data.get("injection_score_min"),
        "injection_score_max": data.get("injection_score_max"),
        "separation_ok": data.get("separation_ok"),
        "all_injections_detected": data.get("all_injections_detected"),
    }
