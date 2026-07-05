from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.deps import get_db
from api.schemas import RunSummary
from models.anomaly_result import AnomalyResult
from models.session import Session as SessionModel
from models.tool_event import ToolEvent

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("", response_model=list[RunSummary])
async def list_runs(limit: int = 50, db: Session = Depends(get_db)) -> list[RunSummary]:
    sessions = db.query(SessionModel).order_by(SessionModel.id.desc()).limit(limit).all()
    out: list[RunSummary] = []
    for s in sessions:
        events = db.query(ToolEvent).filter(ToolEvent.session_id == s.id).order_by(ToolEvent.id.asc()).all()
        anomaly = (
            db.query(AnomalyResult)
            .filter(AnomalyResult.session_id == s.id)
            .order_by(AnomalyResult.id.desc())
            .first()
        )
        out.append(
            RunSummary(
                session_id=s.id,
                session_name=s.session_name,
                prompt=(s.metadata_payload or {}).get("prompt"),
                tool_sequence=[e.tool_name for e in events],
                score=float(anomaly.score) if anomaly else None,
                flagged=(not anomaly.is_resolved) if anomaly else None,
                severity=anomaly.severity if anomaly else None,
                started_at=s.started_at.isoformat() if s.started_at else None,
            )
        )
    return out
