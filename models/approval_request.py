"""
ApprovalRequest -- durable record of a paused, human-approval-gated agent
turn (see `app/policy.py` and `POST /agent/execute`, `/approve`, `/reject`).

This is a genuinely new table, not a change to any existing table's
schema: nothing about Agent/Session/ToolEvent/BaselineProfile/AnomalyResult/
Alert is altered. It's created automatically the same way every other model
here is (`Base.metadata.create_all` on startup) -- no migration step needed.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, JSON, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class ApprovalRequest(Base):
    __tablename__ = "approval_requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    approval_token: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    anomaly_result_id: Mapped[int | None] = mapped_column(
        ForeignKey("anomaly_results.id", ondelete="SET NULL"), nullable=True, index=True
    )
    agent_name: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="offline")
    poison_payload: Mapped[str | None] = mapped_column(String(64), nullable=True)

    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    score: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    threshold: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)

    # Everything the agent planned to do (tool names) vs. specifically the
    # sensitive subset that was held rather than dispatched.
    planned_tools: Mapped[list] = mapped_column(JSON, nullable=False)
    sensitive_tools: Mapped[list] = mapped_column(JSON, nullable=False)
    reasons: Mapped[list] = mapped_column(JSON, nullable=False)

    # The actual held execution plan: [{"tool_name": ..., "params": {...}}].
    # Replayed verbatim by POST /approve via app.agent.finalize_pending_call.
    pending_plan: Mapped[list] = mapped_column(JSON, nullable=False)
    partial_response_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")  # pending|approved|rejected
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    final_response: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_approval_requests_session_status", "session_id", "status"),
    )
