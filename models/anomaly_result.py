from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, JSON, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


class AnomalyResult(Base):
    __tablename__ = "anomaly_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    baseline_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("baseline_profiles.id", ondelete="SET NULL"), nullable=True, index=True
    )
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    score: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    anomaly_type: Mapped[str] = mapped_column(String(128), nullable=False)
    summary: Mapped[str] = mapped_column(String(256), nullable=False)
    details: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    is_resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    session: Mapped[Session] = relationship("Session", back_populates="anomaly_results")
    baseline_profile: Mapped[BaselineProfile | None] = relationship(
        "BaselineProfile", back_populates="anomaly_results"
    )
    alerts: Mapped[list[Alert]] = relationship(
        "Alert",
        back_populates="anomaly_result",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_anomaly_results_session_detected", "session_id", "detected_at"),
    )
