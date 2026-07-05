from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, JSON, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


class BaselineProfile(Base):
    __tablename__ = "baseline_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_id: Mapped[int] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    profile_name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(String(256), nullable=True)
    window_size: Mapped[int] = mapped_column(nullable=False, default=1)
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    agent: Mapped[Agent] = relationship("Agent", back_populates="baseline_profiles")
    anomaly_results: Mapped[list[AnomalyResult]] = relationship(
        "AnomalyResult",
        back_populates="baseline_profile",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint("agent_id", "profile_name", name="uq_baseline_agent_profile"),
        Index("ix_baseline_profiles_agent_active", "agent_id", "is_active"),
    )
