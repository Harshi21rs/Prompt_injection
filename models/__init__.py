from core.database import Base, engine

from .agent import Agent
from .alert import Alert
from .anomaly_result import AnomalyResult
from .approval_request import ApprovalRequest
from .baseline_profile import BaselineProfile
from .session import Session
from .tool_event import ToolEvent

__all__ = [
    "Base",
    "engine",
    "Agent",
    "Session",
    "ToolEvent",
    "BaselineProfile",
    "AnomalyResult",
    "Alert",
    "ApprovalRequest",
    "create_tables",
]


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
