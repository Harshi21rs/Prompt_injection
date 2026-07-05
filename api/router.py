from fastapi import APIRouter

from .approval import router as approval_router
from .baseline import router as baseline_router
from .health import router as health_router
from .metrics import router as metrics_router
from .runs import router as runs_router
from .score import router as score_router

router = APIRouter()
router.include_router(health_router)
router.include_router(baseline_router)
router.include_router(score_router)
router.include_router(runs_router)
router.include_router(approval_router)
router.include_router(metrics_router)
