"""
AWS Lambda entry point.

Wraps the FastAPI app with Mangum so the same application that runs
locally with Uvicorn also runs unchanged as a Lambda container image
behind API Gateway.
"""

import os

os.environ.setdefault("DATABASE_URL", os.environ.get("DATABASE_URL", "sqlite:////tmp/behavioral_anomaly_detector.db"))

from app.main import app  # noqa: E402

try:
    from mangum import Mangum

    handler = Mangum(app, lifespan="off")
except ImportError:
    # Mangum not available (e.g. during local unit tests): no-op
    handler = None  # type: ignore
