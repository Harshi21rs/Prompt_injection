from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

import models
from api.router import router
from core.config import settings
from core.database import Base, engine
from core.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Behavioral anomaly detector",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

from pydantic import BaseModel
from typing import Optional

class SimulateRequest(BaseModel):
    prompt: str

@app.post("/simulate")
async def simulate(req: SimulateRequest):
    from app.agent import run_agent
    from core.config import settings
    logger.info(f"Received prompt for simulation: {req.prompt}")
    trace = run_agent(req.prompt, use_real_llm=True)
    logger.info(f"Simulation trace completed. Provider: {trace.provider}, Tools: {trace.tool_sequence}")
    return {
        "session_id": trace.trace_id,
        "provider": trace.provider,
        "model": settings.groq_model,
        "tool_sequence": trace.tool_sequence,
        "response": trace.response_text
    }


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting Behavioral Anomaly Detector application")

    Base.metadata.create_all(bind=engine)

    logger.info("Database tables verified/created successfully")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("Shutting down Behavioral Anomaly Detector application")