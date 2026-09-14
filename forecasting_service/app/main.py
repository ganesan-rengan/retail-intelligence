"""Demand Forecasting Service.

Serves weekly demand forecasts for the top 50 products at a fixed 4-week horizon.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artifact import load_metrics, load_model  # noqa: E402

from forecasting_service.app.config import settings
from shared.database import engine

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Populated once at startup, reused for every request.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server boots, not on every request."""
    logger.info("Loading model artifact...")
    state["model"] = load_model()
    state["metrics"] = load_metrics()
    logger.info("Loaded model version %s", state["metrics"]["model_version"])
    yield
    state.clear()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Demand Forecasting Service",
    description="Weekly demand forecasts for retail SKUs at a 4-week horizon.",
    version="1.0.0",
    lifespan=lifespan,
)


def database_ok() -> bool:
    """Return True if the database answers a trivial query."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        return False


@app.get("/health")
def health() -> JSONResponse:
    """Readiness check. Returns 503 if the model or database is unavailable."""
    model_loaded = "model" in state
    db_ok = database_ok()
    healthy = model_loaded and db_ok

    body = {
        "status": "healthy" if healthy else "unhealthy",
        "model_loaded": model_loaded,
        "database": "up" if db_ok else "down",
        "model_version": state.get("metrics", {}).get("model_version"),
    }
    return JSONResponse(content=body, status_code=200 if healthy else 503)
@app.get("/")
def root() -> dict:
    """Signpost for anyone who lands on the base URL."""
    return {
        "service": "Demand Forecasting Service",
        "docs": "/docs",
        "health": "/health",
    }    

