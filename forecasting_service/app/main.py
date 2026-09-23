"""Demand Forecasting Service.

Serves weekly demand forecasts for the top 50 products at a fixed 4-week horizon.
"""

import logging
import sys
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artifact import load_metrics, load_model  # noqa: E402

from forecasting_service.app import service
from forecasting_service.app.config import get_settings
from forecasting_service.app.schemas import (
    ErrorResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelInfoResponse,
    ProductsResponse,
)
from shared.database import get_session

logger = logging.getLogger(__name__)

# Populated once at startup, reused for every request.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server boots, not on every request."""
    logging.basicConfig(level=get_settings().log_level)
    logger.info("Loading model artifact...")
    state["model"] = load_model()
    state["metrics"] = load_metrics()
    logger.info("Loaded model version %s", state["metrics"]["model_version"])

    db = get_session()
    try:
        latest_week = service.resolve_latest_week(db)
        target_week = service.resolve_target_week(latest_week)
        logger.info(
            "Forecasting target week %s - %s (latest complete week in DB: %s)",
            target_week.isoformat(),
            (target_week + timedelta(days=6)).isoformat(),
            latest_week.isoformat(),
        )
    finally:
        db.close()

    yield
    state.clear()
    logger.info("Shutdown complete")


def get_db() -> Iterator[Session]:
    """A fresh DB session per request, closed when the request finishes."""
    db = get_session()
    try:
        yield db
    finally:
        db.close()


app = FastAPI(
    title="Demand Forecasting Service",
    description="Weekly demand forecasts for retail SKUs at a 4-week horizon.",
    version="1.0.0",
    lifespan=lifespan,
)


def database_ok(db: Session) -> bool:
    """Return True if the database answers a trivial query."""
    try:
        db.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        return False


@app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health(db: Session = Depends(get_db)) -> JSONResponse:
    """Readiness check. Returns 503 if the model or database is unavailable."""
    model_loaded = "model" in state
    db_ok = database_ok(db)
    healthy = model_loaded and db_ok

    body = HealthResponse(
        status="healthy" if healthy else "unhealthy",
        model_loaded=model_loaded,
        database="up" if db_ok else "down",
        model_version=state.get("metrics", {}).get("model_version"),
    )
    return JSONResponse(content=body.model_dump(mode="json"), status_code=200 if healthy else 503)


@app.post("/forecast", response_model=ForecastResponse, responses={404: {"model": ErrorResponse}})
def forecast(request: ForecastRequest, db: Session = Depends(get_db)) -> ForecastResponse:
    """Predict units for a trained product, horizon_days past the latest complete week in the DB."""
    try:
        return service.build_forecast(db, state["model"], state["metrics"], request.product_id)
    except service.UnknownProductError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Metadata about the currently loaded model, sourced from metrics.json."""
    return service.build_model_info(state["metrics"])


@app.get("/products", response_model=ProductsResponse)
def products() -> ProductsResponse:
    """The set of products the currently loaded model was trained to forecast."""
    trained = service.get_trained_products(state["model"])
    return ProductsResponse(products=trained, count=len(trained))


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")

