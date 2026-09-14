"""Shared fixtures for the fast (non-integration) test suite.

Nothing here touches the real database or the real model artifact. Tests
that need either must be marked @pytest.mark.integration and are excluded
by the default `-m "not integration"` run (see pyproject.toml).
"""
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "forecasting_service" / "src"))

from features import build_features  # noqa: E402

from forecasting_service.app import main, service  # noqa: E402

TRAINED_PRODUCTS = ["P1", "P2", "P3"]


def _feature_frame_columns() -> list[str]:
    """The real feature-column order build_features() produces, computed
    from a throwaway synthetic frame -- never hardcoded, so it can't drift
    from features.py's actual behaviour."""
    tiny = pd.DataFrame({
        "product_id": ["P1"] * 20,
        "week_start": pd.date_range("2020-01-06", periods=20, freq="W-MON"),
        "units_sold": list(range(20)),
    })
    return service._feature_columns(build_features(tiny))


class FakeModel:
    """Stands in for lgb.Booster: exposes only what service.py touches."""

    def __init__(self, products=TRAINED_PRODUCTS, prediction=100.0, feature_names=None):
        self.pandas_categorical = [list(products)]
        self._prediction = prediction
        self._feature_names = feature_names or _feature_frame_columns()

    def feature_name(self):
        return self._feature_names

    def predict(self, df):
        return np.array([self._prediction])


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def sample_metrics() -> dict:
    """Shaped exactly like models/metrics.json, but synthetic."""
    return {
        "model_version": "vTEST",
        "trained_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        "algorithm": "lightgbm",
        "horizon_weeks": 4,
        "n_products": len(TRAINED_PRODUCTS),
        "features": _feature_frame_columns(),
        "metrics": {
            "validation_wape": 0.60,
            "test_wape": 0.65,
            "test_mape": 1.5,
            "test_mae": 100.0,
            "test_bias": 5.0,
        },
        "baseline_comparison": {
            "best_baseline": "4-week moving average",
            "best_baseline_wape": 0.70,
            "improvement_points": 0.05,
            "improvement_relative": 0.0714,
        },
    }


def make_fake_db(latest_week_rows, sales_rows):
    """A MagicMock DB session whose .execute() returns canned results in the
    exact order build_forecast() issues them: first the latest-week query,
    then the per-product weekly-sales query."""
    db = MagicMock()
    db.execute.side_effect = [
        MagicMock(all=MagicMock(return_value=latest_week_rows)),
        MagicMock(all=MagicMock(return_value=sales_rows)),
    ]
    return db


class Row:
    """Minimal stand-in for a SQLAlchemy Row: attribute access plus
    tuple-style iteration, matching what service.py reads."""

    def __init__(self, *values, week_start=None):
        self._values = values
        self.week_start = week_start if week_start is not None else values[0]

    def __iter__(self):
        return iter(self._values)


@pytest.fixture
def fake_db_factory():
    return make_fake_db


@pytest.fixture
def api_client():
    """A TestClient that never triggers lifespan (no `with` block, verified
    empirically not to run startup/shutdown), with state cleared before and
    after so tests can't see each other's fakes."""
    from fastapi.testclient import TestClient

    main.state.clear()
    main.app.dependency_overrides.clear()
    client = TestClient(main.app)
    yield client
    main.app.dependency_overrides.clear()
    main.state.clear()
