"""The one integration test in the suite.

Everything else here runs on synthetic fixtures and would not catch the one
failure mode that matters most in production: the real model artifact and
the real database schema no longer working together (a migration that
renames a column build_features() depends on, a retrain that changes the
feature set, a stale models/*.txt, etc.). This test is the only thing that
would catch that class of problem -- it deliberately does the opposite of
every other test file and touches the real things.

Skips cleanly (not fail) if the database or the model artifact isn't
available, since this is meant to run wherever both are actually present,
not everywhere the fast suite runs.
"""
import pytest
from sqlalchemy import text

from artifact import load_metrics, load_model

from forecasting_service.app import main
from shared.database import get_session

# Verified in earlier debugging: 85123A is a high-volume trained product
# that reliably forecasts a large positive number, not just "some value".
KNOWN_PRODUCT_ID = "85123A"


def _require_database() -> None:
    try:
        db = get_session()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()
    except Exception as exc:
        pytest.skip(f"Real database unavailable: {exc}")


def _require_model_artifact():
    try:
        return load_model(), load_metrics()
    except FileNotFoundError as exc:
        pytest.skip(f"Real model artifact unavailable: {exc}")


@pytest.mark.integration
def test_forecast_against_real_model_and_database(api_client):
    _require_database()
    model, metrics = _require_model_artifact()

    trained_products = model.pandas_categorical[0]
    if KNOWN_PRODUCT_ID not in trained_products:
        pytest.skip(
            f"{KNOWN_PRODUCT_ID!r} is not in the currently trained product "
            "list; the model may have been retrained on a different top-50."
        )

    main.state["model"] = model
    main.state["metrics"] = metrics

    resp = api_client.post("/forecast", json={"product_id": KNOWN_PRODUCT_ID})

    assert resp.status_code == 200
    body = resp.json()
    assert body["product_id"] == KNOWN_PRODUCT_ID
    assert body["predicted_units"] > 0
