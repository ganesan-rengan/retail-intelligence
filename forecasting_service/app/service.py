"""Prediction logic for the forecasting API.

Serving must never reimplement feature logic: raw weekly sales are queried
into a DataFrame shaped exactly like weekly_demand.csv (product_id,
week_start, units_sold), then handed to the same build_features() training
used. This module only does the plumbing around that call -- fetching a
single product's history, zero-filling gaps, and appending the one future
row build_features needs to compute the target week's features.
"""
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baseline import HORIZON_WEEKS  # noqa: E402
from features import NON_FEATURE_COLS, build_features  # noqa: E402

from forecasting_service.app.schemas import ForecastResponse, ModelInfoResponse

# Real weeks of history queried per product. Comfortably above the 16-week
# minimum (12-week rolling + 4-week shift) so that weeks_since_last_sale's
# unbounded ffill() sees the same span of history training did -- a shorter
# window would risk a spurious NaN for a product mid-way through a long dry
# spell that a full-history ffill would have resolved to a real number.
HISTORY_WEEKS = 52

LATEST_WEEK_BUCKETS_SQL = text("""
    SELECT DISTINCT DATE_TRUNC('week', order_date)::date AS week_start
    FROM orders
    WHERE status != 'cancelled'
    ORDER BY week_start DESC
    LIMIT 2
""")

WEEKLY_SALES_FOR_PRODUCT_SQL = text("""
    SELECT
        DATE_TRUNC('week', o.order_date)::date AS week_start,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON o.order_id = oi.order_id
    WHERE o.status != 'cancelled'
      AND oi.product_id = :product_id
      AND o.order_date >= :window_start
      AND o.order_date < :window_end
    GROUP BY DATE_TRUNC('week', o.order_date)
    ORDER BY week_start
""")


class UnknownProductError(ValueError):
    """Raised when a requested product_id is not one of the trained products."""

    def __init__(self, product_id: str):
        self.product_id = product_id
        super().__init__(
            f"Product '{product_id}' was not among the products this model was trained on."
        )


def get_trained_products(model: lgb.Booster) -> list[str]:
    """The exact product_id categories the model was trained on.

    Read from the pandas_categorical trailer LightGBM writes into the saved
    model file -- the authoritative list, versioned with the artifact itself,
    rather than a recomputation of demand.py's top-50 selection.
    """
    return list(model.pandas_categorical[0])


def resolve_latest_week(db: Session) -> date:
    """Return the most recent *complete* week present in the orders table.

    Mirrors demand.py's build_weekly_demand(), which unconditionally drops
    the first and last week-bucket of the full historical range before
    writing weekly_demand.csv -- on the assumption that a real-world data
    export is unlikely to start or end exactly on a week boundary, so the
    edge buckets are partial weeks rather than genuine data points. The
    model was trained and evaluated only up to that trimmed boundary
    (metrics.json's test_period.end matches it exactly: 2011-11-28, one
    week before the raw data's true last order). Serving must use the same
    boundary, or the calendar features (week_of_year, month, is_peak_season)
    would describe a week the model never saw represented the same way in
    training -- reintroducing exactly the skew this service exists to avoid.
    """
    rows = db.execute(LATEST_WEEK_BUCKETS_SQL).all()
    if len(rows) < 2:
        raise RuntimeError(
            "Fewer than two distinct order weeks found in the database; "
            "cannot determine a complete latest week."
        )
    return rows[1].week_start


def resolve_target_week(latest_week: date) -> date:
    """The week being forecast: HORIZON_WEEKS past the latest complete week."""
    return latest_week + timedelta(weeks=HORIZON_WEEKS)


def _build_product_frame(db: Session, product_id: str, latest_week: date) -> pd.DataFrame:
    """HISTORY_WEEKS real weeks (zero-filled) ending at latest_week, plus one
    synthetic row for the target week.

    The target row's own units_sold is never read: build_features's shifts
    and rolling windows for that row only look backward into the real
    history, and its calendar features (week_of_year, month, is_peak_season)
    are derived from its week_start, which is the actual week being
    forecast. Appending it here is what lets build_features compute both
    correctly without any bespoke serving-side feature logic.
    """
    window_start = pd.Timestamp(latest_week) - pd.Timedelta(weeks=HISTORY_WEEKS - 1)
    window_end = pd.Timestamp(latest_week) + pd.Timedelta(days=7)

    rows = db.execute(
        WEEKLY_SALES_FOR_PRODUCT_SQL,
        {"product_id": product_id, "window_start": window_start, "window_end": window_end},
    ).all()
    sales = pd.DataFrame(rows, columns=["week_start", "units_sold"])
    sales["week_start"] = pd.to_datetime(sales["week_start"])

    full_weeks = pd.date_range(end=pd.Timestamp(latest_week), periods=HISTORY_WEEKS, freq="W-MON")
    history = pd.DataFrame({"week_start": full_weeks}).merge(sales, on="week_start", how="left")
    history["units_sold"] = history["units_sold"].fillna(0).astype(int)
    history["product_id"] = product_id

    target_week = resolve_target_week(latest_week)
    target_row = pd.DataFrame({
        "product_id": [product_id],
        "week_start": [pd.Timestamp(target_week)],
        "units_sold": [0],
    })
    return pd.concat([history, target_row], ignore_index=True)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    engineered = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return engineered + ["product_id"]


def _assert_feature_columns_match(built: list[str], expected: list[str]) -> None:
    """Guard against features.py and the deployed model silently drifting
    apart -- e.g. a retrain that changes the feature set without a matching
    service.py update. Fails loudly and names the mismatch instead of
    silently mispredicting on misaligned columns.
    """
    if built == expected:
        return
    missing = [c for c in expected if c not in built]
    extra = [c for c in built if c not in expected]
    if not missing and not extra:
        reason = f"same features, different order -- built={built}, model expects={expected}"
    else:
        reason = f"missing={missing}, unexpected={extra}"
    raise RuntimeError(
        f"Serving-time feature columns do not match the model's trained features ({reason}). "
        "features.py and the deployed model artifact have fallen out of step."
    )


def predict_units(model: lgb.Booster, product_frame: pd.DataFrame) -> float:
    """Run build_features on product_frame and predict the final (target) row."""
    features_df = build_features(product_frame)
    target_row = features_df.iloc[[-1]].copy()

    _assert_feature_columns_match(_feature_columns(features_df), model.feature_name())

    target_row["product_id"] = target_row["product_id"].astype("category")
    raw_prediction = model.predict(target_row[model.feature_name()])[0]
    return float(np.clip(raw_prediction, 0, None))


def build_forecast(db: Session, model: lgb.Booster, metrics: dict, product_id: str) -> ForecastResponse:
    """End-to-end: validate product_id, fetch history, predict, shape the response."""
    if product_id not in get_trained_products(model):
        raise UnknownProductError(product_id)

    latest_week = resolve_latest_week(db)
    target_week = resolve_target_week(latest_week)
    product_frame = _build_product_frame(db, product_id, latest_week)
    predicted_units = predict_units(model, product_frame)

    return ForecastResponse(
        product_id=product_id,
        predicted_units=predicted_units,
        week_start=target_week,
        week_end=target_week + timedelta(days=6),
        model_version=metrics["model_version"],
        horizon_days=HORIZON_WEEKS * 7,
        generated_at=datetime.now(timezone.utc),
    )


def build_model_info(metrics: dict) -> ModelInfoResponse:
    baseline = metrics["baseline_comparison"]
    return ModelInfoResponse(
        model_version=metrics["model_version"],
        trained_at=metrics["trained_at"],
        algorithm=metrics["algorithm"],
        horizon_weeks=metrics["horizon_weeks"],
        n_products=metrics["n_products"],
        n_features=len(metrics["features"]),
        best_baseline=baseline["best_baseline"],
        best_baseline_wape=baseline["best_baseline_wape"],
        test_wape=metrics["metrics"]["test_wape"],
        improvement_points=baseline["improvement_points"],
        improvement_relative=baseline["improvement_relative"],
    )
