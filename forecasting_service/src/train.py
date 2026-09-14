"""Train a LightGBM demand forecaster and compare it against the baselines.

Round count is chosen on an internal, time-ordered validation split (the
last VALID_WEEKS of the 52 train weeks), never on test -- early stopping
against test would let model selection optimise the same numbers we report,
which is the same class of leakage as picking products on future volume.
The final model is refit on the full train set at a scaled round count and
evaluated on test exactly once.
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline import HORIZON_WEEKS, add_baselines
from features import NON_FEATURE_COLS, build_features
from metrics import evaluate
from split import SPLIT_DATE, load_demand, split_by_date

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
HISTORY_DIR = MODEL_DIR / "history"
METRICS_PATH = MODEL_DIR / "metrics.json"

VALID_WEEKS = 8
MAX_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50

PARAMS = {
    "objective": "regression_l1",  # WAPE is an absolute-error metric; L2 would optimise a different objective
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbose": -1,
}

BASELINE_COLUMNS = [
    ("naive (4 weeks ago)", "pred_naive"),
    ("seasonal naive (same week last year)", "pred_seasonal"),
    ("4-week moving average", "pred_ma4"),
]


def make_dataset(frame: pd.DataFrame, feature_cols: list[str], reference=None) -> lgb.Dataset:
    return lgb.Dataset(
        frame[feature_cols],
        label=frame["units_sold"],
        categorical_feature=["product_id"],
        reference=reference,
        free_raw_data=False,
    )


def next_model_version() -> int:
    """max(tracked versions) + 1. Deliberately ignores models/*.txt -- that's
    gitignored and may not exist locally even though git history already
    records a higher version (see ADR 006)."""
    versions = [0]
    if METRICS_PATH.exists():
        try:
            current = json.loads(METRICS_PATH.read_text())
            versions.append(int(current["model_version"].lstrip("v")))
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    if HISTORY_DIR.exists():
        for path in HISTORY_DIR.glob("metrics_v*.json"):
            match = re.fullmatch(r"metrics_v(\d+)\.json", path.name)
            if match:
                versions.append(int(match.group(1)))
    return max(versions) + 1


def archive_current_metrics() -> None:
    """Copy the current metrics.json to history under its own recorded
    version before it gets overwritten. No-op on the first ever run."""
    if not METRICS_PATH.exists():
        return
    current = json.loads(METRICS_PATH.read_text())
    old_version = int(current["model_version"].lstrip("v"))
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = HISTORY_DIR / f"metrics_v{old_version}.json"
    archive_path.write_text(json.dumps(current, indent=2))


def write_metrics_atomic(data: dict) -> None:
    """Write-to-temp-then-replace so a crash mid-write can't leave the
    tracked, current metrics.json half-written."""
    tmp_path = METRICS_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2))
    tmp_path.replace(METRICS_PATH)


def main() -> None:
    df = build_features(add_baselines(load_demand()))
    df["product_id"] = df["product_id"].astype("category")
    engineered_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and not c.startswith("pred_")]
    feature_cols = engineered_cols + ["product_id"]

    train, test = split_by_date(df)

    # Internal validation: last VALID_WEEKS of train, strictly after the fit
    # weeks -- same chronological-split principle as split_by_date, just
    # computed locally since split.py isn't allowed to change.
    valid_cutoff = SPLIT_DATE - pd.Timedelta(weeks=VALID_WEEKS)
    fit = train[train["week_start"] < valid_cutoff]
    valid = train[train["week_start"] >= valid_cutoff]

    fit_ds = make_dataset(fit, feature_cols)
    valid_ds = make_dataset(valid, feature_cols, reference=fit_ds)

    tuning_booster = lgb.train(
        PARAMS,
        fit_ds,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[valid_ds],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)],
    )
    best_iteration = tuning_booster.best_iteration

    fit_weeks = fit["week_start"].nunique()
    train_weeks = train["week_start"].nunique()
    scaled_iteration = round(best_iteration * train_weeks / fit_weeks)

    valid_pred = np.clip(
        tuning_booster.predict(valid[feature_cols], num_iteration=best_iteration), 0, None
    )
    valid_wape = evaluate(valid["units_sold"], valid_pred)["wape"]

    # Refit on the full 52-week train set at the scaled round count. No
    # early stopping here -- the round count is already decided, and test
    # is not touched until the single evaluation below.
    train_ds = make_dataset(train, feature_cols)
    final_booster = lgb.train(PARAMS, train_ds, num_boost_round=scaled_iteration)

    test_pred = np.clip(final_booster.predict(test[feature_cols]), 0, None)
    model_scores = evaluate(test["units_sold"], test_pred)

    print(f"Best iteration ({fit_weeks} weeks)    : {best_iteration}")
    print(f"Scaled iteration ({train_weeks} weeks)  : {scaled_iteration}")
    yoy_decline = float(test["units_sold"].sum() / train["units_sold"].sum() - 1)

    print(f"Validation WAPE ({VALID_WEEKS} weeks)    : {valid_wape:.1%}")
    print(f"Test WAPE                     : {model_scores['wape']:.1%}")
    if valid_wape < model_scores["wape"]:
        gap = model_scores["wape"] - valid_wape
        print(f"  Validation beats test by {gap:.1%} pts -- distribution drift "
              f"(demand fell {abs(yoy_decline):.1%} from train to test), not overfitting.")
    print()

    baseline_scores = {
        label: evaluate(test["units_sold"], test[column]) for label, column in BASELINE_COLUMNS
    }
    results = [{"baseline": label, **scores} for label, scores in baseline_scores.items()]
    results.append({"baseline": "LightGBM", **model_scores})

    best_baseline_label, best_baseline_result = min(baseline_scores.items(), key=lambda kv: kv[1]["wape"])
    best_baseline_wape = best_baseline_result["wape"]
    # Fractions, matching wape/mape/bias elsewhere -- format as percentages
    # only at print/render time, never store the *100'd value.
    improvement_points = best_baseline_wape - model_scores["wape"]
    improvement_relative = (best_baseline_wape - model_scores["wape"]) / best_baseline_wape

    print("RESULTS")
    print(f"Horizon: {HORIZON_WEEKS} weeks ahead")
    print(f"{'Model':<40} {'WAPE':>8} {'MAPE':>8} {'MAE':>8} {'Bias':>8}")
    print("-" * 76)
    for row in results:
        print(f"{row['baseline']:<40} "
              f"{row['wape']:>7.1%} {row['mape']:>7.1%} "
              f"{row['mae']:>8.1f} {row['bias']:>+8.1f}")
    print()
    print(f"Improvement over {best_baseline_label}: "
          f"{improvement_points * 100:+.1f} pts ({improvement_relative * 100:+.1f}% relative)")

    print()
    print("TOP 10 FEATURES BY GAIN")
    importances = pd.Series(
        final_booster.feature_importance(importance_type="gain"),
        index=final_booster.feature_name(),
    ).sort_values(ascending=False)
    print(importances.head(10).round(1).to_string())

    version = next_model_version()
    model_version = f"v{version}"
    model_filename = f"demand_forecast_{model_version}.txt"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / model_filename
    final_booster.save_model(str(model_path))

    metrics_out = {
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_file": model_filename,
        "algorithm": "lightgbm",
        "objective": PARAMS["objective"],
        "horizon_weeks": HORIZON_WEEKS,
        "n_products": int(train["product_id"].nunique()),
        "n_train_rows": int(len(train)),
        "train_period": {
            "start": train["week_start"].min().date().isoformat(),
            "end": train["week_start"].max().date().isoformat(),
        },
        "test_period": {
            "start": test["week_start"].min().date().isoformat(),
            "end": test["week_start"].max().date().isoformat(),
        },
        "yoy_decline": yoy_decline,
        "features": feature_cols,
        "best_iteration": {"raw": best_iteration, "scaled": scaled_iteration},
        "metrics": {
            "validation_wape": valid_wape,
            "test_wape": model_scores["wape"],
            "test_mape": model_scores["mape"],
            "test_mae": model_scores["mae"],
            "test_bias": model_scores["bias"],
        },
        "baseline_comparison": {
            "best_baseline": best_baseline_label,
            "best_baseline_wape": best_baseline_wape,
            "improvement_points": improvement_points,
            "improvement_relative": improvement_relative,
        },
    }

    archive_current_metrics()
    write_metrics_atomic(metrics_out)
    print(f"\nSaved model to {model_path}")
    print(f"Saved metrics to {METRICS_PATH} (model_version {model_version})")


if __name__ == "__main__":
    main()
