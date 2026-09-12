"""Train a LightGBM demand forecaster and compare it against the baselines.

Round count is chosen on an internal, time-ordered validation split (the
last VALID_WEEKS of the 52 train weeks), never on test -- early stopping
against test would let model selection optimise the same numbers we report,
which is the same class of leakage as picking products on future volume.
The final model is refit on the full train set at a scaled round count and
evaluated on test exactly once.
"""
import json
import sys
from datetime import date
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
MODEL_VERSION = "v1"

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
    print(f"Validation WAPE ({VALID_WEEKS} weeks)    : {valid_wape:.1%}")
    print(f"Test WAPE                     : {model_scores['wape']:.1%}")
    if valid_wape < model_scores["wape"]:
        gap = model_scores["wape"] - valid_wape
        print(f"  Validation beats test by {gap:.1%} pts -- possible overfitting "
              f"to the 2010 period given the year-over-year demand decline.")
    print()

    results = []
    for label, column in BASELINE_COLUMNS:
        results.append({"baseline": label, **evaluate(test["units_sold"], test[column])})
    results.append({"baseline": "LightGBM", **model_scores})

    ma4_wape = next(r["wape"] for r in results if r["baseline"] == "4-week moving average")
    improvement_pts = 100 * (ma4_wape - model_scores["wape"])
    improvement_rel = 100 * (ma4_wape - model_scores["wape"]) / ma4_wape

    print("RESULTS")
    print(f"Horizon: {HORIZON_WEEKS} weeks ahead")
    print(f"{'Model':<40} {'WAPE':>8} {'MAPE':>8} {'MAE':>8} {'Bias':>8}")
    print("-" * 76)
    for row in results:
        print(f"{row['baseline']:<40} "
              f"{row['wape']:>7.1%} {row['mape']:>7.1%} "
              f"{row['mae']:>8.1f} {row['bias']:>+8.1f}")
    print()
    print(f"Improvement over 4-week moving average: "
          f"{improvement_pts:+.1f} pts ({improvement_rel:+.1f}% relative)")

    print()
    print("TOP 10 FEATURES BY GAIN")
    importances = pd.Series(
        final_booster.feature_importance(importance_type="gain"),
        index=final_booster.feature_name(),
    ).sort_values(ascending=False)
    print(importances.head(10).round(1).to_string())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    training_date = date.today().isoformat()
    model_path = MODEL_DIR / f"demand_forecast_{MODEL_VERSION}.txt"
    final_booster.save_model(str(model_path))

    metrics = {
        "model_version": MODEL_VERSION,
        "training_date": training_date,
        "horizon_weeks": HORIZON_WEEKS,
        "baselines": {
            label: evaluate(test["units_sold"], test[column])
            for label, column in BASELINE_COLUMNS
        },
        "model": model_scores,
        "validation_wape": valid_wape,
        "best_iteration_raw": best_iteration,
        "best_iteration_scaled": scaled_iteration,
        "feature_list": feature_cols,
    }
    metrics_path = MODEL_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
