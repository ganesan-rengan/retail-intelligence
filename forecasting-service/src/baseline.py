"""Naive and seasonal-naive baselines.

Any real model must beat these. Recorded before training so the comparison
cannot be adjusted after the fact.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import evaluate
from split import load_demand, split_by_date

SEASONAL_LAG_WEEKS = 52
METRICS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_metrics.csv"
)


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Attach naive and seasonal-naive predictions to every row.

    Both shifts look strictly backwards within each product, so computing them
    on the full frame before splitting introduces no leakage: a test-set row
    can only ever read values that precede it in time.
    """
    df = df.sort_values(["product_id", "week_start"]).copy()
    grouped = df.groupby("product_id")["units_sold"]

    df["pred_naive"] = grouped.shift(1)
    df["pred_seasonal"] = grouped.shift(SEASONAL_LAG_WEEKS)
    df["pred_ma4"] = grouped.transform(lambda s: s.shift(1).rolling(4).mean())

    return df


def main() -> None:
    df = add_baselines(load_demand())
    _, test = split_by_date(df)

    missing = test[["pred_naive", "pred_seasonal", "pred_ma4"]].isna().sum()
    if missing.any():
        raise ValueError(f"Test set has missing predictions:\n{missing}")

    results = []
    for label, column in [
        ("naive (last week)", "pred_naive"),
        ("seasonal naive (same week last year)", "pred_seasonal"),
        ("4-week moving average", "pred_ma4"),
    ]:
        scores = evaluate(test["units_sold"], test[column])
        results.append({"baseline": label, **scores})

    out = pd.DataFrame(results)

    print("BASELINE RESULTS")
    print(f"Test period: {test['week_start'].min().date()} -> "
          f"{test['week_start'].max().date()}")
    print(f"Rows: {len(test):,}  Products: {test['product_id'].nunique()}")
    print()
    print(f"{'Baseline':<40} {'WAPE':>8} {'MAPE':>8} {'MAE':>8} {'Bias':>8}")
    print("-" * 76)
    for row in results:
        print(f"{row['baseline']:<40} "
              f"{row['wape']:>7.1%} {row['mape']:>7.1%} "
              f"{row['mae']:>8.1f} {row['bias']:>+8.1f}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(METRICS_PATH, index=False)
    print(f"\nSaved to {METRICS_PATH}")


if __name__ == "__main__":
    main()