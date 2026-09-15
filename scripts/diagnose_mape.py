"""Throwaway diagnostic: is the 256% naive MAPE driven by low-volume rows?

Not wired into any pipeline. Answers one question and can be deleted once
the answer is written up.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting_service" / "src"))

from baseline import add_baselines
from split import load_demand, split_by_date

BINS = [0, 3, 10, 50, np.inf]
LABELS = ["1-3", "4-10", "11-50", "51+"]

BASELINES = [
    ("naive (last week)", "pred_naive"),
    ("seasonal naive (same week last year)", "pred_seasonal"),
]


def bucket_table(test: pd.DataFrame, column: str) -> pd.DataFrame:
    """MAPE and WAPE per units_sold bucket, restricted to non-zero actuals
    (same restriction metrics.mape() itself applies)."""
    rows = test[test["units_sold"] != 0].copy()
    rows["bucket"] = pd.cut(rows["units_sold"], bins=BINS, labels=LABELS)
    rows["abs_err"] = (rows["units_sold"] - rows[column]).abs()
    rows["ape"] = rows["abs_err"] / rows["units_sold"]

    out = rows.groupby("bucket", observed=True).apply(
        lambda g: pd.Series({
            "rows": len(g),
            "mape": g["ape"].mean(),
            "wape": g["abs_err"].sum() / g["units_sold"].sum(),
        }),
        include_groups=False,
    )
    return out


def worst_rows(test: pd.DataFrame, column: str, n: int = 5) -> pd.DataFrame:
    rows = test[test["units_sold"] != 0].copy()
    rows["ape"] = (rows["units_sold"] - rows[column]).abs() / rows["units_sold"]
    cols = ["product_id", "week_start", "units_sold", column, "ape"]
    return rows.nlargest(n, "ape")[cols]


def main() -> None:
    df = add_baselines(load_demand())
    _, test = split_by_date(df)

    for label, column in BASELINES:
        print("=" * 60)
        print(label)
        print("=" * 60)
        print(bucket_table(test, column).round(3))
        print()
        print(f"5 worst rows by APE ({label}):")
        print(worst_rows(test, column).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
