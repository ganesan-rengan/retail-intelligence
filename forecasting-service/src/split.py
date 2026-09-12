"""Time-based train/test split for the weekly demand table.

Defined once and imported everywhere so the boundary cannot drift between
training, backtesting and evaluation.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "weekly_demand.csv"
)

# 104 weeks total, split in half. Test covers 2010-12-06 .. 2011-11-28.
SPLIT_DATE = pd.Timestamp("2010-12-06")


def load_demand(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the weekly demand table with week_start parsed as a datetime."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: uv run python forecasting-service/src/demand.py"
        )
    df = pd.read_csv(path, parse_dates=["week_start"])
    return df.sort_values(["product_id", "week_start"]).reset_index(drop=True)


def split_by_date(
    df: pd.DataFrame, split_date: pd.Timestamp = SPLIT_DATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically. Everything before split_date trains, the rest tests."""
    train = df[df["week_start"] < split_date].copy()
    test = df[df["week_start"] >= split_date].copy()

    # Guardrails: cheap to check, expensive to discover later.
    assert not train.empty, "Train split is empty"
    assert not test.empty, "Test split is empty"
    assert train["week_start"].max() < test["week_start"].min(), (
        "Train and test overlap in time"
    )

    missing = set(test["product_id"]) - set(train["product_id"])
    assert not missing, f"Products in test but not train: {sorted(missing)[:5]}"

    return train, test


def main() -> None:
    df = load_demand()
    train, test = split_by_date(df)

    for name, part in [("TRAIN", train), ("TEST", test)]:
        print(f"{name}")
        print(f"  Rows     : {len(part):,}")
        print(f"  Products : {part['product_id'].nunique()}")
        print(f"  Weeks    : {part['week_start'].nunique()}")
        print(f"  Range    : {part['week_start'].min().date()} -> "
              f"{part['week_start'].max().date()}")
        print(f"  Units    : {part['units_sold'].sum():,}")
        print()

    drop = 100 * (test["units_sold"].sum() / train["units_sold"].sum() - 1)
    print(f"Test period is {drop:+.1f}% versus train period")


if __name__ == "__main__":
    main()