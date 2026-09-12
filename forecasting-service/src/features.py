"""Feature engineering for the 4-week-ahead demand forecast.

Leakage rule: predicting week t, the most recent usable observation is week
t - HORIZON_WEEKS. Lags in LAG_WEEKS are all >= HORIZON_WEEKS, so they are
safe to use as plain shifts. Rolling windows are not safe as plain windows
(an unshifted rolling(w) would reach into week t-1), so each one shifts by
HORIZON_WEEKS first, inside groupby().transform() so windows never cross
product boundaries.

lag_52 is deliberately excluded: it needs 52 weeks of prior history, but the
train split is exactly the first 52 weeks, so lag_52 is NaN for 100% of
training rows. Keeping it would not be rescued by LightGBM's native NaN
handling -- with zero non-missing training rows there is nothing for a split
to learn, missing or not.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline import HORIZON_WEEKS
from split import load_demand, split_by_date

LAG_WEEKS = [4, 5, 6, 8, 12]
ROLLING_WINDOWS = [4, 8, 12]
PEAK_MONTHS = {10, 11}

NON_FEATURE_COLS = {"product_id", "week_start", "units_sold"}


def _weeks_since_last_sale(s: pd.Series) -> pd.Series:
    """Weeks since the most recent nonzero value, as of week t - HORIZON_WEEKS."""
    shifted = s.shift(HORIZON_WEEKS)
    position = np.arange(len(s))
    had_sale = np.where(shifted.isna(), np.nan, (shifted != 0).astype(float))
    last_sale_position = pd.Series(
        np.where(had_sale == 1, position, np.nan), index=s.index
    ).ffill()
    return pd.Series(position, index=s.index) - last_sale_position


def _nonzero_indicator(s: pd.Series) -> pd.Series:
    """1/0/NaN indicator, NaN-aware so rolling().mean() inherits the same
    missing-data span as the other shift(HORIZON_WEEKS) rolling features."""
    shifted = s.shift(HORIZON_WEEKS)
    return pd.Series(
        np.where(shifted.isna(), np.nan, (shifted != 0).astype(float)), index=s.index
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach lag, rolling, intermittency, spike, and calendar features."""
    df = df.sort_values(["product_id", "week_start"]).copy()
    grouped = df.groupby("product_id")["units_sold"]

    for lag in LAG_WEEKS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"roll_mean_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(HORIZON_WEEKS).rolling(w).mean()
        )
        df[f"roll_std_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(HORIZON_WEEKS).rolling(w).std()
        )
        df[f"roll_max_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(HORIZON_WEEKS).rolling(w).max()
        )

    df["weeks_since_last_sale"] = grouped.transform(_weeks_since_last_sale)
    df["nonzero_rate_8"] = grouped.transform(
        lambda s: _nonzero_indicator(s).rolling(8).mean()
    )

    df["spike_ratio"] = df["roll_max_8"] / df["roll_mean_8"].replace(0, np.nan)

    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    df["is_peak_season"] = df["month"].isin(PEAK_MONTHS).astype(int)

    return df


def main() -> None:
    df = build_features(load_demand())
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    print(f"Feature count: {len(feature_cols)}")

    # The pipeline never drops NaN: LightGBM handles missing values natively,
    # and on a dataset this small, discarding ~16% of rows to force
    # complete cases costs more signal than it saves. Split first, same
    # order as baseline.py, so a product with no learnable history can't
    # silently disappear from train.
    train, test = split_by_date(df)
    train_complete = train.dropna(subset=feature_cols)
    test_complete = test.dropna(subset=feature_cols)

    print(f"Split (what the model trains on): "
          f"train {len(train):,} / test {len(test):,}")
    print(f"Diagnostic - complete-case rows:  "
          f"train {len(train_complete):,} ({100 * len(train_complete) / len(train):.1f}%) / "
          f"test {len(test_complete):,} ({100 * len(test_complete) / len(test):.1f}%)")

    no_history = train.groupby("product_id")["units_sold"].sum()
    no_history = no_history[no_history == 0]
    if not no_history.empty:
        print(f"\nWARNING: {len(no_history)} product(s) have zero sales in the "
              f"entire train period, so every lag/rolling feature is NaN for "
              f"100% of their train rows: {list(no_history.index)}")

    print()
    print(df.head())


if __name__ == "__main__":
    main()
