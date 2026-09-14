"""features.py builds the model's input columns. The leakage test below is
the single most important test in the suite: if week t's own units_sold
ever leaked into week t's own feature row, the service would be scoring on
information it doesn't have at forecast time.
"""
import numpy as np
import pandas as pd
import pytest

from baseline import HORIZON_WEEKS
from features import LAG_WEEKS, NON_FEATURE_COLS, ROLLING_WINDOWS, build_features


def make_series(product_id: str, start: str, n: int, values) -> pd.DataFrame:
    weeks = pd.date_range(start, periods=n, freq="W-MON")
    return pd.DataFrame({"product_id": product_id, "week_start": weeks, "units_sold": values})


def engineered_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def test_all_lags_are_at_least_horizon_weeks():
    """Guards the safety invariant features.py's own docstring documents:
    a lag shorter than HORIZON_WEEKS would read data not yet available at
    forecast time. If this ever fails, the reasoning in features.py's
    module docstring needs to be revisited, not just this test."""
    assert all(lag >= HORIZON_WEEKS for lag in LAG_WEEKS)


class TestNoLeakage:
    def test_mutating_week_t_does_not_change_week_ts_own_features(self):
        n = 30
        values = list(range(5, 5 + n))  # 5..34: strictly increasing, all nonzero
        df = make_series("P1", "2020-01-06", n, values)

        target_idx = 15  # far enough from both edges that every feature is fully defined
        target_week = df.loc[target_idx, "week_start"]

        baseline = build_features(df)
        baseline_row = baseline.loc[baseline["week_start"] == target_week, engineered_columns(baseline)]

        mutated = df.copy()
        mutated.loc[mutated["week_start"] == target_week, "units_sold"] = 999_999
        rebuilt = build_features(mutated)
        rebuilt_row = rebuilt.loc[rebuilt["week_start"] == target_week, engineered_columns(rebuilt)]

        pd.testing.assert_frame_equal(
            baseline_row.reset_index(drop=True),
            rebuilt_row.reset_index(drop=True),
        )

    def test_mutation_is_not_a_no_op_it_shows_up_later(self):
        """Positive control: proves the mutation above would have been
        caught if leakage existed, by confirming it DOES surface in a later
        row's lag_4, exactly HORIZON_WEEKS after the mutated week."""
        n = 30
        values = list(range(5, 5 + n))
        df = make_series("P1", "2020-01-06", n, values)

        target_idx = 15
        target_week = df.loc[target_idx, "week_start"]
        later_week = df.loc[target_idx + HORIZON_WEEKS, "week_start"]

        mutated = df.copy()
        mutated.loc[mutated["week_start"] == target_week, "units_sold"] = 999_999
        rebuilt = build_features(mutated)

        later_row = rebuilt.loc[rebuilt["week_start"] == later_week]
        assert later_row["lag_4"].iloc[0] == 999_999


class TestLagAlignment:
    @pytest.mark.parametrize("lag", LAG_WEEKS)
    def test_lag_equals_plain_shift(self, lag):
        n = 25
        values = list(range(n))
        df = make_series("P1", "2020-01-06", n, values)
        features = build_features(df)
        expected = df["units_sold"].shift(lag)
        pd.testing.assert_series_equal(
            features[f"lag_{lag}"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


class TestRollingWindowArithmetic:
    def test_roll_mean_4_is_mean_of_shifted_4_window(self):
        """roll_mean_4 at row t = mean of units_sold over weeks
        t-7..t-4 (shift(HORIZON_WEEKS) first, then a trailing 4-window)."""
        n = 20
        values = [10, 20, 30, 15, 25, 5, 40, 12, 8, 22, 18, 33, 7, 29, 14, 26, 9, 31, 16, 24]
        df = make_series("P1", "2020-01-06", n, values)
        features = build_features(df)

        t = 15
        expected_window = values[t - HORIZON_WEEKS - 3 : t - HORIZON_WEEKS + 1]
        assert len(expected_window) == 4
        expected_mean = sum(expected_window) / 4

        assert features.iloc[t]["roll_mean_4"] == pytest.approx(expected_mean)


class TestRollingWindowsRespectProductBoundaries:
    def test_second_products_first_row_does_not_see_first_products_history(self):
        """If shift/rolling were applied to the whole concatenated frame
        instead of per-product groups, B's first row -- which sits right
        after A's last row once sorted -- would pick up A's tail values
        instead of NaN."""
        n = 15
        product_a = make_series("A", "2020-01-06", n, [1000 + i for i in range(n)])
        product_b = make_series("B", "2020-01-06", n, [1 + i for i in range(n)])
        df = pd.concat([product_a, product_b], ignore_index=True)

        features = build_features(df)
        b_first_row = features[features["product_id"] == "B"].iloc[0]

        lookback_cols = [f"lag_{lag}" for lag in LAG_WEEKS] + [
            f"roll_{stat}_{w}" for stat in ("mean", "std", "max") for w in ROLLING_WINDOWS
        ]
        assert b_first_row[lookback_cols].isna().all()

    def test_rolling_stats_are_computed_independently_per_product(self):
        n = 20
        product_a = make_series("A", "2020-01-06", n, [100] * n)          # flat, high volume
        product_b = make_series("B", "2020-01-06", n, [1, 2, 3, 4] * 5)   # varying, low volume
        df = pd.concat([product_a, product_b], ignore_index=True)

        features = build_features(df)
        t = 15
        a_row = features[features["product_id"] == "A"].iloc[t]
        b_row = features[features["product_id"] == "B"].iloc[t]

        assert a_row["roll_mean_4"] != pytest.approx(b_row["roll_mean_4"])
        assert a_row["roll_std_4"] == pytest.approx(0.0)  # constant series has zero variance
        assert b_row["roll_std_4"] > 0


class TestCalendarFeatures:
    def test_peak_season_flag(self):
        df = make_series("P1", "2021-01-04", 52, list(range(52)))
        features = build_features(df)

        october_row = features.loc[features["week_start"] == pd.Timestamp("2021-10-04")].iloc[0]
        june_row = features.loc[features["week_start"] == pd.Timestamp("2021-06-07")].iloc[0]

        assert october_row["month"] == 10
        assert october_row["is_peak_season"] == 1
        assert june_row["month"] == 6
        assert june_row["is_peak_season"] == 0

    def test_week_of_year_matches_iso_calendar(self):
        df = make_series("P1", "2021-01-04", 10, list(range(10)))
        features = build_features(df)
        first_row = features.iloc[0]
        assert first_row["week_of_year"] == pd.Timestamp("2021-01-04").isocalendar().week


class TestSpikeRatio:
    def test_zero_mean_window_gives_nan_not_inf(self):
        n = 20
        values = [0] * 12 + [50] * 8  # long enough zero run to zero out roll_mean_8's window
        df = make_series("P1", "2020-01-06", n, values)
        features = build_features(df)

        t = 15  # shift(4) + rolling(8) window here sits entirely within the zero run
        row = features.iloc[t]
        assert row["roll_mean_8"] == 0.0
        assert np.isnan(row["spike_ratio"])
