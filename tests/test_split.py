"""split.py's chronological boundary is what keeps train/test honest across
the whole pipeline. These tests exercise it directly on synthetic frames,
never touching data/processed/weekly_demand.csv.
"""
import pandas as pd
import pytest

from split import load_demand, split_by_date


def make_demand(product_id: str, weeks: list[str], units=None) -> pd.DataFrame:
    week_starts = pd.to_datetime(weeks)
    units = units if units is not None else [10] * len(weeks)
    return pd.DataFrame({"product_id": product_id, "week_start": week_starts, "units_sold": units})


class TestSplitByDate:
    def test_train_max_strictly_before_test_min(self):
        df = make_demand("P1", ["2021-01-04", "2021-01-11", "2021-01-18", "2021-01-25", "2021-02-01"])
        split_date = pd.Timestamp("2021-01-18")

        train, test = split_by_date(df, split_date)

        assert train["week_start"].max() < test["week_start"].min()
        assert train["week_start"].max() < split_date
        assert test["week_start"].min() >= split_date

    def test_rows_partition_exactly_by_the_boundary(self):
        df = make_demand("P1", ["2021-01-04", "2021-01-11", "2021-01-18", "2021-01-25"])
        split_date = pd.Timestamp("2021-01-18")

        train, test = split_by_date(df, split_date)

        assert len(train) + len(test) == len(df)
        assert set(train["week_start"]).isdisjoint(set(test["week_start"]))

    def test_raises_when_train_is_empty(self):
        df = make_demand("P1", ["2021-06-01", "2021-06-08"])
        split_date = pd.Timestamp("2021-01-01")  # every row lands in test

        with pytest.raises(AssertionError, match="Train split is empty"):
            split_by_date(df, split_date)

    def test_raises_when_test_is_empty(self):
        df = make_demand("P1", ["2021-01-01", "2021-01-08"])
        split_date = pd.Timestamp("2021-06-01")  # every row lands in train

        with pytest.raises(AssertionError, match="Test split is empty"):
            split_by_date(df, split_date)

    def test_raises_when_a_product_appears_only_in_test(self):
        """A product with no history before split_date can't be modeled at
        all -- this must fail loudly rather than silently train on 49/50
        products and test on 50."""
        split_date = pd.Timestamp("2021-01-18")
        established = make_demand("P1", ["2021-01-04", "2021-01-11", "2021-01-25"])
        late_arrival = make_demand("P2", ["2021-01-25", "2021-02-01"])  # only after split_date
        df = pd.concat([established, late_arrival], ignore_index=True)

        with pytest.raises(AssertionError, match="Products in test but not train"):
            split_by_date(df, split_date)


class TestLoadDemand:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            load_demand(missing)

    def test_parses_week_start_as_datetime_and_sorts(self, tmp_path):
        csv_path = tmp_path / "synthetic_demand.csv"
        # deliberately out of order and interleaved across products
        pd.DataFrame({
            "product_id": ["B", "A", "B", "A"],
            "week_start": ["2021-01-11", "2021-01-04", "2021-01-04", "2021-01-11"],
            "units_sold": [5, 1, 2, 3],
        }).to_csv(csv_path, index=False)

        df = load_demand(csv_path)

        assert pd.api.types.is_datetime64_any_dtype(df["week_start"])
        # Sorted by [product_id, week_start] -- not a global date sort, so
        # week_start repeats out of date order once product_id changes.
        assert list(df["product_id"]) == ["A", "A", "B", "B"]
        assert list(df["week_start"]) == list(pd.to_datetime(
            ["2021-01-04", "2021-01-11", "2021-01-04", "2021-01-11"]
        ))
