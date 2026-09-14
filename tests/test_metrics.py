"""metrics.py is pure arithmetic on pandas Series -- every expected value
here is computed by hand in the test, not just re-derived from the code.
"""
import math

import pandas as pd
import pytest

from metrics import bias, evaluate, mae, mape, wape


def s(*values) -> pd.Series:
    return pd.Series(values, dtype=float)


class TestWape:
    def test_basic(self):
        actual, predicted = s(10, 20, 30), s(12, 18, 33)
        # |10-12| + |20-18| + |30-33| = 2 + 2 + 3 = 7; sum(actual) = 60
        assert wape(actual, predicted) == pytest.approx(7 / 60)

    def test_perfect_forecast_is_zero(self):
        actual = s(5, 10, 15)
        assert wape(actual, actual.copy()) == 0.0

    def test_all_zero_predictions(self):
        actual, predicted = s(10, 20), s(0, 0)
        # total error 30, total actual 30
        assert wape(actual, predicted) == pytest.approx(1.0)

    def test_zero_actuals_is_nan(self):
        actual, predicted = s(0, 0, 0), s(1, 2, 3)
        assert math.isnan(wape(actual, predicted))


class TestMape:
    def test_basic_excludes_zero_actuals(self):
        actual, predicted = s(10, 20, 0, 30), s(12, 18, 5, 33)
        # rows with actual=0 are masked out entirely, not counted as error
        # |10-12|/10=0.2, |20-18|/20=0.1, |30-33|/30=0.1 -> mean = 0.4/3
        assert mape(actual, predicted) == pytest.approx(0.4 / 3)

    def test_perfect_forecast_is_zero(self):
        actual = s(5, 10)
        assert mape(actual, actual.copy()) == 0.0

    def test_all_zero_actuals_is_nan(self):
        actual, predicted = s(0, 0, 0), s(1, 2, 3)
        assert math.isnan(mape(actual, predicted))

    def test_zero_actual_row_is_silently_dropped_not_penalized(self):
        """Documents the exact unreliability CLAUDE.md flags: a huge miss on
        a zero-actual row contributes nothing to MAPE because that row is
        masked out, not because the forecast was good."""
        actual, predicted = s(0, 5), s(100, 5)
        assert mape(actual, predicted) == pytest.approx(0.0)


class TestMae:
    def test_basic(self):
        actual, predicted = s(10, 20, 30), s(12, 18, 33)
        assert mae(actual, predicted) == pytest.approx(7 / 3)

    def test_all_zero_predictions(self):
        actual, predicted = s(10, 20), s(0, 0)
        assert mae(actual, predicted) == pytest.approx(15.0)


class TestBias:
    def test_forecast_runs_high(self):
        actual, predicted = s(10, 20), s(12, 25)
        assert bias(actual, predicted) == pytest.approx(3.5)

    def test_forecast_runs_low(self):
        actual, predicted = s(10, 20), s(8, 15)
        assert bias(actual, predicted) == pytest.approx(-3.5)

    def test_perfect_forecast_is_zero(self):
        actual = s(5, 10, 15)
        assert bias(actual, actual.copy()) == 0.0


class TestEvaluate:
    def test_returns_all_four_metrics_consistently(self):
        actual, predicted = s(10, 20, 30), s(12, 18, 33)
        result = evaluate(actual, predicted)
        assert result.keys() == {"wape", "mape", "mae", "bias"}
        assert result["wape"] == pytest.approx(wape(actual, predicted))
        assert result["mape"] == pytest.approx(mape(actual, predicted))
        assert result["mae"] == pytest.approx(mae(actual, predicted))
        assert result["bias"] == pytest.approx(bias(actual, predicted))
