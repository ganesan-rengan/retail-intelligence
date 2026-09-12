"""Forecast accuracy metrics.

WAPE is the headline metric: the demand series is sparse (9% zero-sales weeks)
and MAPE is undefined when the actual is zero.
"""

import numpy as np
import pandas as pd


def wape(actual: pd.Series, predicted: pd.Series) -> float:
    """Weighted absolute percentage error. Total error / total actual volume."""
    denominator = actual.sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(actual - predicted).sum() / denominator)


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean absolute percentage error, computed on non-zero actuals only."""
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float((np.abs(actual[mask] - predicted[mask]) / actual[mask]).mean())


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean absolute error in units."""
    return float(np.abs(actual - predicted).mean())


def bias(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean signed error. Positive means the forecast runs high."""
    return float((predicted - actual).mean())


def evaluate(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """All metrics for one forecast."""
    return {
        "wape": wape(actual, predicted),
        "mape": mape(actual, predicted),
        "mae": mae(actual, predicted),
        "bias": bias(actual, predicted),
    }