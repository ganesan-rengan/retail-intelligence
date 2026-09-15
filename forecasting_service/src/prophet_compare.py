"""Classical time-series baseline: Prophet, compared both fairly and naively.

Two Prophet variants are reported:

  "Prophet, rolling-origin, 4-week horizon" -- refit every 4 weeks using all
  data known at that point, forecasting only the next 4-week block. Same
  fixed horizon and continuously-refreshed history as LightGBM's test-set
  predictions. 50 products x 13 origins = 650 fits.

  "Prophet, single fit, 52-week horizon" -- fit once on train, forecast the
  entire test period in one shot. Included so the gap between the two rows
  is visible: it shows how much of Prophet's disadvantage is the method
  versus the much longer horizon it was originally asked to solve.

The other four rows (naive, seasonal naive, 4-week moving average, LightGBM)
are read from models/metrics.json rather than recomputed, so the numbers
here can never drift from what train.py actually produced.
"""
import contextlib
import io
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline import HORIZON_WEEKS
from metrics import evaluate
from split import load_demand, split_by_date

logging.getLogger("prophet").disabled = True
logging.getLogger("cmdstanpy").disabled = True

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
N_ORIGINS = 13  # 52 test weeks / HORIZON_WEEKS


def _fit_predict(history: pd.DataFrame, target_dates: pd.DatetimeIndex) -> np.ndarray | None:
    """Fit one Prophet model on `history` and predict `target_dates`.

    Returns None on failure instead of raising, so a single bad product or
    origin cannot stop the whole run.
    """
    prophet_df = history.rename(columns={"week_start": "ds", "units_sold": "y"})
    model = Prophet(weekly_seasonality=False, yearly_seasonality=True, daily_seasonality=False)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model.fit(prophet_df)
            forecast = model.predict(pd.DataFrame({"ds": target_dates}))
        return np.clip(forecast["yhat"].to_numpy(), 0, None)
    except Exception:
        return None


def run_single_fit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """One Prophet fit per product on train, forecasting the full test period."""
    actuals, preds, failures = [], [], []
    products = sorted(test["product_id"].unique())

    for i, pid in enumerate(products, 1):
        product_train = train.loc[train["product_id"] == pid, ["week_start", "units_sold"]]
        product_test = test.loc[test["product_id"] == pid, ["week_start", "units_sold"]]

        pred = _fit_predict(product_train, pd.DatetimeIndex(product_test["week_start"]))
        if pred is None:
            failures.append(pid)
        else:
            actuals.append(product_test["units_sold"].to_numpy())
            preds.append(pred)

        print(f"  [{i}/{len(products)}] single-fit: product {pid} done")

    return {"actual": np.concatenate(actuals), "predicted": np.concatenate(preds), "failures": failures}


def run_rolling_origin(df: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Refit every HORIZON_WEEKS on realized history, forecasting the next block."""
    test_weeks = sorted(test["week_start"].unique())
    origins = test_weeks[::HORIZON_WEEKS]
    assert len(origins) == N_ORIGINS, f"Expected {N_ORIGINS} origins, got {len(origins)}"

    actuals, preds, failures = [], [], []
    products = sorted(test["product_id"].unique())

    for i, pid in enumerate(products, 1):
        product_df = df.loc[df["product_id"] == pid, ["week_start", "units_sold"]]

        for origin in origins:
            block_end = origin + pd.Timedelta(weeks=HORIZON_WEEKS)
            history = product_df[product_df["week_start"] < origin]
            block = product_df[
                (product_df["week_start"] >= origin) & (product_df["week_start"] < block_end)
            ]

            pred = _fit_predict(history, pd.DatetimeIndex(block["week_start"]))
            if pred is None:
                failures.append((pid, origin.date().isoformat()))
            else:
                actuals.append(block["units_sold"].to_numpy())
                preds.append(pred)

        print(f"  [{i}/{len(products)}] rolling-origin: product {pid} done "
              f"({i * N_ORIGINS}/{len(products) * N_ORIGINS} fits)")

    return {"actual": np.concatenate(actuals), "predicted": np.concatenate(preds), "failures": failures}


def main() -> None:
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run: uv run python forecasting_service/src/train.py"
        )
    saved = json.loads(metrics_path.read_text())

    df = load_demand()
    train, test = split_by_date(df)

    print(f"Fitting rolling-origin Prophet "
          f"({test['product_id'].nunique()} products x {N_ORIGINS} origins = "
          f"{test['product_id'].nunique() * N_ORIGINS} fits, several minutes)...")
    rolling = run_rolling_origin(df, test)
    rolling_scores = evaluate(pd.Series(rolling["actual"]), pd.Series(rolling["predicted"]))

    print()
    print(f"Fitting single-fit Prophet ({test['product_id'].nunique()} fits)...")
    single = run_single_fit(train, test)
    single_scores = evaluate(pd.Series(single["actual"]), pd.Series(single["predicted"]))

    print()
    total_rolling_fits = test["product_id"].nunique() * N_ORIGINS
    if rolling["failures"]:
        print(f"Rolling-origin: {len(rolling['failures'])}/{total_rolling_fits} fits failed "
              f"and were excluded from the aggregate (rows dropped, not zero-filled): "
              f"{rolling['failures'][:10]}{' ...' if len(rolling['failures']) > 10 else ''}")
    else:
        print(f"Rolling-origin: 0/{total_rolling_fits} fits failed.")
    if single["failures"]:
        print(f"Single-fit: {len(single['failures'])}/{test['product_id'].nunique()} products "
              f"failed and were excluded from the aggregate: {single['failures']}")
    else:
        print(f"Single-fit: 0/{test['product_id'].nunique()} products failed.")

    rows = [
        ("naive (4 weeks ago)", saved["baselines"]["naive (4 weeks ago)"]),
        ("seasonal naive (same week last year)", saved["baselines"]["seasonal naive (same week last year)"]),
        ("4-week moving average", saved["baselines"]["4-week moving average"]),
        ("LightGBM", saved["model"]),
        ("Prophet, rolling-origin, 4-week horizon", rolling_scores),
        ("Prophet, single fit, 52-week horizon", single_scores),
    ]

    print()
    print("RESULTS")
    print(f"{'Model':<42} {'WAPE':>8} {'MAPE':>8} {'MAE':>8} {'Bias':>8}")
    print("-" * 78)
    for label, scores in rows:
        print(f"{label:<42} {scores['wape']:>7.1%} {scores['mape']:>7.1%} "
              f"{scores['mae']:>8.1f} {scores['bias']:>+8.1f}")

    gap = 100 * (single_scores["wape"] - rolling_scores["wape"])
    print()
    print(f"Gap between the two Prophet rows: {gap:+.1f} WAPE pts "
          f"(single-fit minus rolling-origin) -- this is the cost of the "
          f"52-week horizon versus the method itself.")

    prophet_metrics = {
        "generated_date": date.today().isoformat(),
        "horizon_weeks": HORIZON_WEEKS,
        "n_origins": N_ORIGINS,
        "rolling_origin": rolling_scores,
        "rolling_origin_failures": rolling["failures"],
        "single_fit": single_scores,
        "single_fit_failures": single["failures"],
    }
    metrics_out = MODEL_DIR / "prophet_metrics.json"
    metrics_out.write_text(json.dumps(prophet_metrics, indent=2))
    print(f"\nSaved Prophet metrics to {metrics_out}")


if __name__ == "__main__":
    main()
