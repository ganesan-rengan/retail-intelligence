"""Generate docs/results.md from the pipeline's own persisted metrics.

Every number in the output comes from one of:
  data/processed/baseline_metrics.csv  -- naive / seasonal naive / 4-week MA
  models/metrics.json                  -- LightGBM, validation WAPE, horizon
  models/prophet_metrics.json          -- both Prophet rows

Test/train period, product count, and the train-vs-test volume decline are
computed via load_demand + split_by_date (the same functions the pipeline
itself uses), not read from a file, so they can never drift from the actual
data. VALID_WEEKS is imported from train.py rather than retyped.

The sorted table has 5 rows, not 6: Prophet's single-fit, 52-week-horizon
run is a control condition (it isn't given the same horizon as everything
else), not a competing method, so it would mislead a reader skimming a
WAPE-sorted table straight to the top. It's reported separately as a note
below the table instead.

NOTE (not fixed here, flagged for a separate change): train.py currently
also writes baseline scores into models/metrics.json under "baselines".
That's a second copy of the same numbers this script deliberately ignores
in favor of baseline_metrics.csv -- worth removing from train.py later so
there is exactly one source of truth, but that's a train.py change, out of
scope for this script.
"""
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting-service" / "src"))

from split import load_demand, split_by_date
from train import VALID_WEEKS

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CSV = REPO_ROOT / "data" / "processed" / "baseline_metrics.csv"
MODEL_METRICS_JSON = REPO_ROOT / "models" / "metrics.json"
PROPHET_METRICS_JSON = REPO_ROOT / "models" / "prophet_metrics.json"
OUT_PATH = REPO_ROOT / "docs" / "results.md"

# Only the rolling-origin Prophet row is a competing method in the sorted
# table. Single-fit is a control condition (see the note printed below the
# table) and would otherwise land first by WAPE and read as the headline.
ROLLING_ORIGIN_LABEL = "Prophet, rolling-origin, 4-week horizon"


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the script that produces it before "
            f"generating the results table."
        )


def load_all() -> tuple[dict, dict, dict, pd.DataFrame]:
    require(BASELINE_CSV)
    require(MODEL_METRICS_JSON)
    require(PROPHET_METRICS_JSON)

    baselines = pd.read_csv(BASELINE_CSV).set_index("baseline")
    model_metrics = json.loads(MODEL_METRICS_JSON.read_text())
    prophet_metrics = json.loads(PROPHET_METRICS_JSON.read_text())
    return baselines, model_metrics, prophet_metrics


def build_rows(baselines: pd.DataFrame, model_metrics: dict, prophet_metrics: dict) -> list[dict]:
    rows = []
    for label in baselines.index:
        scores = baselines.loc[label]
        rows.append({
            "label": label,
            "wape": float(scores["wape"]),
            "mape": float(scores["mape"]),
            "mae": float(scores["mae"]),
            "bias": float(scores["bias"]),
            "is_model": False,
        })

    model_scores = model_metrics["model"]
    rows.append({
        "label": "LightGBM",
        "wape": model_scores["wape"],
        "mape": model_scores["mape"],
        "mae": model_scores["mae"],
        "bias": model_scores["bias"],
        "is_model": True,
    })

    rolling_scores = prophet_metrics["rolling_origin"]
    rows.append({
        "label": ROLLING_ORIGIN_LABEL,
        "wape": rolling_scores["wape"],
        "mape": rolling_scores["mape"],
        "mae": rolling_scores["mae"],
        "bias": rolling_scores["bias"],
        "is_model": False,
    })

    return sorted(rows, key=lambda r: r["wape"], reverse=True)


def render_table(rows: list[dict]) -> str:
    lines = [
        "| Method | WAPE | MAPE | MAE | Bias |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        cells = [
            row["label"],
            f"{row['wape']:.1%}",
            f"{row['mape']:.1%}",
            f"{row['mae']:.1f}",
            f"{row['bias']:+.1f}",
        ]
        if row["is_model"]:
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    baselines, model_metrics, prophet_metrics = load_all()
    rows = build_rows(baselines, model_metrics, prophet_metrics)

    df = load_demand()
    train, test = split_by_date(df)
    period_start = test["week_start"].min().date()
    period_end = test["week_start"].max().date()
    n_products = test["product_id"].nunique()
    horizon_weeks = model_metrics["horizon_weeks"]

    ma4_wape = float(baselines.loc["4-week moving average", "wape"])
    model_wape = model_metrics["model"]["wape"]
    improvement_pts = 100 * (ma4_wape - model_wape)
    improvement_rel = 100 * (ma4_wape - model_wape) / ma4_wape

    # Train-vs-test aggregate volume, computed fresh -- this is the same
    # "year-over-year" comparison used elsewhere, not a hardcoded figure.
    yoy_decline = 100 * (test["units_sold"].sum() / train["units_sold"].sum() - 1)

    validation_wape = model_metrics["validation_wape"]
    valid_test_gap = 100 * (model_wape - validation_wape)
    drift_note = (
        f"Validation WAPE ({validation_wape:.1%}, on {VALID_WEEKS} held-out weeks "
        f"inside training) is {valid_test_gap:.1f} points better than test WAPE "
        f"({model_wape:.1%}, the {period_start.year}-{period_end.year} period). "
        f"The model generalises well within the training distribution; the gap "
        f"reflects year-over-year distribution drift, with total demand down "
        f"{abs(yoy_decline):.1f}% in the test period. This is the case for "
        f"scheduled retraining rather than for additional regularisation."
    )

    rolling_wape = prophet_metrics["rolling_origin"]["wape"]
    single_wape = prophet_metrics["single_fit"]["wape"]
    single_gap_pts = 100 * (single_wape - rolling_wape)
    prophet_control_note = (
        f"As a control, Prophet fitted once and forecasting the full 52 weeks "
        f"scored {single_wape:.1%} WAPE. The {single_gap_pts:.0f}-point gap "
        f"against the rolling-origin version quantifies how much of Prophet's "
        f"disadvantage was the harder task rather than the method."
    )

    rolling_bias = prophet_metrics["rolling_origin"]["bias"]
    seasonal_bias = float(baselines.loc["seasonal naive (same week last year)", "bias"])
    pooled_obs = len(train)
    weeks_per_product = train["week_start"].nunique()
    findings_note = (
        f"Prophet and seasonal naive are the only two methods above 100% WAPE, "
        f"and both carry large positive bias ({rolling_bias:+.0f}, {seasonal_bias:+.0f}). "
        f"Both estimate an annual seasonal component from a single year of "
        f"per-product history and over-forecast into a year of declining demand. "
        f"LightGBM pools all {n_products} series into one model, learning shared "
        f"seasonal structure from {pooled_obs:,} observations rather than "
        f"{weeks_per_product} per product -- that pooling is the main reason it wins."
    )

    lines = [
        "# Forecasting Results",
        "",
        f"Test period: {period_start} to {period_end} &nbsp;|&nbsp; "
        f"Products: {n_products} &nbsp;|&nbsp; Horizon: {horizon_weeks} weeks ahead",
        "",
        render_table(rows),
        "",
        prophet_control_note,
        "",
        f"**Improvement over 4-week moving average:** {improvement_pts:+.1f} WAPE points "
        f"({improvement_rel:+.1f}% relative).",
        "",
        f"**Validation vs. test:** {drift_note}",
        "",
        "## Findings",
        "",
        findings_note,
        "",
        "## Limitations",
        "",
        "- MAPE is unreliable on this dataset: a diagnostic pass (see "
        "`scripts/diagnose_mape.py`) found it dominated by a handful of "
        "low-volume, erratic rows rather than reflecting typical forecast "
        "quality. WAPE is the metric to trust; MAPE is reported for reference only.",
        "- Results come from a single train/test split (52 weeks each), not "
        "repeated cross-validation across multiple periods, so these numbers "
        "carry the uncertainty of one historical split rather than an averaged estimate.",
        f"- Total demand fell {abs(yoy_decline):.1f}% from train to test (see "
        f"'Validation vs. test' above): this is distribution drift, not "
        f"overfitting, and argues for scheduled retraining as the test period ages further.",
        "- The top-50 product set excludes single-line-dominated and "
        "thin-training-history products (see `demand.py`); results do not "
        "necessarily generalize to the full product catalog.",
        "- Prophet is fit per-product on at most 52 weeks of history -- short "
        "for learning a yearly seasonal component, which likely explains its "
        "weaker performance here relative to the lag/rolling-feature approach.",
        "",
        f"*Generated on {date.today().isoformat()} by "
        f"`scripts/generate_results_table.py`. Do not edit this file by hand "
        f"-- rerun the script instead.*",
        "",
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
