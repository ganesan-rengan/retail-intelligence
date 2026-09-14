"""Sanity-check /forecast against recent actuals, for one product or a sample.

Starts the real service as a subprocess and calls it over HTTP, so every
check goes through the actual serving path (routing, request validation,
response serialization) -- not just the internal service module. The
"actual weekly values" side of the comparison is read directly from the
database using service._build_product_frame(), the same zero-filled window
the model itself is fed, so what we compare against is what the model saw,
not a differently-shaped ad hoc query.

A forecast is flagged if it falls outside [0.3x, 3x] of the trailing 4-week
mean (the same 4 weeks roll_mean_4 averages: t-4..t-1 relative to the
target week). That is a coarse plausibility check, not a correctness proof
-- a flagged product may just be entering a real seasonal swing.

Usage:
    uv run python scripts/sanity_check_forecast.py --product-id 85123A
    uv run python scripts/sanity_check_forecast.py --n 10
    uv run python scripts/sanity_check_forecast.py --n 50   # all trained products
"""
import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "forecasting_service" / "src"))

from forecasting_service.app import service  # noqa: E402
from shared.database import get_session  # noqa: E402

HOST = "127.0.0.1"
DEFAULT_PORT = 8931
STARTUP_TIMEOUT_S = 30
LOW_MULTIPLIER = 0.3
HIGH_MULTIPLIER = 3.0


def start_server(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "forecasting_service.app.main:app",
            "--host", HOST, "--port", str(port), "--log-level", "warning",
        ],
        cwd=REPO_ROOT,
    )
    base_url = f"http://{HOST}:{port}"
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server process exited early with code {proc.returncode}")
        try:
            if requests.get(f"{base_url}/health", timeout=1).status_code in (200, 503):
                return proc
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"Server did not become ready within {STARTUP_TIMEOUT_S}s")


def trailing_actuals(db, product_id: str) -> list[int]:
    """Last 8 real weekly values ending at the latest complete week."""
    latest_week = service.resolve_latest_week(db)
    frame = service._build_product_frame(db, product_id, latest_week)
    real_weeks = frame.iloc[:-1]  # drop the synthetic target-week row
    return [int(v) for v in real_weeks["units_sold"].tail(8).tolist()]


def check_product(db, base_url: str, product_id: str) -> dict:
    actuals = trailing_actuals(db, product_id)
    trailing_mean = sum(actuals[-4:]) / 4

    resp = requests.post(f"{base_url}/forecast", json={"product_id": product_id}, timeout=30)
    resp.raise_for_status()
    predicted = resp.json()["predicted_units"]

    low, high = LOW_MULTIPLIER * trailing_mean, HIGH_MULTIPLIER * trailing_mean
    flagged = not (low <= predicted <= high)

    return {
        "product_id": product_id,
        "actuals": actuals,
        "trailing_mean": trailing_mean,
        "predicted": predicted,
        "bounds": (low, high),
        "flagged": flagged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--product-id", help="Check a single product instead of a sample")
    parser.add_argument("--n", type=int, default=5, help="Sample size when --product-id is not given (default 5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    base_url = f"http://{HOST}:{args.port}"

    print(f"Starting service on {base_url} ...")
    proc = start_server(args.port)
    try:
        trained_products = requests.get(f"{base_url}/products", timeout=10).json()["products"]

        if args.product_id:
            if args.product_id not in trained_products:
                raise SystemExit(f"'{args.product_id}' is not one of the trained products.")
            targets = [args.product_id]
        else:
            rng = random.Random(args.seed)
            targets = rng.sample(trained_products, min(args.n, len(trained_products)))

        db = get_session()
        try:
            results = [check_product(db, base_url, pid) for pid in targets]
        finally:
            db.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    header = f"{'product_id':<10} {'last 8 actuals':<45} {'trail_mean':>10} {'predicted':>10} {'bounds':>18}  flag"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        bounds = f"[{r['bounds'][0]:.1f}, {r['bounds'][1]:.1f}]"
        flag = "FLAG" if r["flagged"] else "ok"
        print(
            f"{r['product_id']:<10} {str(r['actuals']):<45} "
            f"{r['trailing_mean']:>10.1f} {r['predicted']:>10.1f} {bounds:>18}  {flag}"
        )

    flagged = [r for r in results if r["flagged"]]
    print()
    print(f"{len(flagged)}/{len(results)} flagged")
    if flagged:
        print("Flagged products:", ", ".join(r["product_id"] for r in flagged))


if __name__ == "__main__":
    main()
