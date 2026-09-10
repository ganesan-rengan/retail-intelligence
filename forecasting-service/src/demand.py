"""Build the weekly demand table used for forecasting."""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.database import engine

TOP_N_PRODUCTS = 50

TOP_PRODUCTS_SQL = text("""
    SELECT oi.product_id, SUM(oi.quantity) AS total_units
    FROM order_items oi
    JOIN orders o ON o.order_id = oi.order_id
    WHERE o.status != 'cancelled'
    GROUP BY oi.product_id
    ORDER BY total_units DESC
    LIMIT :top_n
""")

WEEKLY_SALES_SQL = text("""
    SELECT
        oi.product_id,
        DATE_TRUNC('week', o.order_date)::date AS week_start,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON o.order_id = oi.order_id
    WHERE o.status != 'cancelled'
      AND oi.product_id = ANY(:product_ids)
    GROUP BY oi.product_id, DATE_TRUNC('week', o.order_date)
    ORDER BY oi.product_id, week_start
""")


def get_top_products(n: int = TOP_N_PRODUCTS) -> list[str]:
    """Return the n highest-volume product ids."""
    with engine.connect() as conn:
        rows = conn.execute(TOP_PRODUCTS_SQL, {"top_n": n}).fetchall()
    return [row.product_id for row in rows]


def build_weekly_demand(product_ids: list[str]) -> pd.DataFrame:
    """Return one row per product per week, zero-filled, edges trimmed."""
    with engine.connect() as conn:
        df = pd.read_sql(WEEKLY_SALES_SQL, conn, params={"product_ids": product_ids})

    df["week_start"] = pd.to_datetime(df["week_start"])

    # Full product x week grid so absent weeks become explicit zeros
    all_weeks = pd.date_range(
        df["week_start"].min(), df["week_start"].max(), freq="W-MON"
    )
    grid = pd.MultiIndex.from_product(
        [product_ids, all_weeks], names=["product_id", "week_start"]
    )

    df = (
        df.set_index(["product_id", "week_start"])
        .reindex(grid, fill_value=0)
        .reset_index()
    )

    # Drop partial weeks at both ends
    first_week, last_week = df["week_start"].min(), df["week_start"].max()
    df = df[(df["week_start"] > first_week) & (df["week_start"] < last_week)]

    df["units_sold"] = df["units_sold"].astype(int)
    return df.sort_values(["product_id", "week_start"]).reset_index(drop=True)


def main() -> None:
    products = get_top_products()
    print(f"Top {len(products)} products selected")

    df = build_weekly_demand(products)

    print(f"Rows: {len(df):,}")
    print(f"Products: {df['product_id'].nunique()}")
    print(f"Weeks: {df['week_start'].nunique()}")
    print(f"Date range: {df['week_start'].min().date()} -> {df['week_start'].max().date()}")
    print(f"Zero-sales weeks: {(df['units_sold'] == 0).sum():,} "
          f"({100 * (df['units_sold'] == 0).mean():.1f}%)")
    print()
    print(df.head(10))

    out = Path(__file__).resolve().parents[2] / "data" / "processed" / "weekly_demand.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()