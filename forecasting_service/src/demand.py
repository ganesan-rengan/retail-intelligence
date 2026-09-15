"""Build the weekly demand table used for forecasting."""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.database import get_engine
from split import SPLIT_DATE

TOP_N_PRODUCTS = 50

# Exclude products with no learnable demand pattern: where a single line item
# dominates total volume. Products 23843 (100%), 23166 (95%) and 37410 (75%)
# have too few transactions to model. Bulk-wholesale SKUs like 21980-21984
# sit near 30-40% but have 500+ line items each, so they are kept.
MAX_SINGLE_LINE_SHARE = 0.50

# Exclude products with too little presence in the training period to learn
# a pattern from (e.g. a product that only starts selling near/after the
# train/test boundary). Selection itself must also only look at training
# orders -- otherwise "top 50" is picked using future information.
MIN_WEEKS_ACTIVE_SHARE = 0.25

TRAIN_WEEKS_SQL = text("""
    SELECT COUNT(DISTINCT DATE_TRUNC('week', order_date)) AS train_weeks
    FROM orders
    WHERE status != 'cancelled' AND order_date < :split_date
""")

TOP_PRODUCTS_SQL = text("""
    WITH product_stats AS (
        SELECT
            oi.product_id,
            SUM(oi.quantity) AS total_units,
            MAX(oi.quantity) AS max_line,
            COUNT(*) AS line_items,
            COUNT(DISTINCT DATE_TRUNC('week', o.order_date)) AS weeks_active
        FROM order_items oi
        JOIN orders o ON o.order_id = oi.order_id
        WHERE o.status != 'cancelled'
          AND o.order_date < :split_date
        GROUP BY oi.product_id
    )
    SELECT
        product_id,
        total_units,
        max_line,
        line_items,
        weeks_active,
        max_line::numeric / total_units AS single_line_share
    FROM product_stats
    ORDER BY total_units DESC
    LIMIT :candidate_n
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
    """Return the n highest-volume product ids, ranked on training-period
    orders only, excluding single-line outliers and thin-history products."""
    with get_engine().connect() as conn:
        train_weeks = conn.execute(
            TRAIN_WEEKS_SQL, {"split_date": SPLIT_DATE}
        ).scalar()
        rows = conn.execute(
            TOP_PRODUCTS_SQL, {"candidate_n": n + 20, "split_date": SPLIT_DATE}
        ).fetchall()

    min_weeks_active = MIN_WEEKS_ACTIVE_SHARE * train_weeks

    kept, excluded = [], []
    for row in rows:
        if row.single_line_share > MAX_SINGLE_LINE_SHARE:
            excluded.append((row, "single line item share"))
        elif row.weeks_active < min_weeks_active:
            excluded.append((row, "thin training-period presence"))
        else:
            kept.append(row)

    if excluded:
        print(f"Excluded {len(excluded)} product(s) "
              f"(candidates ranked on {train_weeks} training weeks only):")
        for row, reason in excluded:
            print(f"  {row.product_id:<10} {reason:<28} "
                  f"{row.total_units:>8,} train units, "
                  f"active {row.weeks_active:>3}/{train_weeks} weeks, "
                  f"single-line {row.single_line_share:.1%}")
        print()

    return [row.product_id for row in kept[:n]]


def build_weekly_demand(product_ids: list[str]) -> pd.DataFrame:
    """Return one row per product per week, zero-filled, edges trimmed."""
    with get_engine().connect() as conn:
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
    out = Path(__file__).resolve().parents[2] / "data" / "processed" / "weekly_demand.csv"

    old_products: set[str] = set()
    if out.exists():
        old_products = set(pd.read_csv(out, usecols=["product_id"])["product_id"].astype(str).unique())

    products = get_top_products()
    print(f"Top {len(products)} products selected")

    new_products = set(products)
    entered, left = new_products - old_products, old_products - new_products
    if old_products:
        print(f"Entered top-50 ({len(entered)}): {sorted(entered)}")
        print(f"Left top-50    ({len(left)}): {sorted(left)}")
        print()

    df = build_weekly_demand(products)

    print(f"Rows: {len(df):,}")
    print(f"Products: {df['product_id'].nunique()}")
    print(f"Weeks: {df['week_start'].nunique()}")
    print(f"Date range: {df['week_start'].min().date()} -> {df['week_start'].max().date()}")
    print(f"Zero-sales weeks: {(df['units_sold'] == 0).sum():,} "
          f"({100 * (df['units_sold'] == 0).mean():.1f}%)")
    print()
    print(df.head(10))

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()