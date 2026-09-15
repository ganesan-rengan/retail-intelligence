"""Transform the UCI Online Retail II CSV into the six-table schema and
bulk-load it into Postgres, local or remote.

Resumable, not just idempotent: each table loads in its own transaction and
is skipped if it already holds exactly the expected row count. A table with
a nonzero but wrong count is refused rather than guessed at -- that shape
only happens when an earlier run died mid-COPY, and silently overwriting or
skipping it would hide a partial load. Pass --truncate for a clean full
reload instead of resolving that by hand.
"""

import argparse
import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from faker import Faker

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "online_retail_ii.csv"

TABLES = ["customers", "products", "orders", "order_items", "returns"]
TRUNCATE_TABLES = [*TABLES, "agent_actions"]

COLUMNS = {
    "customers": ["customer_id", "name", "email", "country"],
    "products": ["product_id", "name", "category", "unit_price"],
    "orders": ["order_id", "customer_id", "order_date", "status"],
    "order_items": ["order_item_id", "order_id", "product_id", "quantity", "price"],
    "returns": ["return_id", "order_id", "reason", "status", "idempotency_key"],
}

RETURN_REASONS = [
    "damaged in transit",
    "wrong item received",
    "no longer needed",
    "changed mind",
    "quality issue",
]

# First keyword match wins; default category is "Other".
# Ordered so more specific categories are checked before broader ones
# (Party before Stationery, Frames before Storage, Bags before Toys).
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Christmas": ["CHRISTMAS", "XMAS", "ADVENT", "SANTA", "REINDEER"],
    "Party": [
        "TISSUE",
        "NAPKIN",
        "PAPER CUP",
        "PAPER PLATE",
        "PARTY",
        "BALLOON",
        "PAPER CHAIN",
        "DOILY",
        "STRAW",
        "CRAYON",
        "PAPER CRAFT",
    ],
    "Signs": ["SIGN", "PLAQUE", "DOORMAT", "HOOK", "WALL ART"],
    "Frames": ["FRAME", "PHOTO", "PICTURE", "MIRROR", "RECORD COVER"],
    "Bags": ["BAG", "LUNCH BOX", "SHOPPER", "PURSE", "WALLET", "CASE", "POUCH"],
    "Kitchen": ["MUG", "CAKE", "BOWL", "PLATE", "TEA", "JAR", "BAKING", "CUTLERY", "TRAY"],
    "Decorative": [
        "HEART",
        "HANGING",
        "ORNAMENT",
        "DECORATION",
        "LANTERN",
        "GARLAND",
        "FAN",
        "LEIS",
        "RIBBON",
        "CHARM",
        "LIGHTS",
        "FLOWER",
        "WREATH",
    ],
    "Candles": ["CANDLE", "T-LIGHT", "TEALIGHT", "HOLDER"],
    "Stationery": ["CARD", "NOTEBOOK", "PENCIL", "PEN", "WRAP", "GIFT TAG"],
    "Toys": [
        "TOY",
        "GAME",
        "PUZZLE",
        "DOLL",
        "BUNTING",
        "PLAYHOUSE",
        "GLIDER",
        "PAINT SET",
        "PATCHES",
        "BUBBLE GUM",
        "NIGHT LIGHT",
        "SKIPPING",
        "SPACEBOY",
    ],
    "Storage": ["BOX", "TIN", "BASKET", "DRAWER", "RACK", "HOLDER"],
}


def categorize(description: str) -> str:
    """Return the first matching category for a product description, else 'Other'."""
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in description for keyword in keywords):
            return category
    return "Other"


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Read the raw CSV, tag cancellations, and drop invalid rows."""
    df = pd.read_csv(
        csv_path,
        dtype={"Invoice": str, "StockCode": str},
        parse_dates=["InvoiceDate"],
    )
    df["customer_id"] = df["Customer ID"].fillna(0).astype(int)
    df["is_cancellation"] = df["Invoice"].str.startswith("C")

    # Price must be >= 0.005, not just > 0: order_items.price is Numeric(10,2),
    # so anything below half a cent (e.g. 0.001) would round to 0.00 on insert
    # and pass this filter while failing verify_data.py's "price > 0" check.
    valid = (
        (df["Price"] >= 0.005)
        & df["Description"].notna()
        & (df["is_cancellation"] | (df["Quantity"] > 0))
    )
    return df[valid].copy()


def build_customers(df: pd.DataFrame) -> pd.DataFrame:
    """One row per customer_id. Guest checkout (0) is fixed; others via Faker."""
    fake = Faker()
    Faker.seed(42)

    countries = (
        df.groupby("customer_id")["Country"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index(name="country")
    )

    records = []
    for customer_id, country in countries.itertuples(index=False):
        if customer_id == 0:
            records.append((0, "Guest Checkout", "guest@example.com", country))
        else:
            records.append((customer_id, fake.name(), fake.email(), country))

    return pd.DataFrame(records, columns=["customer_id", "name", "email", "country"])


def build_products(df: pd.DataFrame) -> pd.DataFrame:
    """One row per StockCode: mode description, median price, keyword-derived category."""
    grouped = df.groupby("StockCode").agg(
        name=("Description", lambda s: s.mode().iloc[0]),
        unit_price=("Price", "median"),
    )
    grouped["category"] = grouped["name"].apply(categorize)
    return grouped.reset_index().rename(columns={"StockCode": "product_id"})


def build_orders(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One row per Invoice: order_date, customer_id, and randomized/cancelled status."""
    customer_counts = df.groupby("Invoice")["customer_id"].nunique()
    bad_invoices = customer_counts[customer_counts != 1]
    if not bad_invoices.empty:
        raise ValueError(
            f"Invoices with more than one distinct customer_id: {bad_invoices.index.tolist()}"
        )

    grouped = df.groupby("Invoice").agg(
        customer_id=("customer_id", "first"),
        order_date=("InvoiceDate", "min"),
        is_cancellation=("is_cancellation", "first"),
    )

    n = len(grouped)
    random_status = rng.choice(["delivered", "shipped", "pending"], size=n, p=[0.70, 0.20, 0.10])
    grouped["status"] = np.where(grouped["is_cancellation"], "cancelled", random_status)

    return grouped.drop(columns="is_cancellation").reset_index().rename(
        columns={"Invoice": "order_id"}
    )


def build_order_items(df: pd.DataFrame) -> pd.DataFrame:
    """One row per surviving non-cancellation CSV row."""
    items = df.loc[~df["is_cancellation"], ["Invoice", "StockCode", "Quantity", "Price"]].copy()
    items.columns = ["order_id", "product_id", "quantity", "price"]
    items.insert(0, "order_item_id", range(1, len(items) + 1))
    return items


def build_returns(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One row per cancelled invoice, with a seeded reason and status."""
    cancelled_invoices = df.loc[df["is_cancellation"], "Invoice"].unique()
    reasons = rng.choice(RETURN_REASONS, size=len(cancelled_invoices))
    statuses = rng.choice(
        ["completed", "approved", "rejected"],
        size=len(cancelled_invoices),
        p=[0.80, 0.15, 0.05],
    )

    returns = pd.DataFrame({"order_id": cancelled_invoices, "reason": reasons, "status": statuses})
    returns.insert(0, "return_id", range(1, len(returns) + 1))
    returns["idempotency_key"] = None
    return returns


def copy_dataframe(conn, df: pd.DataFrame, table: str, columns: list[str]) -> None:
    """Bulk-load a DataFrame into a table via Postgres COPY."""
    buffer = io.StringIO()
    df[columns].to_csv(buffer, index=False, header=False, na_rep="\\N")
    buffer.seek(0)

    with conn.cursor() as cur:
        cur.copy_expert(
            f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')",
            buffer,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres DSN to load into. Falls back to the DATABASE_URL env var.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate all tables before loading, for a clean full reload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv()
    database_url = args.database_url or os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("No database URL: pass --database-url or set DATABASE_URL.")

    print(f"Reading {CSV_PATH} ...")
    df = load_and_clean(CSV_PATH)
    print(f"Cleaned rows: {len(df):,}")

    rng = np.random.default_rng(42)

    dataframes = {
        "customers": build_customers(df),
        "products": build_products(df),
        "orders": build_orders(df, rng),
        "order_items": build_order_items(df),
        "returns": build_returns(df, rng),
    }

    conn = psycopg2.connect(database_url)
    try:
        if args.truncate:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE {', '.join(TRUNCATE_TABLES)} RESTART IDENTITY CASCADE")
            conn.commit()
            print("Truncated all tables.")

        print("\nLoading:")
        for table in TABLES:
            table_df = dataframes[table]
            expected = len(table_df)

            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                actual = cur.fetchone()[0]

            if actual == expected:
                print(f"  {table}: skipping (already has {actual:,} rows)")
                continue

            if actual != 0:
                conn.rollback()
                raise SystemExit(
                    f"{table}: has {actual:,} rows but expected {expected:,}. This looks like "
                    f"a partial load from an earlier run -- not resuming automatically. "
                    f"Truncate it and rerun: TRUNCATE {table} RESTART IDENTITY CASCADE;"
                )

            start = time.perf_counter()
            copy_dataframe(conn, table_df, table, COLUMNS[table])
            conn.commit()
            elapsed = time.perf_counter() - start
            print(f"  {table}: loaded {expected:,} rows in {elapsed:.2f}s")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print("\nRow counts:")
    for table in TABLES:
        print(f"  {table}: {len(dataframes[table]):,}")


if __name__ == "__main__":
    main()
