"""Transform the UCI Online Retail II CSV into the six-table schema and
bulk-load it into Postgres.

Idempotent: truncates the target tables first, so it can be re-run freely.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.database import engine

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "online_retail_ii.csv"

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

    valid = (
        (df["Price"] > 0)
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


def main() -> None:
    print(f"Reading {CSV_PATH} ...")
    df = load_and_clean(CSV_PATH)
    print(f"Cleaned rows: {len(df):,}")

    rng = np.random.default_rng(42)

    customers = build_customers(df)
    products = build_products(df)
    orders = build_orders(df, rng)
    order_items = build_order_items(df)
    returns = build_returns(df, rng)

    conn = engine.raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE customers, products, orders, order_items, returns, agent_actions "
                "RESTART IDENTITY CASCADE"
            )

        copy_dataframe(conn, customers, "customers", ["customer_id", "name", "email", "country"])
        copy_dataframe(conn, products, "products", ["product_id", "name", "category", "unit_price"])
        copy_dataframe(conn, orders, "orders", ["order_id", "customer_id", "order_date", "status"])
        copy_dataframe(
            conn,
            order_items,
            "order_items",
            ["order_item_id", "order_id", "product_id", "quantity", "price"],
        )
        copy_dataframe(
            conn,
            returns,
            "returns",
            ["return_id", "order_id", "reason", "status", "idempotency_key"],
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print("\nRow counts:")
    print(f"  customers:   {len(customers):,}")
    print(f"  products:    {len(products):,}")
    print(f"  orders:      {len(orders):,}")
    print(f"  order_items: {len(order_items):,}")
    print(f"  returns:     {len(returns):,}")


if __name__ == "__main__":
    main()
