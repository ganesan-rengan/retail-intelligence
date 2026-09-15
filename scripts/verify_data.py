"""Run verification queries against the loaded retail data and report
PASS/FAIL per check, plus an informational summary.

Exits with status 1 if any check fails.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Connection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.database import get_engine


@dataclass
class CheckResult:
    id: int
    name: str
    passed: bool
    detail: str


def check_01_tables_have_rows(conn: Connection) -> CheckResult:
    rows = conn.execute(
        text(
            """
            SELECT * FROM (
                SELECT 1 AS ord, 'customers' AS table_name, count(*) AS n FROM customers
                UNION ALL SELECT 2, 'products', count(*) FROM products
                UNION ALL SELECT 3, 'orders', count(*) FROM orders
                UNION ALL SELECT 4, 'order_items', count(*) FROM order_items
                UNION ALL SELECT 5, 'returns', count(*) FROM returns
            ) t ORDER BY ord
            """
        )
    ).all()
    empty = [r.table_name for r in rows if r.n == 0]
    passed = not empty
    detail = "all tables have rows" if passed else f"empty tables: {', '.join(empty)}"
    return CheckResult(1, "Every table has rows", passed, detail)


def check_02_guest_customer_exists(conn: Connection) -> CheckResult:
    n = conn.execute(
        text("SELECT count(*) FROM customers WHERE customer_id = 0 AND name = 'Guest Checkout'")
    ).scalar_one()
    return CheckResult(2, "Guest customer exists (id=0)", n == 1, f"matching rows: {n}")


def check_03_orders_customer_fk(conn: Connection) -> CheckResult:
    n = conn.execute(
        text(
            """
            SELECT count(*) FROM orders o
            LEFT JOIN customers c ON c.customer_id = o.customer_id
            WHERE c.customer_id IS NULL
            """
        )
    ).scalar_one()
    return CheckResult(3, "No order references a missing customer", n == 0, f"violations: {n}")


def check_04_order_items_product_fk(conn: Connection) -> CheckResult:
    n = conn.execute(
        text(
            """
            SELECT count(*) FROM order_items oi
            LEFT JOIN products p ON p.product_id = oi.product_id
            WHERE p.product_id IS NULL
            """
        )
    ).scalar_one()
    return CheckResult(4, "No order_item references a missing product", n == 0, f"violations: {n}")


def check_05_order_items_order_fk(conn: Connection) -> CheckResult:
    n = conn.execute(
        text(
            """
            SELECT count(*) FROM order_items oi
            LEFT JOIN orders o ON o.order_id = oi.order_id
            WHERE o.order_id IS NULL
            """
        )
    ).scalar_one()
    return CheckResult(5, "No order_item references a missing order", n == 0, f"violations: {n}")


def check_06_no_items_on_cancelled_orders(conn: Connection) -> CheckResult:
    n = conn.execute(
        text(
            """
            SELECT count(*) FROM order_items oi
            JOIN orders o ON o.order_id = oi.order_id
            WHERE o.status = 'cancelled'
            """
        )
    ).scalar_one()
    return CheckResult(6, "No order_items belong to cancelled orders", n == 0, f"violations: {n}")


def check_07_returns_reference_cancelled_orders(conn: Connection) -> CheckResult:
    n = conn.execute(
        text(
            """
            SELECT count(*) FROM returns r
            JOIN orders o ON o.order_id = r.order_id
            WHERE o.status <> 'cancelled'
            """
        )
    ).scalar_one()
    return CheckResult(7, "Every return references a cancelled order", n == 0, f"violations: {n}")


def check_08_products_category_not_null(conn: Connection) -> CheckResult:
    n = conn.execute(text("SELECT count(*) FROM products WHERE category IS NULL")).scalar_one()
    return CheckResult(8, "No product has a NULL category", n == 0, f"violations: {n}")


def check_09_order_items_quantity_price_positive(conn: Connection) -> CheckResult:
    n = conn.execute(
        text("SELECT count(*) FROM order_items WHERE quantity <= 0 OR price <= 0")
    ).scalar_one()
    return CheckResult(
        9, "All order_items have quantity > 0 and price > 0", n == 0, f"violations: {n}"
    )


def check_10_order_status_values(conn: Connection) -> CheckResult:
    n = conn.execute(
        text(
            """
            SELECT count(*) FROM orders
            WHERE status NOT IN ('delivered', 'shipped', 'pending', 'cancelled')
            """
        )
    ).scalar_one()
    return CheckResult(
        10, "Order status values are within the allowed set", n == 0, f"violations: {n}"
    )


def check_11_return_status_values(conn: Connection) -> CheckResult:
    n = conn.execute(
        text(
            """
            SELECT count(*) FROM returns
            WHERE status NOT IN ('completed', 'approved', 'rejected')
            """
        )
    ).scalar_one()
    return CheckResult(
        11, "Return status values are within the allowed set", n == 0, f"violations: {n}"
    )


def check_12_order_date_range(conn: Connection) -> CheckResult:
    n = conn.execute(
        text(
            """
            SELECT count(*) FROM orders
            WHERE order_date::date < DATE '2009-12-01' OR order_date::date > DATE '2011-12-10'
            """
        )
    ).scalar_one()
    return CheckResult(
        12, "order_date falls within 2009-12-01..2011-12-10", n == 0, f"out-of-range orders: {n}"
    )


def check_13_order_count_reconciles(conn: Connection) -> CheckResult:
    row = conn.execute(
        text(
            """
            SELECT
                (SELECT count(*) FROM orders) AS total_orders,
                (SELECT count(DISTINCT order_id) FROM order_items) AS orders_with_items,
                (SELECT count(*) FROM orders WHERE status = 'cancelled') AS cancelled_orders,
                (SELECT count(*) FROM orders o
                    WHERE o.status <> 'cancelled'
                    AND NOT EXISTS (SELECT 1 FROM order_items oi WHERE oi.order_id = o.order_id)
                ) AS orphaned_orders
            """
        )
    ).one()
    expected = row.orders_with_items + row.cancelled_orders + row.orphaned_orders
    passed = row.total_orders == expected
    detail = (
        f"total_orders={row.total_orders}, orders_with_items={row.orders_with_items}, "
        f"cancelled_orders={row.cancelled_orders}, orphaned_orders={row.orphaned_orders}"
    )
    return CheckResult(13, "Order count reconciles", passed, detail)


CHECKS = [
    check_01_tables_have_rows,
    check_02_guest_customer_exists,
    check_03_orders_customer_fk,
    check_04_order_items_product_fk,
    check_05_order_items_order_fk,
    check_06_no_items_on_cancelled_orders,
    check_07_returns_reference_cancelled_orders,
    check_08_products_category_not_null,
    check_09_order_items_quantity_price_positive,
    check_10_order_status_values,
    check_11_return_status_values,
    check_12_order_date_range,
    check_13_order_count_reconciles,
]


def info_row_counts(conn: Connection) -> list[list[str]]:
    rows = conn.execute(
        text(
            """
            SELECT * FROM (
                SELECT 1 AS ord, 'customers' AS table_name, count(*) AS n FROM customers
                UNION ALL SELECT 2, 'products', count(*) FROM products
                UNION ALL SELECT 3, 'orders', count(*) FROM orders
                UNION ALL SELECT 4, 'order_items', count(*) FROM order_items
                UNION ALL SELECT 5, 'returns', count(*) FROM returns
            ) t ORDER BY ord
            """
        )
    ).all()
    return [[r.table_name, f"{r.n:,}"] for r in rows]


def info_category_distribution(conn: Connection) -> list[list[str]]:
    rows = conn.execute(
        text(
            """
            SELECT p.category, sum(oi.quantity) AS units,
                   round(100.0 * sum(oi.quantity) / sum(sum(oi.quantity)) OVER (), 2) AS pct
            FROM order_items oi
            JOIN products p ON p.product_id = oi.product_id
            GROUP BY p.category
            ORDER BY units DESC
            """
        )
    ).all()
    return [[r.category, f"{r.units:,}", f"{r.pct}%"] for r in rows]


def info_order_status_distribution(conn: Connection) -> list[list[str]]:
    rows = conn.execute(
        text(
            """
            SELECT status, count(*) AS orders,
                   round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
            FROM orders
            GROUP BY status
            ORDER BY orders DESC
            """
        )
    ).all()
    return [[r.status, f"{r.orders:,}", f"{r.pct}%"] for r in rows]


def info_top_products(conn: Connection) -> list[list[str]]:
    rows = conn.execute(
        text(
            """
            SELECT p.product_id, p.name, sum(oi.quantity) AS units
            FROM order_items oi
            JOIN products p ON p.product_id = oi.product_id
            GROUP BY p.product_id, p.name
            ORDER BY units DESC
            LIMIT 10
            """
        )
    ).all()
    return [[r.product_id, r.name, f"{r.units:,}"] for r in rows]


def info_orphaned_orders(conn: Connection) -> list[list[str]]:
    """Check 14 (info only): non-cancelled orders with zero order_items."""
    row = conn.execute(
        text(
            """
            SELECT
                count(*) FILTER (
                    WHERE status <> 'cancelled'
                    AND NOT EXISTS (SELECT 1 FROM order_items oi WHERE oi.order_id = o.order_id)
                ) AS orphaned,
                count(*) AS total
            FROM orders o
            """
        )
    ).one()
    pct = round(100.0 * row.orphaned / row.total, 2) if row.total else 0.0
    return [
        ["non-cancelled orders with zero order_items", f"{row.orphaned:,}"],
        ["% of all orders", f"{pct}%"],
    ]


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    with get_engine().connect() as conn:
        results = [check(conn) for check in CHECKS]

        print("CHECKS")
        print_table(
            ["#", "Check", "Result", "Detail"],
            [[r.id, r.name, "PASS" if r.passed else "FAIL", r.detail] for r in results],
        )

        print("\nINFO\n")
        print("Row count per table:")
        print_table(["Table", "Rows"], info_row_counts(conn))

        print("\nCategory distribution by units:")
        print_table(["Category", "Units", "% of units"], info_category_distribution(conn))

        print("\nOrder status distribution:")
        print_table(["Status", "Orders", "% of orders"], info_order_status_distribution(conn))

        print("\nTop 10 products by units sold:")
        print_table(["Product ID", "Name", "Units"], info_top_products(conn))

        print("\nCheck 14 (info only) - non-cancelled orders with zero order_items:")
        print_table(["Metric", "Value"], info_orphaned_orders(conn))

    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n{len(failed)} check(s) FAILED.")
        sys.exit(1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
