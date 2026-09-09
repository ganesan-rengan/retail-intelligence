"""add status and price check constraints

Revision ID: e20608c5f822
Revises: 116533fd879c
Create Date: 2026-09-09 10:03:50.196967

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e20608c5f822'
down_revision: Union[str, Sequence[str], None] = '116533fd879c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_check_constraint(
        "ck_orders_status_valid",
        "orders",
        "status IN ('delivered', 'shipped', 'pending', 'cancelled')",
    )
    op.create_check_constraint(
        "ck_returns_status_valid",
        "returns",
        "status IN ('requested', 'completed', 'approved', 'rejected')",
    )
    op.create_check_constraint(
        "ck_order_items_price_positive",
        "order_items",
        "price > 0",
    )
    op.create_check_constraint(
        "ck_order_items_quantity_positive",
        "order_items",
        "quantity > 0",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("ck_order_items_quantity_positive", "order_items", type_="check")
    op.drop_constraint("ck_order_items_price_positive", "order_items", type_="check")
    op.drop_constraint("ck_returns_status_valid", "returns", type_="check")
    op.drop_constraint("ck_orders_status_valid", "orders", type_="check")
