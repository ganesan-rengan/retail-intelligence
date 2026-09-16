"""Real tools for the support agent, backed by the actual database.

Replaces the fake, canned-dict get_order_status in
experiments/03_agent_loop.py with a real query against shared.models.
"""

from google.genai import types
from sqlalchemy import func, select

from shared.database import get_session
from shared.models import Order, OrderItem

# TEMPORARY: stands in for real auth (step 2.21). Once auth exists, the
# caller's customer_id must come from an authenticated session, never from
# a module constant.
CURRENT_CUSTOMER_ID = 12346


def get_order_status(order_id: str, customer_id: int) -> dict:
    """Look up an order, scoped to the given customer.

    Returns {"error": "Order not found"} if no order matches BOTH
    order_id and customer_id -- an order that exists but belongs to a
    different customer is indistinguishable from one that doesn't exist
    at all.
    """
    with get_session() as session:
        # Single query handles both "order doesn't exist" and "order
        # exists but belongs to someone else" identically -- do not split
        # this into two queries with different error messages, that
        # would leak which order IDs are real.
        order = session.execute(
            select(Order).where(
                Order.order_id == order_id,
                Order.customer_id == customer_id,
            )
        ).scalar_one_or_none()

        if order is None:
            return {"error": "Order not found"}

        # Only reachable once the ownership check above has passed, and
        # keyed on order.order_id (not the raw order_id parameter) so
        # there is no path where an item count is computed for an order
        # that failed the ownership check.
        item_count = session.execute(
            select(func.count())
            .select_from(OrderItem)
            .where(OrderItem.order_id == order.order_id)
        ).scalar_one()

        return {
            "order_id": order.order_id,
            "status": order.status,
            "order_date": str(order.order_date),
            "item_count": item_count,
        }


def get_order_status_for_model(order_id: str) -> dict:
    """Wrapper actually registered as a tool. Binds CURRENT_CUSTOMER_ID so
    there is no code path where the model's output could reach
    customer_id -- the model never sees that parameter and never supplies
    it."""
    return get_order_status(order_id, CURRENT_CUSTOMER_ID)


TOOLS = {
    "get_order_status": get_order_status_for_model,
}


order_status_declaration = types.FunctionDeclaration(
    name="get_order_status",
    description=(
        "Look up the current status and order date of a specific order. "
        "Requires the exact order ID. Use this when a customer asks where "
        "their order is, whether it has shipped, or when it was placed."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "order_id": types.Schema(
                type=types.Type.STRING,
                description="The order ID, e.g. '1041'",
            ),
        },
        required=["order_id"],
    ),
)
