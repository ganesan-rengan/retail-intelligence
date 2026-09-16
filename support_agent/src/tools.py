"""Real tools for the support agent, backed by the actual database.

Replaces the fake, canned-dict implementations in
experiments/agent_loop_03.py: get_order_status (real query, scoped to
customer_id) and search_policy (real cosine-similarity search over
embedded policy chunks).
"""

import httpx
from google.genai import types
from sentence_transformers import SentenceTransformer
from sqlalchemy import func, select

from shared.database import get_session
from shared.models import Order, OrderItem, PolicyChunk

# TEMPORARY: stands in for real auth (step 2.21). Once auth exists, the
# caller's customer_id must come from an authenticated session, never from
# a module constant.
CURRENT_CUSTOMER_ID = 12346

FORECASTING_SERVICE_URL = "https://retail-forecasting-service.onrender.com"

# Loaded once at import time, not per-call -- same reasoning as
# chunk_policies.py: model loading is the expensive part, and this must be
# the SAME model used there, or the embeddings live in incompatible spaces.
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Initial guess, not a validated cutoff. Checked live against the real
# 18-chunk table on 2026-09-16: a real returns question scored
# 0.3401/0.3910/0.4351 for its top 3 matches, while an off-topic question
# ("What is the capital of France?") scored 0.9502 for its closest match.
# That gap is wide enough to place a threshold in, but it's one spot-check,
# not a tuned value -- re-derive against a broader set of real customer
# questions before trusting this number, and keep the old numbers here as
# a before/after baseline when it's re-tuned.
SIMILARITY_THRESHOLD = 0.5


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


def search_policy(question: str) -> dict:
    """Search the policy documents for content relevant to a question.

    Embeds the question with the same model used to embed the policy
    chunks, orders policy_chunks by cosine distance (pgvector's `<=>`,
    NOT `<->` which is Euclidean and wrong for text similarity), and
    takes the 3 closest. Any of those 3 whose distance still exceeds
    SIMILARITY_THRESHOLD are then dropped -- a bad top-3 is still
    reported honestly as fewer (or zero) results, never padded out with
    a 4th, weaker match to make up the count.
    """
    question_embedding = EMBEDDING_MODEL.encode(question)

    with get_session() as session:
        distance = PolicyChunk.embedding.cosine_distance(question_embedding)
        rows = session.execute(
            select(
                PolicyChunk.document,
                PolicyChunk.section,
                PolicyChunk.content,
                distance.label("distance"),
            )
            .order_by(distance)
            .limit(3)
        ).all()

    results = [row for row in rows if row.distance <= SIMILARITY_THRESHOLD]

    if not results:
        return {"error": "No matching policy found"}

    return {
        "results": [
            {
                "document": row.document,
                "section": row.section,
                "content": row.content,
                "distance": row.distance,
            }
            for row in results
        ]
    }


def get_demand_forecast(product_id: str) -> dict:
    """Look up the 4-week demand forecast for a product.

    horizon_days is never model-settable -- always 28, the only horizon
    the deployed model supports -- same reasoning as customer_id never
    being model-settable in get_order_status_for_model.
    """
    try:
        response = httpx.post(
            f"{FORECASTING_SERVICE_URL}/forecast",
            json={"product_id": product_id, "horizon_days": 28},
            timeout=45.0,
        )
    except httpx.RequestError as exc:
        # Covers both httpx.TimeoutException (Render's free-tier cold
        # start can take 30-60s) and httpx.ConnectError -- never a bare
        # except Exception, which would also swallow a real bug (a
        # malformed URL, a KeyError from bad parsing) and misreport it as
        # "service unavailable."
        return {"error": f"Forecasting service unavailable: {exc}"}

    if response.status_code == 404:
        return {"error": "No demand forecast available for this product"}

    data = response.json()
    return {
        "product_id": data["product_id"],
        "predicted_units": data["predicted_units"],
        # Pydantic serializes `date` fields to ISO 8601 strings on the
        # wire (e.g. "2011-12-26") -- passed through as-is, no
        # date.fromisoformat() round trip, since this is a dict for the
        # model to read, not typed Python objects for further computation.
        "week_start": data["week_start"],
        "week_end": data["week_end"],
    }


TOOLS = {
    "get_order_status": get_order_status_for_model,
    "search_policy": search_policy,
    "get_demand_forecast": get_demand_forecast,
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

search_policy_declaration = types.FunctionDeclaration(
    name="search_policy",
    description=(
        "Search the store's policy documents (returns, refund, shipping) for "
        "general policy questions -- e.g. return eligibility, refund timing, "
        "shipping windows, cancellation rules. Use this for questions about "
        "rules, conditions, or timeframes that apply generally, not to a "
        "specific order. Do NOT use this to look up a specific order's "
        "status, date, or item count -- use get_order_status for that "
        "instead."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "question": types.Schema(
                type=types.Type.STRING,
                description="The customer's policy question, in their own words.",
            ),
        },
        required=["question"],
    ),
)

demand_forecast_declaration = types.FunctionDeclaration(
    name="get_demand_forecast",
    description=(
        "Look up the 4-week demand forecast for a product -- predicted "
        "units and the forecast week's date range. Use this for "
        "demand/stock-outlook questions, e.g. whether a product is "
        "expected to be in high demand or is likely to be low in stock. "
        "Do NOT use this for a specific order's status (use "
        "get_order_status) or for policy questions like returns or "
        "shipping rules (use search_policy)."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "product_id": types.Schema(
                type=types.Type.STRING,
                description="The product ID, e.g. '15036'",
            ),
        },
        required=["product_id"],
    ),
)
