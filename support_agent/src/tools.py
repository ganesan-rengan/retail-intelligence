"""Real tools for the support agent, backed by the actual database.

Replaces the fake, canned-dict implementations in
experiments/agent_loop_03.py: get_order_status (real query, scoped to
customer_id) and search_policy (real cosine-similarity search over
embedded policy chunks).
"""

import json
import secrets
import sys
from datetime import datetime, timedelta

import httpx
from google.genai import types
from sentence_transformers import SentenceTransformer
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from shared.database import get_session
from shared.models import AgentAction, Order, OrderItem, PendingReturn, PolicyChunk, Return

# TEMPORARY: stands in for real auth (step 2.21). Once auth exists, the
# caller's customer_id must come from an authenticated session, never from
# a module constant.
CURRENT_CUSTOMER_ID = 12346


def _mask_tokens(payload: dict) -> dict:
    """Copy of payload with any "confirmation_token" value cut to 8 chars.
    Keyed on the field name, not the tool, so a live token can't reach the
    audit table in plaintext through whichever tool happens to carry it."""
    masked = dict(payload)
    if "confirmation_token" in masked:
        masked["confirmation_token"] = str(masked["confirmation_token"])[:8] + "..."
    return masked


def log_action(customer_id: int, tool_name: str, arguments: dict, outcome: dict) -> None:
    """Append one row to agent_actions. Best-effort audit trail: it uses its
    OWN session (a failure here cannot roll back or poison the tool's
    transaction) and never raises (a logging failure cannot change what the
    customer sees). Failures go to stderr. Only the logged copies are masked;
    the caller's dicts are never mutated."""
    try:
        with get_session() as session:
            session.add(
                AgentAction(
                    customer_id=customer_id,
                    tool_name=tool_name[:60],  # a hallucinated name can exceed String(60)
                    arguments=json.dumps(_mask_tokens(arguments), default=str),
                    outcome=json.dumps(_mask_tokens(outcome), default=str),
                )
            )
            session.commit()
    except Exception as exc:
        print(f"log_action failed ({type(exc).__name__}: {exc})", file=sys.stderr)

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

# From returns_policy.md: a return must be initiated within 30 days of the
# order date.
RETURN_WINDOW_DAYS = 30

# How long a proposed return stays confirmable. This is a real wall-clock
# duration (unlike the return-window anchor in propose_return, which is
# data-anchored).
CONFIRMATION_TTL_MINUTES = 15


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


def propose_return(order_id: str, reason: str, customer_id: int) -> dict:
    """Gate 1, step 1: check eligibility and record a PROPOSED return.

    Writes only to pending_returns, never to returns -- nothing here
    creates a real return. The caller gets a single-use confirmation
    token; confirm_return(token) is the only path to a real return.

    HISTORICAL-DATA-ONLY CAVEAT: the 30-day window is measured against
    MAX(orders.order_date), NOT the current date. The dataset is a frozen
    2010-2011 snapshot, so real "today" would reject every order. A real
    system MUST replace that anchor with actual wall-clock time
    (datetime.now()).
    """
    with get_session() as session:
        # Same single-query ownership check as get_order_status: a missing
        # order and someone else's order are indistinguishable.
        order = session.execute(
            select(Order).where(
                Order.order_id == order_id,
                Order.customer_id == customer_id,
            )
        ).scalar_one_or_none()

        if order is None:
            return {"error": "Order not found"}

        if order.status != "delivered":
            return {
                "error": (
                    f"Order is not eligible for return: status is "
                    f"'{order.status}', only 'delivered' orders can be returned"
                )
            }

        # HISTORICAL-DATA-ONLY: anchor "now" to the newest order in the
        # data, computed live on every call (never hardcoded). This looks
        # like a bug next to real wall-clock logic but is deliberate --
        # the data ends in 2011, so datetime.now() would make every order
        # ineligible. A real system must use actual wall-clock time here.
        anchor = session.execute(select(func.max(Order.order_date))).scalar_one()

        age = anchor - order.order_date
        if age > timedelta(days=RETURN_WINDOW_DAYS):
            return {
                "error": (
                    f"Order is not eligible for return: placed {age.days} "
                    f"days before the reference date {anchor.date()} "
                    f"(latest order date in the data), outside the "
                    f"{RETURN_WINDOW_DAYS}-day return window"
                )
            }

        pending = PendingReturn(
            order_id=order.order_id,
            customer_id=customer_id,
            reason=reason,
            confirmation_token=secrets.token_urlsafe(32),
            # Set explicitly: the column's default is ORM-side only.
            status="pending",
            expires_at=datetime.now() + timedelta(minutes=CONFIRMATION_TTL_MINUTES),
        )
        session.add(pending)
        session.commit()

        return {
            "confirmation_token": pending.confirmation_token,
            "order_id": pending.order_id,
            "reason": pending.reason,
            "expires_at": str(pending.expires_at),
            "message": (
                "Return proposed but NOT yet submitted. Ask the customer to "
                "confirm; only then call confirm_return with this token."
            ),
        }


def propose_return_for_model(order_id: str, reason: str) -> dict:
    """Registered wrapper: binds CURRENT_CUSTOMER_ID, same as
    get_order_status_for_model."""
    return propose_return(order_id, reason, CURRENT_CUSTOMER_ID)


def confirm_return(confirmation_token: str, customer_id: int) -> dict:
    """Gate 1, step 2: turn a pending proposal into a real return.

    Atomic: the returns insert and the pending -> confirmed flip share one
    commit (get_session() has autoflush=False, so nothing is written until
    commit(), same precedent as chunk_policies.py). The returns row uses
    the token as its idempotency_key, so two concurrent confirmations of
    the same token both pass the status check but the second hits
    uq_returns_idempotency_key -- Gate 1 (confirmation) and Gate 3
    (idempotency) reinforce each other structurally.
    """
    with get_session() as session:
        # Token AND customer_id AND status, in one query: an unknown,
        # foreign, or already-used token all get the same answer.
        pending = session.execute(
            select(PendingReturn).where(
                PendingReturn.confirmation_token == confirmation_token,
                PendingReturn.customer_id == customer_id,
                PendingReturn.status == "pending",
            )
        ).scalar_one_or_none()

        if pending is None:
            return {"error": "Confirmation not found or no longer valid"}

        if pending.expires_at < datetime.now():
            pending.status = "expired"
            session.commit()
            return {"error": "This confirmation has expired"}

        new_return = Return(
            order_id=pending.order_id,
            reason=pending.reason,
            status="requested",
            idempotency_key=pending.confirmation_token,
        )
        session.add(new_return)
        pending.status = "confirmed"

        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            # Only the idempotency constraint means "already used". Any
            # other integrity failure (e.g. a primary-key collision) is a
            # real bug and must surface, not be mislabeled as a lost race.
            if getattr(exc.orig.diag, "constraint_name", None) != "uq_returns_idempotency_key":
                raise
            return {"error": "This confirmation has already been used"}

        return {
            "return_id": new_return.return_id,
            "order_id": new_return.order_id,
            "status": new_return.status,
        }


def confirm_return_for_model(confirmation_token: str) -> dict:
    """Registered wrapper: binds CURRENT_CUSTOMER_ID."""
    return confirm_return(confirmation_token, CURRENT_CUSTOMER_ID)


TOOLS = {
    "get_order_status": get_order_status_for_model,
    "search_policy": search_policy,
    "get_demand_forecast": get_demand_forecast,
    "propose_return": propose_return_for_model,
    "confirm_return": confirm_return_for_model,
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

propose_return_declaration = types.FunctionDeclaration(
    name="propose_return",
    description=(
        "Start a return for a delivered order. This only PROPOSES the "
        "return and checks eligibility -- it does not submit anything. It "
        "returns a confirmation token; tell the customer what will be "
        "returned and ask them to confirm before calling confirm_return. "
        "Use only when the customer explicitly asks to return an order."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "order_id": types.Schema(
                type=types.Type.STRING,
                description="The order ID to return, e.g. '541431'",
            ),
            "reason": types.Schema(
                type=types.Type.STRING,
                description="The customer's reason for the return, in their own words.",
            ),
        },
        required=["order_id", "reason"],
    ),
)

confirm_return_declaration = types.FunctionDeclaration(
    name="confirm_return",
    description=(
        "Submit a proposed return. Call this ONLY after propose_return has "
        "succeeded AND the customer has explicitly confirmed in their own "
        "message. Never call it in the same turn as propose_return, and "
        "never invent a token -- use exactly the one propose_return gave."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "confirmation_token": types.Schema(
                type=types.Type.STRING,
                description="The confirmation_token returned by propose_return.",
            ),
        },
        required=["confirmation_token"],
    ),
)
