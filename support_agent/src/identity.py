"""Session-based IDENTIFICATION for the support agent (not authentication).

Kept out of tools.py on purpose: tools.py loads the embedding model at import
time and holds the TOOLS registry, and nothing here should ever be registered
there. These functions run before a conversation, not during one.
"""

import secrets
from datetime import datetime, timedelta

from sqlalchemy import select

from shared.database import get_session
from shared.models import Customer, CustomerSession

SESSION_TTL_MINUTES = 60


def login(customer_id: int) -> dict:
    """Create a session for a customer and return its token.

    NEVER EXPOSE THIS AS A TOOL: no FunctionDeclaration, never added to TOOLS.
    It stands in for a real login step that happens before a conversation
    starts, the way a person enters credentials on a login page before opening
    a chat window. If the model could call it, it could mint a session as any
    customer_id it chose.

    IDENTIFICATION, NOT AUTHENTICATION: the only check is that the customer
    row exists. Anyone who knows or guesses a valid customer_id gets a full
    session as that customer. A real system must verify a credential here.
    """
    with get_session() as session:
        customer = session.execute(
            select(Customer.customer_id).where(Customer.customer_id == customer_id)
        ).one_or_none()
        if customer is None:
            return {"error": "Customer not found"}

        token = secrets.token_urlsafe(32)
        # Real wall-clock time, set explicitly (ADR-013): never a column default.
        expires_at = datetime.now() + timedelta(minutes=SESSION_TTL_MINUTES)
        session.add(
            CustomerSession(
                customer_id=customer_id,
                session_token=token,
                expires_at=expires_at,
            )
        )
        session.commit()

    return {
        "session_token": token,
        "customer_id": customer_id,
        "expires_at": str(expires_at),
    }


def resolve_session(session_token: str) -> int | None:
    """Return the customer_id for a valid session token, else None.

    None covers both "no such token" and "expired": the caller only needs
    valid-or-not, not a customer-facing reason.

    The lookup is a single SELECT on the token (served by the UNIQUE
    constraint's index). Expiry is checked here in Python, never with
    func.now() in the query, so one code path owns the whole expiry decision
    (ADR-013). The boundary matches confirm_return: expires_at < now means
    expired.
    """
    with get_session() as session:
        row = session.execute(
            select(CustomerSession.customer_id, CustomerSession.expires_at).where(
                CustomerSession.session_token == session_token
            )
        ).one_or_none()

    if row is None:
        return None
    if row.expires_at < datetime.now():
        return None
    return row.customer_id
