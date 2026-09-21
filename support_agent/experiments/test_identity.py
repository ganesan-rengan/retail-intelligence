"""Six-scenario check of login() / resolve_session() against the real DB.

Run from the repo root:  uv run python -m support_agent.experiments.test_identity

Scenarios 1-2 test login() (success, unknown customer). Scenarios 3-5 test
resolve_session() (valid token, unknown token, expired token). Scenario 6
pins the expiry BOUNDARY: expires_at < now means expired, so a session whose
expires_at equals "now" is still valid, matching confirm_return (scenario 6
of test_gate1.py). Real time always moves forward between an insert and the
read, so equality can't be reached on the real clock; scenario 6 swaps in a
frozen clock for identity.datetime to sit exactly on the boundary.

This file deliberately does NOT import tools.py, which loads the embedding
model at import time. Customer 12346 (tools.CURRENT_CUSTOMER_ID) is
hardcoded instead, and main() checks that the customer really exists before
running anything. Importing the constant would be the only reason to load
tools.py here; a stale hardcode fails loudly at that check instead.

Every CustomerSession row created (through login() or by direct insert) is
tracked by token and deleted in a finally block. After cleanup the table is
queried again and compared with its pre-test count and max session_id.
"""

from datetime import datetime, timedelta
from unittest import mock

from sqlalchemy import delete, func, select

from shared.database import get_session
from shared.models import Customer, CustomerSession
from support_agent.src import identity as identity_module
from support_agent.src.identity import SESSION_TTL_MINUTES, login, resolve_session

TEST_CUSTOMER_ID = 12346  # same value as tools.CURRENT_CUSTOMER_ID; see docstring
MISSING_CUSTOMER_ID = -1

results: list[tuple[str, str, str]] = []  # (scenario, verdict, detail)
created_tokens: list[str] = []  # every session_token this run created
state: dict = {}  # carries the scenario 1 login result into scenario 3


def record(scenario: str, verdict: str, detail: str = "") -> None:
    results.append((scenario, verdict, detail))
    print(f"  {verdict}: {detail}" if detail else f"  {verdict}")


def check(scenario: str, condition: bool, detail: str) -> None:
    record(scenario, "PASS" if condition else "FAIL", detail)


def count_rows(model, *where) -> int:
    with get_session() as session:
        return session.execute(select(func.count()).select_from(model).where(*where)).scalar_one()


def max_session_id() -> int:
    with get_session() as session:
        return session.execute(select(func.max(CustomerSession.session_id))).scalar() or 0


def session_row(token: str) -> CustomerSession | None:
    with get_session() as session:
        return session.execute(
            select(CustomerSession).where(CustomerSession.session_token == token)
        ).scalar_one_or_none()


def insert_session(token: str, expires_at: datetime) -> None:
    """Direct insert, bypassing login(), so expires_at can be set to any value."""
    created_tokens.append(token)
    with get_session() as session:
        session.add(
            CustomerSession(
                customer_id=TEST_CUSTOMER_ID, session_token=token, expires_at=expires_at
            )
        )
        session.commit()


def delete_created() -> None:
    """Scoped strictly to the tokens this run created."""
    with get_session() as session:
        session.execute(
            delete(CustomerSession).where(CustomerSession.session_token.in_(created_tokens))
        )
        session.commit()


def scenario_1() -> None:
    print("\n[1] login() succeeds for a real customer")
    before = datetime.now()
    result = login(TEST_CUSTOMER_ID)
    token = result.get("session_token")
    if token is not None:
        created_tokens.append(token)  # track before asserting anything
    state["login"] = result
    check("1", set(result) == {"session_token", "customer_id", "expires_at"}, f"result keys={sorted(result)}")
    check("1", token is not None and len(token) == 43, f"token length={len(token) if token else None} (expect 43)")
    check("1", result.get("customer_id") == TEST_CUSTOMER_ID, f"customer_id={result.get('customer_id')}")
    row = session_row(token) if token else None
    check("1", row is not None, "a real customer_sessions row exists with that token (DB checked)")
    if row is not None:
        check("1", row.customer_id == TEST_CUSTOMER_ID, f"row.customer_id={row.customer_id}")
        expected = before + timedelta(minutes=SESSION_TTL_MINUTES)
        drift = abs((row.expires_at - expected).total_seconds())
        check("1", drift < 5, f"row.expires_at is ~{SESSION_TTL_MINUTES} min ahead (off by {drift:.3f}s)")
        check("1", result.get("expires_at") == str(row.expires_at),
              f"returned expires_at matches the stored one: {result.get('expires_at')}")


def scenario_2() -> None:
    print("\n[2] login() fails cleanly for a customer that does not exist")
    total_before = count_rows(CustomerSession)
    result = login(MISSING_CUSTOMER_ID)
    check("2", result == {"error": "Customer not found"}, f"result={result}")
    check("2", count_rows(CustomerSession, CustomerSession.customer_id == MISSING_CUSTOMER_ID) == 0,
          "no row for the missing customer (DB checked)")
    check("2", count_rows(CustomerSession) == total_before, "table count unchanged (DB checked)")


def scenario_3() -> None:
    print("\n[3] resolve_session() resolves the token from scenario 1")
    token = state["login"]["session_token"]
    resolved = resolve_session(token)
    check("3", resolved == TEST_CUSTOMER_ID, f"resolved={resolved}, expected {TEST_CUSTOMER_ID}")


def scenario_4() -> None:
    print("\n[4] resolve_session() returns None for an unknown token")
    for bogus in ("not-a-real-token", ""):
        check("4", resolve_session(bogus) is None, f"token={bogus!r} -> None")


def scenario_5() -> None:
    print("\n[5] resolve_session() returns None for an expired token")
    token = "TEST-IDENTITY-EXPIRED"
    insert_session(token, datetime.now() - timedelta(minutes=1))
    check("5", session_row(token) is not None, "expired row really exists (DB checked), so None below is expiry, not absence")
    check("5", resolve_session(token) is None, "expired token -> None")


class FrozenDatetime(datetime):
    """datetime whose now() returns a fixed instant, so the boundary can be hit exactly."""

    frozen: datetime

    @classmethod
    def now(cls, tz=None):
        return cls.frozen


def scenario_6() -> None:
    print("\n[6] expiry boundary: expires_at < now is expired, so equality is still valid")
    # Real clock first: an expires_at of "now" is already in the past by the time
    # resolve_session reads it, so it must come back expired. This alone cannot tell
    # < from <=, because equality is unreachable in real time; the frozen part can.
    near_now = "TEST-IDENTITY-NEARNOW"
    insert_session(near_now, datetime.now())
    check("6", resolve_session(near_now) is None, "real clock, expires_at set to now() -> expired (None)")

    # Frozen clock: place expires_at exactly 1 microsecond before, at, and after "now".
    frozen_now = datetime.now().replace(microsecond=0)
    FrozenDatetime.frozen = frozen_now
    cases = [
        ("TEST-IDENTITY-BOUND-BEFORE", frozen_now - timedelta(microseconds=1), None, "1us before now -> expired (None)"),
        ("TEST-IDENTITY-BOUND-EQUAL", frozen_now, TEST_CUSTOMER_ID, "exactly now -> still valid (matches confirm_return's '<')"),
        ("TEST-IDENTITY-BOUND-AFTER", frozen_now + timedelta(microseconds=1), TEST_CUSTOMER_ID, "1us after now -> valid"),
    ]
    for token, expires_at, _, _ in cases:
        insert_session(token, expires_at)
    with mock.patch.object(identity_module, "datetime", FrozenDatetime):
        for token, _, expected, label in cases:
            resolved = resolve_session(token)
            check("6", resolved == expected, f"{label}: resolved={resolved}")


def main() -> int:
    baseline_count = count_rows(CustomerSession)
    baseline_max_id = max_session_id()
    print(f"baseline: customer_sessions={baseline_count} max session_id={baseline_max_id}")

    if count_rows(Customer, Customer.customer_id == TEST_CUSTOMER_ID) != 1:
        print(f"FAIL: customer {TEST_CUSTOMER_ID} does not exist; nothing was run")
        return 1
    print(f"customer {TEST_CUSTOMER_ID} exists")

    try:
        for scenario in (scenario_1, scenario_2, scenario_3, scenario_4, scenario_5, scenario_6):
            try:
                scenario()
            except Exception as exc:
                record(scenario.__name__, "FAIL", f"raised {type(exc).__name__}: {exc}")
    finally:
        delete_created()

    print("\n" + "=" * 70)
    print("cleanup check")
    left = count_rows(CustomerSession, CustomerSession.session_token.in_(created_tokens))
    end_count = count_rows(CustomerSession)
    above = count_rows(CustomerSession, CustomerSession.session_id > baseline_max_id)
    print(f"  tracked tokens: {len(created_tokens)}, still in table: {left}")
    print(f"  customer_sessions: {baseline_count} -> {end_count}, rows above baseline id {baseline_max_id}: {above}")
    clean = left == 0 and end_count == baseline_count and above == 0
    if clean:
        print("cleanup verified: table is back to its pre-test count, 0 test rows remain")
    else:
        print("FAIL: cleanup incomplete or something wrote outside the tracked tokens")

    print("=" * 70)
    for scenario, verdict, detail in results:
        print(f"  [{scenario}] {verdict}: {detail}")
    failed = not clean or any(v == "FAIL" for _, v, _ in results)
    print("\nOVERALL:", "FAIL" if failed else "no failures")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
