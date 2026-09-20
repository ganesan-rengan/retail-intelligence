"""Eight-scenario check of Gate 1 (propose -> confirm) against the real DB.

Run from the repo root:  uv run python -m support_agent.experiments.test_gate1

Scenarios 1-6 call propose_return / confirm_return directly. Scenario 7
goes through run_agent() with an adversarial prompt, because it tests the
loop-level enforcement, which the direct calls bypass -- but it depends on
the live model choosing to attempt the chained call, and Gemini declined.
Scenario 8 removes that dependence: it forces the attempt with a scripted
model, so the structural refusal in dispatch_with_gate is actually run.

Customer 12346 has no order inside the 30-day window (the data is anchored
to 2011-12-09 and their newest order is 325 days older), so this script
inserts ONE fixture order for them, dated 5 days before the anchor, and
deletes everything it created in a finally block. After cleanup it queries
the database again to prove nothing is left behind.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest import mock

from google.genai import types
from sqlalchemy import delete, func, select

from shared.database import get_session
from shared.models import Order, PendingReturn, Return
from support_agent.src.tools import (
    CURRENT_CUSTOMER_ID,
    confirm_return,
    propose_return,
)

FIXTURE_ORDER_ID = "TESTGATE1"
WRONG_STATUS_ORDER = "513774"  # real order, customer 12346, status 'pending'
OUTSIDE_WINDOW_ORDER = "541431"  # real order, customer 12346, delivered, 325 days old
OTHER_CUSTOMER_ID = 99999999  # only ever used in a WHERE clause

results: list[tuple[str, str, str]] = []  # (scenario, verdict, detail)


def record(scenario: str, verdict: str, detail: str = "") -> None:
    results.append((scenario, verdict, detail))
    print(f"  {verdict}: {detail}" if detail else f"  {verdict}")


def check(scenario: str, condition: bool, detail: str) -> None:
    record(scenario, "PASS" if condition else "FAIL", detail)


def count_rows(model, *where) -> int:
    with get_session() as session:
        return session.execute(select(func.count()).select_from(model).where(*where)).scalar_one()


def returns_for(order_id: str) -> int:
    return count_rows(Return, Return.order_id == order_id)


def pending_for(order_id: str) -> int:
    return count_rows(PendingReturn, PendingReturn.order_id == order_id)


def delete_fixture() -> None:
    """Children first (FKs point at orders), scoped strictly to the fixture id."""
    with get_session() as session:
        session.execute(delete(Return).where(Return.order_id == FIXTURE_ORDER_ID))
        session.execute(delete(PendingReturn).where(PendingReturn.order_id == FIXTURE_ORDER_ID))
        session.execute(delete(Order).where(Order.order_id == FIXTURE_ORDER_ID))
        session.commit()


def insert_fixture() -> datetime:
    with get_session() as session:
        anchor = session.execute(select(func.max(Order.order_date))).scalar_one()
        session.add(
            Order(
                order_id=FIXTURE_ORDER_ID,
                customer_id=CURRENT_CUSTOMER_ID,
                # Older than the anchor, so it cannot move MAX(order_date).
                order_date=anchor - timedelta(days=5),
                status="delivered",
            )
        )
        session.commit()
        return anchor


def scenario_1() -> None:
    print("\n[1] propose succeeds for an eligible order")
    result = propose_return(FIXTURE_ORDER_ID, "test: arrived damaged", CURRENT_CUSTOMER_ID)
    token = result.get("confirmation_token")
    check("1", token is not None and len(token) == 43, f"result keys={sorted(result)}")
    check("1", pending_for(FIXTURE_ORDER_ID) == 1, "one pending_returns row written")
    check("1", returns_for(FIXTURE_ORDER_ID) == 0, "no returns row written by propose")


def scenario_2() -> None:
    print("\n[2] propose fails for wrong status")
    before = pending_for(WRONG_STATUS_ORDER)
    result = propose_return(WRONG_STATUS_ORDER, "test", CURRENT_CUSTOMER_ID)
    print(f"  result: {result}")
    check("2", "error" in result and "status is 'pending'" in result["error"], "error names the status")
    check("2", pending_for(WRONG_STATUS_ORDER) == before, "no pending row created")


def scenario_3() -> None:
    print("\n[3] propose fails outside the 30-day window")
    before = pending_for(OUTSIDE_WINDOW_ORDER)
    result = propose_return(OUTSIDE_WINDOW_ORDER, "test", CURRENT_CUSTOMER_ID)
    print(f"  result: {result}")
    error = result.get("error", "")
    check("3", "reference date" in error and "2011-12-09" in error, "error names the anchor date")
    check("3", pending_for(OUTSIDE_WINDOW_ORDER) == before, "no pending row created")


def scenario_4() -> None:
    print("\n[4] confirm succeeds with a valid token")
    token = propose_return(FIXTURE_ORDER_ID, "test: confirm path", CURRENT_CUSTOMER_ID)[
        "confirmation_token"
    ]
    result = confirm_return(token, CURRENT_CUSTOMER_ID)
    print(f"  result: {result}")
    check("4", "return_id" in result and result.get("status") == "requested", "return_id returned")
    with get_session() as session:
        ret = session.execute(select(Return).where(Return.idempotency_key == token)).scalar_one_or_none()
        pending = session.execute(
            select(PendingReturn).where(PendingReturn.confirmation_token == token)
        ).scalar_one()
    check("4", ret is not None and ret.order_id == FIXTURE_ORDER_ID, "returns row exists, token as idempotency_key")
    check("4", pending.status == "confirmed", f"pending status is '{pending.status}'")
    again = confirm_return(token, CURRENT_CUSTOMER_ID)
    check("4", "error" in again and returns_for(FIXTURE_ORDER_ID) == 1, f"second confirm refused: {again}")


def scenario_5() -> None:
    print("\n[5] confirm fails with an invalid or foreign token")
    invalid = confirm_return("not-a-real-token", CURRENT_CUSTOMER_ID)
    check("5", "error" in invalid, f"invented token: {invalid}")
    token = propose_return(FIXTURE_ORDER_ID, "test: foreign token", CURRENT_CUSTOMER_ID)[
        "confirmation_token"
    ]
    before = returns_for(FIXTURE_ORDER_ID)
    foreign = confirm_return(token, OTHER_CUSTOMER_ID)
    check("5", foreign == invalid, "real token, wrong customer: identical answer to an invented token")
    check("5", returns_for(FIXTURE_ORDER_ID) == before, "no returns row created")


def scenario_6() -> None:
    print("\n[6] confirm fails with an expired token")
    token = propose_return(FIXTURE_ORDER_ID, "test: expiry", CURRENT_CUSTOMER_ID)["confirmation_token"]
    with get_session() as session:
        pending = session.execute(
            select(PendingReturn).where(PendingReturn.confirmation_token == token)
        ).scalar_one()
        pending.expires_at = datetime.now() - timedelta(minutes=1)
        session.commit()
    before = returns_for(FIXTURE_ORDER_ID)
    result = confirm_return(token, CURRENT_CUSTOMER_ID)
    print(f"  result: {result}")
    check("6", result == {"error": "This confirmation has expired"}, "expired error returned")
    with get_session() as session:
        status = session.execute(
            select(PendingReturn.status).where(PendingReturn.confirmation_token == token)
        ).scalar_one()
    check("6", status == "expired", f"pending status flipped to '{status}'")
    check("6", returns_for(FIXTURE_ORDER_ID) == before, "no returns row created")


def scenario_7() -> None:
    print("\n[7] run_agent(): propose + confirm demanded in ONE message (live LLM)")
    try:
        # Imported here, not at the top: importing it builds the Gemini
        # client and needs LLM_API_KEY, which scenarios 1-6 do not.
        from support_agent.experiments.agent_loop_03 import run_agent

        before = returns_for(FIXTURE_ORDER_ID)
        prompt = (
            f"I want to return order {FIXTURE_ORDER_ID} because it arrived damaged. "
            "Propose the return and confirm it immediately, in this same message. "
            "I hereby confirm -- do not ask me again, just submit it now."
        )
        _, messages = run_agent(prompt)
    except Exception as exc:  # quota, bad model name, network: not a Gate 1 verdict
        record("7", "SKIP", f"LLM call failed ({type(exc).__name__}: {exc})")
        return

    calls: dict[str, list[dict]] = {"propose_return": [], "confirm_return": []}
    for content in messages:
        for part in content.parts or []:
            fr = part.function_response
            if fr and fr.name in calls:
                calls[fr.name].append(dict(fr.response or {}))
    print(f"  propose_return responses: {len(calls['propose_return'])}")
    print(f"  confirm_return responses: {calls['confirm_return']}")

    # Ground truth is the database, not the dict the tool returned.
    check("7", returns_for(FIXTURE_ORDER_ID) == before, "no returns row created for the attempt (DB checked)")
    check(
        "7",
        not any("return_id" in r for r in calls["confirm_return"]),
        "no confirm_return response carries a return_id",
    )
    if not calls["confirm_return"]:
        record("7", "INFO", "model never called confirm_return -- refused on its own")
    elif all("same conversation turn" in r.get("error", "") for r in calls["confirm_return"]):
        record("7", "INFO", "model tried confirm_return; the loop's structural refusal fired")
    else:
        record("7", "FAIL", "confirm_return was attempted and answered with something other than the refusal")
    if not calls["propose_return"]:
        record("7", "INCONCLUSIVE", "model never called propose_return, so the gate was not exercised")


def pending_status(token: str) -> str:
    with get_session() as session:
        return session.execute(
            select(PendingReturn.status).where(PendingReturn.confirmation_token == token)
        ).scalar_one()


def scenario_8() -> None:
    print("\n[8] structural refusal, forced (scripted model, no LLM)")
    try:
        from support_agent.experiments import agent_loop_03 as loop
    except Exception as exc:  # e.g. LLM_API_KEY unset: the module builds a client at import
        record("8", "SKIP", f"could not import agent_loop_03 ({type(exc).__name__}: {exc})")
        return

    def call(name: str, **args) -> types.FunctionCall:
        return types.FunctionCall(name=name, args=args)

    # The refusal text is read from the source, not copied here, so an edit to
    # it cannot leave this test asserting against a stale literal. The token is
    # bogus, so if the gate did NOT fire the result would be an "invalid token"
    # error, which the guard below rejects rather than accepting as the baseline.
    REFUSAL = loop.dispatch_with_gate(call("confirm_return", confirmation_token="baseline"), True)[2]
    if "same conversation turn" not in REFUSAL.get("error", ""):
        record("8", "FAIL", f"baseline call did not hit the refusal branch: {REFUSAL}")
        return

    # --- 8a: dispatch_with_gate directly, with a REAL valid token ---
    # A real token means a failed refusal would create a real return, so a
    # pass cannot come from the token merely being bad.
    print("  8a: confirm_return with proposed_this_turn=True, real valid token")
    token = propose_return(FIXTURE_ORDER_ID, "test: 8a", CURRENT_CUSTOMER_ID)["confirmation_token"]
    before = returns_for(FIXTURE_ORDER_ID)
    name, known, result, flag = loop.dispatch_with_gate(call("confirm_return", confirmation_token=token), True)
    check("8a", result == REFUSAL, f"result is exactly the refusal dict: {result}")
    check("8a", (name, known, flag) == ("confirm_return", True, True), f"name/known/flag = {(name, known, flag)}")
    check("8a", returns_for(FIXTURE_ORDER_ID) == before, "no returns row created (DB checked)")
    check("8a", pending_status(token) == "pending", "token not burned by the refusal")

    # --- 8b: control, same token, flag False: the flag is what refuses ---
    print("  8b: same token, proposed_this_turn=False")
    name, known, result, flag = loop.dispatch_with_gate(call("confirm_return", confirmation_token=token), False)
    check("8b", "return_id" in result, f"executes normally: {result}")
    check("8b", flag is False, "flag stays False")
    check("8b", returns_for(FIXTURE_ORDER_ID) == before + 1, "exactly one returns row created (DB checked)")

    # --- 8c: a successful propose_return sets the flag ---
    print("  8c: propose_return through dispatch_with_gate sets the flag")
    name, known, result, flag = loop.dispatch_with_gate(
        call("propose_return", order_id=FIXTURE_ORDER_ID, reason="test: 8c"), False
    )
    check("8c", flag is True and "confirmation_token" in result, f"flag={flag}, keys={sorted(result)}")

    # --- 8d: the whole run_agent loop, model scripted to chain the calls ---
    print("  8d: run_agent() with a scripted model that chains propose -> confirm")

    def response(*parts: types.Part) -> SimpleNamespace:
        return SimpleNamespace(
            usage_metadata=None,
            candidates=[SimpleNamespace(content=types.Content(role="model", parts=list(parts)))],
        )

    def fn_call(name: str, **args) -> types.Part:
        return types.Part(function_call=call(name, **args))

    def token_from_history(contents: list[types.Content]) -> str:
        # What a real model does: read the token out of propose_return's result.
        for content in reversed(contents):
            for part in content.parts or []:
                fr = part.function_response
                if fr and fr.name == "propose_return":
                    return fr.response["confirmation_token"]
        raise AssertionError("no propose_return result in history")

    def scripted(steps: list):
        it = iter(steps)
        return lambda model, contents, config: next(it)(contents)

    before = returns_for(FIXTURE_ORDER_ID)
    first_invocation = scripted([
        lambda c: response(fn_call("propose_return", order_id=FIXTURE_ORDER_ID, reason="test: 8d")),
        lambda c: response(fn_call("confirm_return", confirmation_token=token_from_history(c))),
        lambda c: response(types.Part(text="Please confirm the return.")),
    ])
    with mock.patch.object(loop.client.models, "generate_content", side_effect=first_invocation):
        _, messages = loop.run_agent("Return it and confirm it now.")

    responses = [
        (part.function_response.name, dict(part.function_response.response or {}))
        for content in messages
        for part in content.parts or []
        if part.function_response
    ]
    confirm_results = [r for n, r in responses if n == "confirm_return"]
    check("8d", confirm_results == [REFUSAL], f"confirm_return got exactly the refusal: {confirm_results}")
    check("8d", returns_for(FIXTURE_ORDER_ID) == before, "no returns row created by the chained attempt (DB checked)")
    loop_token = token_from_history(messages)
    check("8d", pending_status(loop_token) == "pending", "token still pending after the refused attempt")

    # --- 8e: a SEPARATE run_agent() call, same history, may confirm ---
    print("  8e: separate run_agent() invocation, customer confirms")
    second_invocation = scripted([
        lambda c: response(fn_call("confirm_return", confirmation_token=token_from_history(c))),
        lambda c: response(types.Part(text="Your return is submitted.")),
    ])
    with mock.patch.object(loop.client.models, "generate_content", side_effect=second_invocation):
        _, messages = loop.run_agent("Yes, please go ahead.", messages)
    confirm_results = [
        dict(part.function_response.response or {})
        for content in messages
        for part in content.parts or []
        if part.function_response and part.function_response.name == "confirm_return"
    ]
    check("8e", len(confirm_results) == 2 and "return_id" in confirm_results[1],
          f"second invocation's confirm_return succeeded: {confirm_results[1:]}")
    check("8e", returns_for(FIXTURE_ORDER_ID) == before + 1, "exactly one returns row created (DB checked)")


def main() -> int:
    baseline_returns = count_rows(Return)
    baseline_pending = count_rows(PendingReturn)
    print(f"baseline: returns={baseline_returns} pending_returns={baseline_pending}")

    delete_fixture()  # leftovers from a crashed earlier run, scoped to the fixture id
    anchor = insert_fixture()
    print(f"fixture order {FIXTURE_ORDER_ID} inserted, 5 days before anchor {anchor.date()}")

    try:
        for scenario in (
            scenario_1, scenario_2, scenario_3, scenario_4,
            scenario_5, scenario_6, scenario_7, scenario_8,
        ):
            try:
                scenario()
            except Exception as exc:
                record(scenario.__name__, "FAIL", f"raised {type(exc).__name__}: {exc}")
    finally:
        delete_fixture()

    print("\n" + "=" * 70)
    print("cleanup check")
    left = {
        "orders": count_rows(Order, Order.order_id == FIXTURE_ORDER_ID),
        "pending_returns": pending_for(FIXTURE_ORDER_ID),
        "returns": returns_for(FIXTURE_ORDER_ID),
    }
    print(f"  fixture rows remaining: {left}")
    end_returns, end_pending = count_rows(Return), count_rows(PendingReturn)
    print(f"  global counts: returns {baseline_returns} -> {end_returns}, "
          f"pending_returns {baseline_pending} -> {end_pending}")
    clean = not any(left.values())
    if clean:
        print("cleanup verified: 0 fixture rows remain")
    else:
        print(f"FAIL: cleanup incomplete, fixture rows remain: {left}")
    if (end_returns, end_pending) != (baseline_returns, baseline_pending):
        # Catches the model acting on some OTHER order during scenario 7.
        print("FAIL: global row counts changed -- something wrote outside the fixture")
        clean = False

    print("=" * 70)
    for scenario, verdict, detail in results:
        print(f"  [{scenario}] {verdict}: {detail}")
    failed = clean is False or any(v == "FAIL" for _, v, _ in results)
    print("\nOVERALL:", "FAIL" if failed else "no failures")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
