# ADR 012: Gate 2 scope boundary confirmed across all five tools

## Context
Reviewed support_agent/src/tools.py directly (grep against the live
file, 2026-09-19) to confirm which tools can be influenced by or leak
information based on a model-supplied customer_id.

Five tools are registered in TOOLS: get_order_status_for_model,
search_policy, get_demand_forecast, propose_return_for_model,
confirm_return_for_model.

## Findings, verified against the real file
- get_order_status_for_model, propose_return_for_model,
  confirm_return_for_model: bind CURRENT_CUSTOMER_ID server-side.
  customer_id appears in each underlying function's real signature and
  in real query WHERE clauses (Order.customer_id ==,
  PendingReturn.customer_id ==) -- never in any FunctionDeclaration's
  schema. confirm_return's query filters on confirmation_token AND
  customer_id in the same select() (lines 281-282), not the token
  alone.
- search_policy, get_demand_forecast: no customer_id parameter, bound
  or otherwise, anywhere in either function. search_policy queries
  PolicyChunk (no customer_id column). get_demand_forecast calls an
  external HTTP API keyed only on product_id. Neither touches any
  customer-keyed table (Order, Return, PendingReturn all confirmed
  absent from both functions' code paths).

## Consequence
Gate 2 is correctly and consistently enforced on every tool that
touches customer-private data, and correctly absent from every tool
that touches genuinely public data. No gap found. Verified by grep
against the live file, not carried forward from memory of earlier
design discussion.

## Update 2026-09-21: CURRENT_CUSTOMER_ID replaced by a real session mechanism

The constant this audit was written against no longer exists (commit
73605f3). It was replaced, not removed: identity.login(customer_id) issues
a token (customer_sessions table, 60-minute real-wall-clock TTL),
identity.resolve_session(token) validates it, and run_agent() resolves the
token once per call and sets a contextvars.ContextVar
(identity.current_customer_id) through the as_customer() context manager,
which resets it when the call ends. The three customer-scoped wrappers read
customer_id from that ContextVar and return {"error": "No active session"}
if it is None. login() is never registered as a tool.

The original conclusion is unchanged and was re-checked against the current
file, not carried forward: none of the five FunctionDeclarations mentions
customer_id or session_token (grepped per declaration block, 2026-09-21).
session_token is a run_agent() parameter, not a tool parameter. The model
has no more ability to choose customer_id than before. The binding source
moved from a hardcoded constant to a resolved session; the property Gate 2
depends on, that customer_id is never model-supplied, did not change.

One thing is new, and it is separate from Gate 2: this is session-based
IDENTIFICATION, not authentication. Customer has no credential field of any
kind (customer_id, name, email, country), so login() only verifies that a
customer_id exists. Anyone who knows or guesses a valid customer_id gets a
full session as that customer. This does not weaken Gate 2, which only ever
guaranteed that the model cannot choose who it acts as, and that still
holds. It is an adjacent gap, recorded here so "customer scoping is
correctly enforced" is not read as "only the right person can obtain a
session." Also: the session is resolved once, at the start of each
run_agent() call; one that expires mid-call is not re-checked until the
next call.

Verification:
- (2026-09-21, before scenario 7 existed) test_identity.py, 6 scenarios
  (19 checks) against the real database, all passing: login success and
  failure, resolve_session for a valid, unknown and expired token, and
  the expiry boundary (expires_at < now is expired, so equality is
  still valid, matching confirm_return), pinned with a frozen clock.
- Verified 2026-09-21: all 11 scenarios in test_gate1.py and all 7
  scenarios in test_identity.py pass. Scenario 11 specifically proves the
  None-session backstop fires: calling get_order_status_for_model()
  directly, with no session context set (confirmed via an explicit
  current_customer_id.get() is None precondition), against a real existing
  order, returns exactly {'error': 'No active session'} rather than
  leaking through to the raw query -- the first time this branch has
  executed anywhere in the suite. All four tracked tables (returns,
  pending_returns, agent_actions, customer_sessions) confirmed unchanged
  by the call, and full suite cleanup verified back to baseline in both
  files.
- Verified in test_identity.py scenario 7: a plain threading.Thread and a
  ThreadPoolExecutor.submit() worker both read the ContextVar's default
  (None), not the main thread's set value; a copy_context().run() call on
  the same pool worker DOES see the value, confirming the isolation is a
  property of how context propagates, not that ContextVar is unreadable
  from threads; a second plain submit() on the same reused worker reads
  None again, confirming no retention between tasks; and as_customer()
  resets its value on both normal exit and when the wrapped block raises.
