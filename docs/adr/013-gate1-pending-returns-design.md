# ADR 013: Gate 1 — pending_returns design and verification

## Context

The write path (`propose_return` / `confirm_return`) is the only part of
the system that mutates real business state. A naive design would give
a single tool a `confirmed: bool` parameter set by the model — but that
makes "confirmation" just another value the model supplies on its own
judgment, not a fact about the conversation. The same reasoning that put
customer scoping in a SQL `WHERE` clause rather than a system-prompt
instruction applies here: confirmation has to be a structural property
of the conversation, not something the model asserts.

## Decision

Split the write into two tools with a real pause enforced between them.

**`pending_returns` is a separate table from `returns`, deliberately.**
`returns` must only ever contain returns that actually happened.
`pending_returns` holds proposals that may expire or be confirmed. No
`Order.pending_returns` collection was added on the ORM side — a
navigable Order → PendingReturn relationship would be a second access
path into proposal state that bypasses the single-use token entirely,
which would undermine the design's central guarantee that
`confirm_return(token)` is the *only* way in.

**The confirmation token is `secrets.token_urlsafe(32)`** (43 characters,
verified directly against Python's stdlib), not a sequential ID or
anything guessable, and doubles as `returns.idempotency_key` under a
`UNIQUE` constraint. This means Gate 1 (confirmation) and Gate 3
(idempotency) reinforce each other at the database level, not just by
design intent.

**The 30-day return-eligibility window and the 15-minute confirmation
TTL use different time references, deliberately.** The 30-day window is
anchored to `MAX(orders.order_date)` in the data (2011-12-09), computed
live on every call, because the dataset is a frozen historical snapshot
where real wall-clock time would reject every order regardless of
status. The 15-minute TTL is real wall-clock time
(`datetime.now() + timedelta(minutes=15)`), because a confirmation
window is about how long a live conversation stays valid, not something
tied to the dataset's age.

**The loop-level enforcement lives in `dispatch_with_gate()`**, not in
either tool. Nothing in `propose_return` or `confirm_return` themselves
prevents both being called in the same conversational turn — the
`FunctionDeclaration` description asks the model not to chain them, but
that is a request, not a guarantee. `dispatch_with_gate()` tracks
`proposed_this_turn`, scoped to a single `run_agent()` invocation, and
structurally refuses `confirm_return` if it is requested in the same
invocation that already called `propose_return`.

## Verification

Six checks, of increasing rigor, cover this design:

- **Sequential double-confirm** (`test_gate1.py` scenario 4): confirming
  the same token twice in a row is refused on the second attempt, and
  exactly one `returns` row exists.
- **Deterministic refusal, no model involved** (scenario 8a/8b):
  `dispatch_with_gate` called directly with `proposed_this_turn=True`
  returns the refusal and creates no `returns` row; the identical call
  with the flag `False` (the negative control) executes normally and
  creates exactly one row — proving the flag, not a general failure, is
  what causes the refusal.
- **Deterministic refusal through the full loop** (scenario 8d):
  `run_agent()` with the model's response mocked to force the chained
  attempt — the loop itself refuses, independent of whether a live
  model chooses to attempt the exploit on any given run.
- **Live-model corroboration** (scenario 7): asked directly to chain the
  calls, `gemini-3.5-flash-lite` declined on its own, twice, across
  separate runs. This is informative about the model's behavior but
  proves nothing about the code-level guarantee on its own, since the
  refusal branch was never exercised by it — scenario 8d exists
  specifically because this could not be relied on as proof.
- **A genuine concurrent race**, forced manually (not currently
  committed as a repository test — see Remaining Work below): two
  separate `get_session()` objects were opened, both read the same
  `PendingReturn` row with `status='pending'` before either wrote
  (mirroring true concurrent requests, not sequential ones), both
  staged a `Return` insert with the same `idempotency_key`, and both
  committed. The first commit succeeded. The second raised a real
  `psycopg2.errors.UniqueViolation` on `uq_returns_idempotency_key`,
  caught by `confirm_return`'s `except IntegrityError` handler, which
  correctly identified the constraint name and returned a clean
  "already been used" error rather than propagating the exception or
  creating a duplicate row.
- **A forced masking failure** (Gate 4): `_mask_tokens()` was
  temporarily disabled, and scenario 8d's log assertions -- added
  specifically to check for plaintext token leakage -- failed exactly as
  expected: three specific checks reported the raw token appearing in the
  logged arguments and outcome. Restoring masking made all nine checks in
  that scenario pass again. This confirms the log-masking assertions can
  actually fail, not just pass regardless of whether masking exists. The
  distinction from the race test matters: the assertions themselves are
  committed in `test_gate1.py` and re-run every time, so masking IS
  covered by a permanent test. Only the meta-verification -- deliberately
  breaking masking to prove those assertions can fail -- was a one-time
  manual step during development, with no standing automated way to
  repeat it.

## Consequence

The confirmation gate and the idempotency guarantee are proven to hold
under conditions stronger than normal sequential use — a mocked
adversarial loop attempt and a genuinely forced database race — not
just under happy-path testing.

**Remaining work:** the concurrent-race verification was run as an ad
hoc script during development and is not currently a committed,
re-runnable test. It should be added to `test_gate1.py` as a permanent
scenario so this specific guarantee is re-verified on every run rather
than resting on a one-time manual check.