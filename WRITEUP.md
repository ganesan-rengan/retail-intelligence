# Retail Intelligence — Project Write-Up

This document is the engineering story behind the two services in this
repo: what was built, why each significant decision was made, and —
the part most write-ups skip — exactly how each claim about the system
was actually verified, not just asserted. Where something is a real,
unresolved limitation, it's stated as one.

## The shape of the project

Two services share one Postgres database, by design from the start,
not bolted on later:

- **Project 1 — Demand Forecasting Service.** A LightGBM model over
  weekly retail demand, served through FastAPI, deployed on Render
  with a Neon Postgres backend. 70.3% WAPE against a 76.6% moving-
  average baseline, on a held-out chronological split.
- **Project 2 — Support Agent.** A tool-calling LLM agent (Gemini,
  via `google-genai`) that answers order-status and policy questions
  over the same database, retrieves policy text through a small RAG
  pipeline, and — the part that took the most engineering care — can
  actually file a return through a real, gated write path.

Both are real, deployed or locally-runnable, and both are covered by
real test suites. The rest of this document focuses on Project 2,
since that's where the interesting engineering decisions live.

## Why a write path is the hard part

Most portfolio agent projects stop at "the LLM can look things up."
The moment an agent can *change* something — file a return, issue a
refund, cancel an order — a new class of problem shows up: the model's
output is not a reliable authorization signal. Asking it nicely not to
confirm a return in the same breath it proposes one is a prompt, not a
guarantee.

So the actual engineering question for this project was: **what does
it take to make "the agent cannot do X" a property of the code, not a
property of the prompt?** Four specific answers to that question, each
built as its own gate, each independently proven rather than assumed
correct because it was designed carefully.

### Gate 1 — Confirmation

`propose_return()` and `confirm_return()` are two separate tools. A
proposal writes to a `pending_returns` table — never to the real
`returns` table — and returns a single-use token
(`secrets.token_urlsafe(32)`). Only `confirm_return(token)` can turn
that into a real return.

The part that actually enforces the gap between propose and confirm
doesn't live in either tool — it lives in the dispatcher
(`dispatch_with_gate()`), which tracks whether `propose_return` was
already called in the current conversation turn and structurally
refuses `confirm_return` if so, regardless of what the model attempts.

**How this was actually proven, not just designed:**
- A scripted, mocked model response that deliberately tries to chain
  `propose_return` → `confirm_return` in one turn — the refusal fires
  every time, independent of whether a live model would have
  cooperated (it usually did decline on its own, but that's the
  model's behavior, not the system's guarantee — the two are
  different claims, and only the mocked version proves the second one)
- A forced concurrent race: two real threads, synchronized with a
  SQLAlchemy engine-event listener so both threads' reads of the
  pending token genuinely overlap before either writes — not just two
  sequential calls, an actual race. One thread wins, the other gets a
  clean "already used" error, and exactly one `returns` row exists
  afterward. Verified stable across 5 consecutive runs.

### Gate 2 — Scoping

Every tool that touches customer-private data resolves `customer_id`
server-side; the model never supplies it and it never appears in any
tool's schema. Verified by direct code audit (not memory of the
design) against the live source twice — once early, and once again
after the identity mechanism changed, specifically to confirm the
*property* Gate 2 depends on (customer_id is never model-supplied)
survived that change even though the underlying binding mechanism did.

### Gate 3 — Idempotency

A confirmation token doubles as the real return's `idempotency_key`
under a database `UNIQUE` constraint — so Gate 1 and Gate 3 reinforce
each other at the database level, not just by design intent. This
claim was specifically *not* trusted on the strength of the schema
alone: the same forced concurrent-race test above is what proves it,
since a genuine double-confirm attempt is exactly the condition this
constraint exists to catch.

### Gate 4 — Audit logging

Every tool call — successes and Gate-1 refusals alike — writes one row
to an `agent_actions` table, with any confirmation token masked to 8
characters before being stored. This masking was verified by
deliberately breaking it: temporarily disabling the masking function
and confirming the test suite's assertions correctly failed and named
which field leaked, then restoring it and confirming the suite passed
clean again. A test that can't fail isn't proving anything; this one
was shown to be able to.

## Retrieval, and an honestly documented limitation

Policy documents are chunked by section heading (18 chunks across
returns/shipping/refund policy), embedded with `all-MiniLM-L6-v2`, and
retrieved via cosine similarity through pgvector.

This works well for direct phrasings and measurably worse for
others — a real, reproducible gap: "What is your return window?"
retrieves nothing useful, because the word "window" never appears in
the source policy text, and this small general-purpose embedding model
doesn't reliably bridge that lexical gap. This was found, measured with
real distance numbers, and — critically — **not silently tuned away**.
Two obvious fixes (loosening the similarity threshold globally, or
adding phrasing hints to the source documents) were considered and
rejected, both for the same reason: they'd optimize the system for
questions already known to be asked, not fix the actual limitation.
It's documented as a known gap (ADR-010), including a later finding
that it once caused the agent to retry five times and still fail a
conversation, and a separate instance where a retrieval miss produced
a confidently wrong claim about what the policy documents contain — a
different, more concerning failure shape than just "slow," now
recorded as its own eval case.

## Session identity

Customer identity in the agent's tool calls is resolved through a real
session mechanism (`login()` issues a token; `resolve_session()`
validates it; a `contextvars.ContextVar`, not a global, carries the
resolved `customer_id` for the duration of one conversation) — this
replaced an earlier hardcoded constant used during initial development.

Worth stating plainly, the same way every other limitation here is
stated: this is **identification, not authentication**. The `Customer`
table has no credential field of any kind, so `login()` only verifies
that a `customer_id` exists — anyone who knows or guesses one gets a
full session as that customer. The gate this system actually
guarantees is narrower and still real: the *model* can never choose or
influence which customer it's acting as, only a session set up before
the conversation began can. That's a meaningful property; it is not
the same property as "only the right person can start a session," and
the two are not conflated anywhere in this project's documentation.

The `ContextVar` choice over a simpler module-level variable was
deliberate and specifically checked, not assumed: a plain global would
have been a real, silent cross-conversation leak risk the moment
anything in this codebase ran concurrently — which turned out to be a
real, not hypothetical, scenario, since the Gate 1/3 race test above
genuinely does run two threads at once. Measured directly (not
inferred from documentation) that a `ContextVar` set in one thread
fails closed — returns the default, not another thread's value — in
both a plain `threading.Thread` and a `ThreadPoolExecutor` worker,
unless explicitly propagated via `copy_context()`. That measurement is
now a permanent, committed test, not a one-time finding.

## The engineering process itself

A pattern that shows up repeatedly enough to be worth naming directly:
several real mistakes were caught and corrected over the course of
this build, rather than shipped or glossed over —

- A ~4x undercount of the test suite's real size, caught by actually
  running `pytest --collect-only` instead of a manual grep
- A commit accidentally describing the wrong scope of its own diff,
  caught and amended before being trusted
- A test's expiry-boundary convention that briefly disagreed with an
  already-established one elsewhere in the same codebase, caught by
  cross-referencing rather than writing the new test in isolation
- A documentation reconstruction that initially introduced a factual
  error (misstating which of two retrieved passages was actually the
  closer match) — caught by checking the reconstructed prose against
  the real, underlying numbers, not by trusting the prose read
  smoothly

None of these were caught by assuming things were fine. They were
caught by the same standing discipline applied throughout: read the
real file before trusting a description of it, run the real command
before trusting a claimed result, and treat "this looks right" as a
reason to check harder, not a reason to stop checking.

## What's genuinely still open

Stated plainly, matching how every other limitation in this project is
handled:

- **No CI coverage for the support agent's test suites.** The existing
  GitHub Actions workflow covers Project 1's tests (which mock the
  database) but not `test_gate1.py`/`test_identity.py`, which
  deliberately exercise a real, unmocked Postgres database and one
  scenario makes live LLM calls. Wiring this in properly means a
  Postgres service container and a live-quota decision, not just
  adding a line to a workflow file — a real next step, not done yet.
- **Identification, not authentication**, as detailed above.
- **The RAG retrieval gap**, as detailed above.

## Why this write-up exists

Most of what makes this project worth looking at isn't the tech stack
or the line count — it's that nearly every non-trivial claim in it
(the gates hold, the race is real, the masking works, the thread
isolation fails closed) is backed by a test that was shown to actually
exercise the failure mode it claims to prevent, not just a test that
passes. That distinction — between "this looks correct" and "this was
shown to be correct, including under the specific condition where it
would fail if it weren't" — is the actual engineering practice this
project was built to demonstrate.
