# ADR 010: RAG retrieval is not robust to all natural phrasings

## Context

`search_policy` uses cosine similarity over `all-MiniLM-L6-v2` embeddings
(384-dim) against 18 policy chunks. A spot-check during development
tested two phrasings for the same underlying question:

- "Can I return an item I received a month ago?" — correctly retrieved
  all three relevant sections in the top 3, ordered by distance:
  `returns_policy / 2. Return Conditions` (0.3401), `1. Return
  Eligibility` (0.3910), and `shipping_policy / 4. Delivered Orders`
  (0.4351).
- "What is your return window?" — the SAME correct chunk
  (`returns_policy / 1. Return Eligibility`) ranked last of the top 5
  candidates (distance 0.6927), behind three topically unrelated
  sections including `refund_policy / 7. Policy Relationship` (0.6411).

Both questions ask the same thing. The second phrasing failed because
"window" does not appear anywhere in the source policy text (the
document says "within 30 days," never "window"), and `all-MiniLM-L6-v2`
— a small, general-purpose 22M-parameter model — does not reliably
bridge that lexical gap without stronger domain-specific training.

## Decision

Treat this as a known limitation of the retrieval approach, not a bug
to patch per-query. Two options were considered and rejected:

- Raising `SIMILARITY_THRESHOLD` to let "return window" through:
  rejected because the threshold is a global cutoff — loosening it to
  admit one weak match admits weak matches everywhere, trading
  precision on every other question for recall on this one.
- Adding phrasing hints to the policy documents (e.g. "also called the
  return window"): rejected because it optimizes the source text for
  the embedding model rather than for human readers, and only patches
  the specific phrasings anticipated in advance — it does not fix the
  underlying limitation, only hides individual instances of it.

Instead: the retrieval gap is documented and measured as it appears,
rather than the system quietly failing on phrasings nobody happened to
test.

## Consequence

`search_policy` will predictably underperform on questions phrased with
vocabulary distant from the source documents' literal wording. This is
measured and reported honestly rather than concealed by tuning the demo
to a small set of known-good phrasings. A production system facing this
limitation would likely need a larger or fine-tuned embedding model,
hybrid keyword+semantic search, or query rewriting.

## Update: caused a complete conversation failure under MAX_ITERATIONS

The gap escalated from "answered slowly via retry" (previous update) to
"never answered" in a later, unprompted live run. Asked to make a return
exception for a 45-day-late item, the agent tried 5 query
reformulations -- "return window policy exception late returns", "return
window time limit policy", "returns", "return window days", "window" --
and hit MAX_ITERATIONS=5 without ever finding returns_policy / 1. Return
Eligibility, the correct chunk. That same chunk was found on the FIRST
policy search of the immediately preceding conversation, seconds earlier,
using the phrasing "How many days do I have to return an item?"

The failed reformulations share a pattern: each is shorter and more
keyword-like than the last (5 words -> 4 words -> 1 word), while the
phrasing that actually works is a full natural-language question. This
is backwards for a sentence-embedding model -- all-MiniLM-L6-v2 is
trained to match semantic content across full sentences, not sparse
keyword overlap, so the model's own retry strategy under repeated
failure moves in the wrong direction for the retrieval method it's
working against.

This confirms MAX_ITERATIONS=5 (set at step 2.5, never previously
triggered by a real conversation) is doing necessary work: without it,
this conversation would retry indefinitely rather than fail cleanly with
an honest "did not produce a final answer" message.

## Consequence, revised

This is no longer a "sometimes slower" limitation -- it can cause total
conversation failure. Two concrete mitigations for V2, beyond measuring
retrieval accuracy directly: (1) a system-prompt instruction encouraging
full natural-language reformulation over keyword stripping when a policy
search fails, tested to see if it changes the model's retry pattern; (2)
a lower per-conversation retry cap specific to search_policy (e.g. 2
attempts) with a graceful "let me connect you with a human for policy
questions I can't find an answer to" fallback, rather than spending the
full MAX_ITERATIONS budget on one tool.

## Update: reproduced in the full agent loop, with a partial mitigation

The "return window" gap was reproduced live inside the full conversational
loop, not just in isolation: search_policy("What is your return window?")
returned {"error": "No matching policy found"}, matching the original
spot-check exactly.

The agent did not stop there. It reformulated the question twice --
"return policy days window" (weak match, distance 0.4878, borderline) then
"How many days do I have to return an item?" (distance 0.3570, correctly
matched Return Eligibility) -- and produced a correct final answer. This
mirrors the query-reformulation behavior first observed at step 2.5's
"cancel order" test.

This is a partial, not a fix: the correct answer arrived at a real cost of
3 tool calls and roughly 1,500 tokens for what should have been a 1-call,
~500-token exchange. The retrieval gap this ADR documents is unchanged;
what changes is that the agent's own retry behavior sometimes routes
around it, at meaningfully higher cost. The eval suite (2.20) should
measure both retrieval accuracy AND the token/iteration cost of recovery,
since a system that "eventually gets there" via retries is a materially
different (and more expensive) reliability story than one with accurate
first-pass retrieval.


## Update: a retrieval failure produced a false claim about document content

In a later run, the same underlying question (order 491725's return
window) retrieved returns_policy / 2. Return Conditions (about item
condition) but never found / 1. Return Eligibility (which states the
actual 30-day figure). Rather than expressing uncertainty, the agent
stated: "Our policy documents do not specify a fixed time-window in
days for returns" -- a confident claim that the DOCUMENT lacks
information, when the actual gap is in RETRIEVAL, not content. This is
a distinct failure shape from ADR-011's pattern (overstating a rule
that WAS retrieved): here the agent asserts something false about what
a document contains, based on not having seen the part that contains
it. Worth an eval case distinguishing "correctly reports uncertainty"
("I couldn't find that specific detail") from "incorrectly asserts
absence" ("the policy doesn't cover this") when retrieval genuinely
missed relevant content that exists.