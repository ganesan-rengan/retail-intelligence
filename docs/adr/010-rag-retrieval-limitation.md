
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