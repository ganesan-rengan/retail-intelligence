
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