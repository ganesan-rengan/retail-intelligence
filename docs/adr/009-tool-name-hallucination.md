# ADR 009: Tool name hallucination under domain-plausible questions

## Context

Early in agent-loop development, with only `get_order_status` registered
as a tool, the model was asked a return-policy question. It returned a
function call for `search_faq` — a tool name that was never defined
anywhere in the codebase. This was reproducible at temperature 0 across
multiple runs, not a one-off fluke.

A separate control question, clearly outside the retail-support domain
entirely (asking about the weather), did not trigger any invented tool
call — the model correctly responded in plain text that it had no way to
help with that.

The pattern that emerged: hallucination scaled with domain plausibility.
The model did not invent tools indiscriminately; it invented a tool
specifically when a question was plausibly within its stated role but no
matching capability existed. A question far outside any plausible
capability produced an honest decline instead.

## Decision

Treat this as expected LLM behavior to defend against structurally, not
a bug to suppress through prompting. The dispatcher (`dispatch_call`,
later `dispatch_with_gate`) validates every requested tool name against
the real registry before executing anything. An unrecognized name —
whether hallucinated or otherwise — returns a structured
`{"error": "Unknown tool: <name>"}` result rather than raising, so the
model receives a normal tool-result turn and can respond to the customer
honestly instead of the loop crashing.

No attempt was made to prevent the model from proposing nonexistent
tools through prompt wording. A system-prompt instruction is a request,
not a guarantee — the same reasoning applied consistently elsewhere in
this project (customer scoping, write-path confirmation) — so the
defense belongs in code that runs regardless of what the model attempts,
not in text asking it not to.

## Consequence

Adding `search_policy` as a real, registered tool later removed this
specific failure case — the model had a real capability to reach for
instead of inventing one. But the general risk remains open-ended: any
future gap between what the model believes it should be able to do and
what tools actually exist could reproduce the same pattern under a
different invented name. The dispatcher's validate-before-execute
behavior is the permanent defense; it does not depend on having
anticipated every specific hallucinated name in advance.