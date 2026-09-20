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