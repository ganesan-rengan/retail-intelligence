"""A hand-written agent loop: multi-round tool calling with a name registry,
graceful handling of invented tool names, and per-iteration token accounting.

Continues from 02_tool_calling.py's single manual round trip -- this loops
until the model returns a turn with no function calls, or MAX_ITERATIONS is
hit. Uses the real tools from support_agent.src.tools: get_order_status,
search_policy, get_demand_forecast, propose_return, confirm_return.
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from support_agent.src.tools import (
    get_order_status_for_model,
    order_status_declaration,
    search_policy,
    search_policy_declaration,
    get_demand_forecast,
    demand_forecast_declaration,
    propose_return_for_model,
    propose_return_declaration,
    confirm_return_for_model,
    confirm_return_declaration,
)

load_dotenv()
client = genai.Client(api_key=os.environ["LLM_API_KEY"])
MODEL = "gemini-3.5-flash-lite"

MAX_ITERATIONS = 5


TOOLS = {
    "get_order_status": get_order_status_for_model,
    "search_policy": search_policy,
    "get_demand_forecast": get_demand_forecast,
    "propose_return": propose_return_for_model,
    "confirm_return": confirm_return_for_model,
}


tools = types.Tool(
    function_declarations=[
        order_status_declaration,
        search_policy_declaration,
        demand_forecast_declaration,
        propose_return_declaration,
        confirm_return_declaration,
    ]
)
config = types.GenerateContentConfig(tools=[tools], temperature=0)


def dispatch_call(call: types.FunctionCall) -> tuple[str, bool, dict]:
    """Run one function call against the registry. Never raises: an unknown
    tool name returns a structured error dict instead of a KeyError, so a
    hallucinated tool name degrades gracefully rather than crashing the loop.
    """
    name = call.name
    args = dict(call.args or {})
    known = name in TOOLS
    if known:
        result = TOOLS[name](**args)
    else:
        result = {"error": f"Unknown tool: {name}"}
    return name, known, result


def run_agent(
    user_message: str, messages: list[types.Content] | None = None
) -> tuple[str, list[types.Content]]:
    """Loop the model against the tool registry until it returns a turn with
    no function calls, or MAX_ITERATIONS is hit.

    messages is the caller's held conversation history, not any state
    run_agent() keeps itself. Omitted (None), this starts a fresh
    conversation, unchanged from before this parameter existed. Passed in,
    user_message is appended to it before the loop continues -- enabling a
    genuine pause between two separate top-level calls (e.g. a propose/
    confirm write-path gate), since only the caller, not run_agent(), holds
    what happened in between.
    """
    if messages is None:
        messages = []
    messages.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
    total_tokens = 0
    proposed_this_turn = False

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = client.models.generate_content(model=MODEL, contents=messages, config=config)

        usage = response.usage_metadata
        thoughts = usage.thoughts_token_count or 0 if usage else 0
        iter_total = (usage.total_token_count or 0) if usage else 0
        total_tokens += iter_total

        print(f"--- iteration {iteration} ---")
        print(
            f"  tokens: prompt={getattr(usage, 'prompt_token_count', None)}"
            f"candidates={getattr(usage, 'candidates_token_count', None)} "
            f"thoughts={thoughts} total={iter_total} "
            f"(running total={total_tokens})"
        )

        parts = response.candidates[0].content.parts or []
        function_calls = [p for p in parts if p.function_call]
        text_parts = "".join(p.text for p in parts if p.text)

        if not function_calls:
            print("  no function calls -- this is the final answer")
            print(f"  TOTAL TOKENS THIS CONVERSATION (incl. thoughts): {total_tokens}")
            messages.append(response.candidates[0].content)
            return text_parts, messages

        if text_parts:
            print(f"  lead-in text: {text_parts.strip()!r}")

        messages.append(response.candidates[0].content)

        response_parts = []
        for call_part in function_calls:
            call = call_part.function_call
            args = dict(call.args or {})

            # Gate 1's actual enforcement point. propose_return_declaration's
            # description asks the model not to chain these in one turn, but
            # a prompt instruction is a request, not a guarantee -- this is
            # what actually stops it if it tries anyway. proposed_this_turn
            # is local to this single run_agent() invocation, so a genuinely
            # separate, later run_agent() call is never affected by it.
            if call.name == "confirm_return" and proposed_this_turn:
                name, known = call.name, True
                result = {
                    "error": (
                        "confirm_return cannot be called in the same "
                        "conversation turn as propose_return. A separate "
                        "message from the customer confirming they want to "
                        "proceed is required first."
                    )
                }
            else:
                name, known, result = dispatch_call(call)
                if name == "propose_return" and known:
                    proposed_this_turn = True

            print(f"  tool: {name} [{'KNOWN' if known else 'INVENTED'}]")
            print(f"    args: {args}")
            print(f"    result: {result}")

            response_parts.append(types.Part.from_function_response(name=name, response=result))

        messages.append(types.Content(role="user", parts=response_parts))

    print(f"--- hit MAX_ITERATIONS={MAX_ITERATIONS} without a final answer ---")
    print(f"  TOTAL TOKENS THIS CONVERSATION (incl. thoughts): {total_tokens}")
    return f"Agent did not produce a final answer within {MAX_ITERATIONS} iterations.", messages


def test_unknown_tool_dispatch() -> None:
    """Deterministic check of the dispatch-safety path. Rather than waiting
    for the model to hallucinate a tool name -- which it has not done
    reliably since search_policy was added (see ADR-009) -- construct a
    FunctionCall for a name that isn't in the registry and push it through
    the SAME dispatch_call() the live loop uses. Confirms it degrades to a
    structured error rather than raising a KeyError.
    """
    fake_call = types.FunctionCall(name="cancel_subscription", args={"order_id": "1041"})
    name, known, result = dispatch_call(fake_call)

    assert known is False, f"expected unknown, got known={known}"
    assert result == {"error": "Unknown tool: cancel_subscription"}, result

    print("=" * 70)
    print("TEST: unknown tool dispatch (forced, not waiting on the model)")
    print("=" * 70)
    print(f"  name={name} known={known} result={result}")
    print("  PASS: unknown tool name did not raise, returned structured error")
    print()


if __name__ == "__main__":
    test_unknown_tool_dispatch()

    for message in [
        "Is order 491725 shipped, and what is your return window?",
        "Can you make an exception and let me return something 45 days late?",
        "Is product 15036 expected to be in high demand soon?",
    ]:
        print("=" * 70)
        print(f"USER: {message}")
        print("=" * 70)
        answer, _ = run_agent(message)
        print()
        print(f"FINAL ANSWER: {answer}")
        print()