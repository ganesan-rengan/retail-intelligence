"""A hand-written agent loop: multi-round tool calling with a name registry,
graceful handling of invented tool names, and per-iteration token accounting.

Continues from 02_tool_calling.py's single manual round trip -- this loops
until the model returns a turn with no function calls, or MAX_ITERATIONS is
hit. get_order_status is real (queries the database, scoped to a customer);
search_policy stays fake until RAG (2.8-2.11).
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
)
load_dotenv()
client = genai.Client(api_key=os.environ["LLM_API_KEY"])
MODEL = "gemini-3.5-flash-lite"

MAX_ITERATIONS = 5


# --- search_policy is still fake -- stays canned until RAG (2.8-2.11). ----




TOOLS = {
    "get_order_status": get_order_status_for_model,
    "search_policy": search_policy,
}


# --- Tool declarations: the interface the model actually sees. ------------



tools = types.Tool(function_declarations=[order_status_declaration, search_policy_declaration])
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


def run_agent(user_message: str) -> str:
    """Loop the model against the tool registry until it returns a turn with
    no function calls, or MAX_ITERATIONS is hit."""
    messages = [types.Content(role="user", parts=[types.Part(text=user_message)])]
    total_tokens = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = client.models.generate_content(model=MODEL, contents=messages, config=config)

        usage = response.usage_metadata
        thoughts = usage.thoughts_token_count or 0 if usage else 0
        iter_total = (usage.total_token_count or 0) if usage else 0
        total_tokens += iter_total

        print(f"--- iteration {iteration} ---")
        print(
            f"  tokens: prompt={getattr(usage, 'prompt_token_count', None)} "
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
            return text_parts

        # Lead-in text on a turn that also calls a tool: log it (shows the
        # model's stated intent, useful for debugging a wrong tool pick),
        # but it is never part of the final returned answer.
        if text_parts:
            print(f"  lead-in text: {text_parts.strip()!r}")

        # The model's own turn -- including its function_call parts -- must
        # go into the conversation BEFORE the function responses. The
        # responses only make sense next to the request they're answering;
        # getting this backwards produces confusing model behaviour, not an
        # error, so there is nothing that would catch the mistake later.
        messages.append(response.candidates[0].content)

        response_parts = []
        for call_part in function_calls:
            name, known, result = dispatch_call(call_part.function_call)
            args = dict(call_part.function_call.args or {})

            print(f"  tool: {name} [{'KNOWN' if known else 'INVENTED'}]")
            print(f"    args: {args}")
            print(f"    result: {result}")

            response_parts.append(types.Part.from_function_response(name=name, response=result))

        messages.append(types.Content(role="user", parts=response_parts))

    print(f"--- hit MAX_ITERATIONS={MAX_ITERATIONS} without a final answer ---")
    print(f"  TOTAL TOKENS THIS CONVERSATION (incl. thoughts): {total_tokens}")
    return f"Agent did not produce a final answer within {MAX_ITERATIONS} iterations."


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
        "Where is my order 491725?",
        "Is order 491725 shipped, and what is your return window?",
        "Cancel order 491725",
        "Where is my order 999999?",
    ]:
        print("=" * 70)
        print(f"USER: {message}")
        print("=" * 70)
        answer = run_agent(message)
        print()
        print(f"FINAL ANSWER: {answer}")
        print()