"""Tool calling: watch a function_call Part come back instead of text."""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["LLM_API_KEY"])
MODEL = "gemini-3.6-flash"


# The actual Python function. The model never touches this.
def get_order_status(order_id: str) -> dict:
    """Fake lookup. Returns canned data."""
    fake = {
        "1041": {"status": "shipped", "order_date": "2011-11-14"},
        "2055": {"status": "delivered", "order_date": "2011-10-02"},
    }
    if order_id not in fake:
        return {"error": f"Order {order_id} not found"}
    return {"order_id": order_id, **fake[order_id]}


# The DESCRIPTION the model reads to decide. This is the interface.
order_status_tool = types.FunctionDeclaration(
    name="get_order_status",
    description=(
        "Look up the current status and order date of a specific order. "
        "Requires the exact order ID. Use this when a customer asks where "
        "their order is, whether it has shipped, or when it was placed."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "order_id": types.Schema(
                type=types.Type.STRING,
                description="The order ID, e.g. '1041'",
            ),
        },
        required=["order_id"],
    ),
)

tools = types.Tool(function_declarations=[order_status_tool])
config = types.GenerateContentConfig(tools=[tools], temperature=0)

KNOWN_TOOLS = {"get_order_status"}


def ask(text: str) -> object:
    """One-shot call with the toolset attached."""
    return client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=text)])],
        config=config,
    )


def show(label: str, response) -> None:
    print("=" * 60)
    print(label)
    print("=" * 60)

    parts = response.candidates[0].content.parts
    if not parts:
        print(f"  (no parts) finish_reason={response.candidates[0].finish_reason}")
        print()
        return

    for i, part in enumerate(parts):
        if part.function_call:
            name = part.function_call.name
            known = "KNOWN" if name in KNOWN_TOOLS else "*** INVENTED ***"
            print(f"  parts[{i}] -> function_call [{known}]")
            print(f"    name: {name}")
            print(f"    args: {dict(part.function_call.args)}")
        elif part.text:
            print(f"  parts[{i}] -> text")
            print(f"    text: {part.text[:250]}")
        else:
            print(f"  parts[{i}] -> other: {part}")
    print()


# --- Turn 1: a question that needs the tool -------------------------------
messages = [
    types.Content(role="user", parts=[types.Part(text="Where is my order 1041?")])
]

response = client.models.generate_content(
    model=MODEL, contents=messages, config=config
)
show("TURN 1 - model asks for a tool", response)

# --- We decide whether to run it. This is where gates would go. -----------
call = response.candidates[0].content.parts[0].function_call
result = get_order_status(**dict(call.args))
print(f"WE executed {call.name} and got: {result}\n")

# --- Turn 2: send the result back so it can phrase a reply ----------------
messages.append(response.candidates[0].content)
messages.append(
    types.Content(
        role="user",
        parts=[types.Part.from_function_response(name=call.name, response=result)],
    )
)

response2 = client.models.generate_content(
    model=MODEL, contents=messages, config=config
)
show("TURN 2 - model phrases the answer", response2)


# --- Control: adjacent to the domain, but no tool exists for it -----------
show("CONTROL 1 - policy question, no matching tool", ask("What is your return policy?"))

# --- Control 2: clearly outside the domain entirely -----------------------
show("CONTROL 2 - clearly out of scope", ask("What's the weather in Chennai?"))