"""Minimal interactive CLI for the support agent.

Run from the repo root:  uv run python -m support_agent.experiments.chat

Logs in with identity.login(customer_id) -- a stand-in for a real login
step, same as the DEMO_CUSTOMER_ID call in agent_loop_03.py's __main__
block, except here the customer_id comes from whoever is typing, not a
hardcoded constant. This is IDENTIFICATION, not authentication: login()
only checks that the customer_id exists (see README.md's Limitations
section and identity.py's own docstring). Typing a real, guessed
customer_id that is not yours gets you a full session as that customer;
this script does nothing to close that gap.

Everything after login is a plain read-eval-print loop around
run_agent(), which already does the real work (tool calling, Gate 1-4
enforcement, session resolution). This file adds no new tools, no new
gates, and no persistence -- closing it loses the conversation.
"""

import contextlib
import io

from support_agent.src.identity import login

MAX_LOGIN_ATTEMPTS = 3
YOU_PROMPT = "You: "
THINKING_MESSAGE = "Agent is thinking..."
# The visible line when clear_thinking_line() needs to erase it is the
# input() prompt PLUS THINKING_MESSAGE printed right after it, on the same
# line -- not THINKING_MESSAGE alone. Computed from the same two constants
# input(YOU_PROMPT) and print(THINKING_MESSAGE) use, so this can't drift
# out of sync with either one.
CLEAR_WIDTH = len(YOU_PROMPT) + len(THINKING_MESSAGE)


def do_login() -> str:
    """Prompt for a customer ID and log in, up to MAX_LOGIN_ATTEMPTS times.

    Returns a valid session_token. Exits the process (code 1) if every
    attempt fails, including attempts spent on non-integer input -- typing
    "abc" costs an attempt the same as a real but unknown customer_id.
    """
    for attempt in range(1, MAX_LOGIN_ATTEMPTS + 1):
        raw = input("Customer ID: ").strip()
        try:
            customer_id = int(raw)
        except ValueError:
            print(f"'{raw}' is not a valid customer ID (expected a number).")
            continue

        result = login(customer_id)
        if "error" in result:
            print(f"{result['error']}.")
            continue

        print(
            f"Signed in as customer {result['customer_id']} "
            f"(session valid until {result['expires_at']})."
        )
        return result["session_token"]

    print(f"Too many failed login attempts ({MAX_LOGIN_ATTEMPTS}). Exiting.")
    raise SystemExit(1)


def clear_thinking_line() -> None:
    """Erase "You: " + THINKING_MESSAGE from the current terminal line via a
    carriage return, CLEAR_WIDTH spaces to overwrite the whole visible line
    (not just THINKING_MESSAGE's own length), then another carriage return --
    so whatever prints next (the answer, or an error) starts clean at column
    0 instead of appending to, or leaving trailing characters of, "You:
    Agent is thinking...".
    """
    print("\r" + " " * CLEAR_WIDTH + "\r", end="")


def main() -> None:
    try:
        print("Starting support agent...")
        # Deferred past the module's top-level imports so "Starting support
        # agent..." prints before this runs -- this import is what pulls in
        # tools.py, which loads SentenceTransformer at import time (the HF
        # warning and "Loading weights" bar). Both write to stderr, verified
        # empirically (uv run ... captured separately: stdout had none of
        # either line), so only stderr needs redirecting here.
        with contextlib.redirect_stderr(io.StringIO()):
            from support_agent.experiments.agent_loop_03 import INVALID_SESSION_MESSAGE, run_agent

        session_token = do_login()
        messages = None  # None starts a fresh conversation in run_agent()

        while True:
            user_input = input(YOU_PROMPT).strip()
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye.")
                raise SystemExit(0)

            print(THINKING_MESSAGE, end="", flush=True)
            try:
                # redirect_stdout only, and only around this one call: it
                # silences run_agent()/_run_loop's own per-iteration print()s
                # (agent_loop_03.py) without touching that file. An exception
                # raised inside this block still propagates out of the `with`
                # normally to the except below (confirmed with a standalone
                # script: sys.stdout is genuinely restored, not left
                # redirected, once this block exits either way).
                with contextlib.redirect_stdout(io.StringIO()):
                    answer, messages = run_agent(user_input, session_token, messages)
            except Exception as exc:
                # Deliberately bare Exception, not BaseException: KeyboardInterrupt
                # must still reach the handler below, not be swallowed here. Nothing
                # in run_agent()/_run_loop (agent_loop_03.py) catches a network or
                # API error from the model call, so without this it would propagate
                # here uncaught and kill the whole CLI over one bad turn.
                #
                # messages is left exactly as run_agent() last mutated it. It's a
                # mutable list, and _run_loop appends this turn's user message to
                # it (agent_loop_03.py) before calling the model -- so if the
                # exception happened during or after that call, the list here may
                # already contain a user turn with no matching reply. Correcting
                # that would mean reaching into agent_loop_03's Content internals
                # from outside it, which this script isn't meant to touch; accepted
                # as a known limitation instead of guessed at.
                clear_thinking_line()
                print(f"Something went wrong on that turn ({type(exc).__name__}); please try again.")
                continue

            clear_thinking_line()
            if answer == INVALID_SESSION_MESSAGE:
                print("Your session has expired. Please log in again.")
                session_token = do_login()
                # messages is NOT carried into the new session, even though the
                # variable still exists in this scope: a re-login can be a
                # DIFFERENT customer_id, and carrying old messages forward would
                # hand that new session's run_agent() call prior conversation
                # turns containing the PREVIOUS customer's order details as
                # context, regardless of what the new session is actually scoped
                # to. That's a real scoping concern, not just tidiness.
                messages = None
                continue

            print(f"Agent: {answer}")
    except KeyboardInterrupt:
        print("\nGoodbye.")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
