# main.py

import sys
import traceback

from bedrock_client import BedrockClient
from memory import Memory
from errors import ConfigurationError, BedrockInvocationError

MAX_CONTEXT_TURNS = 3

def build_prompt(history: list, memory_obj: Memory) -> str:
    """
    Combine memory text + last few turns into one string ending with "Assistant:".
    """
    lines = []
    mem_text = memory_obj.to_prompt()
    if mem_text:
        lines.append(mem_text)

    limited = history[-(MAX_CONTEXT_TURNS * 2):]
    for role, text in limited:
        prefix = "User:" if role == "user" else "Assistant:"
        lines.append(f"{prefix} {text}")

    lines.append("Assistant:")
    return "\n".join(lines)

def run_cli():
    try:
        bedrock = BedrockClient()
    except ConfigurationError as ce:
        print(f"[Configuration Error] {ce}")
        sys.exit(1)

    memory = Memory()
    history = []

    print("=== Chat with Bedrock (type 'exit' to quit) ===")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        memory.parse_and_store(user_input)
        history.append(("user", user_input))

        prompt_text = build_prompt(history, memory)

        try:
            reply = bedrock.invoke(prompt_text)
        except BedrockInvocationError as bie:
            print(f"[Error] Could not get response from Bedrock: {bie}")
            continue

        print(f"Assistant: {reply}\n")
        history.append(("assistant", reply))

if __name__ == "__main__":
    run_cli()
