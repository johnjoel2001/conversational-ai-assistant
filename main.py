# # main.py

# import sys
# from bedrock_client import BedrockClient
# from memory import Memory
# from errors import ConfigurationError, BedrockInvocationError

# MAX_CONTEXT_TURNS = 3

# def build_prompt(history: list[tuple[str,str]], memory_obj: Memory) -> str:
#     """
#     Build a prompt that starts with a concise SYSTEM instruction, then memory and recent conversation,
#     then ends with 'Assistant:' for Llama to continue.
#     """
#     lines = []

#     # 1) Brief system instruction to suppress meta‐commentary
#     lines.append("System: Respond **only** with the assistant’s direct reply. Do NOT explain your reasoning or talk about instructions.")

#     # 2) Memory (if any)
#     mem_text = memory_obj.to_prompt()
#     if mem_text:
#         lines.append(mem_text)

#     # 3) Last few turns (user + assistant)
#     limited = history[-(MAX_CONTEXT_TURNS * 2):]
#     for role, text in limited:
#         prefix = "User:" if role == "user" else "Assistant:"
#         lines.append(f"{prefix} {text}")

#     # 4) Ready for the next assistant response
#     lines.append("Assistant:")
#     return "\n".join(lines)

# def run_cli():
#     try:
#         bedrock = BedrockClient()
#     except ConfigurationError as ce:
#         print(f"[Configuration Error] {ce}")
#         sys.exit(1)

#     memory = Memory()
#     history: list[tuple[str,str]] = []

#     print("=== Chat with Llama 3.3 70B Instruct (type 'exit' to quit) ===")
#     while True:
#         try:
#             user_input = input("You: ").strip()
#         except (KeyboardInterrupt, EOFError):
#             print("\nGoodbye!")
#             break

#         if not user_input:
#             continue
#         if user_input.lower() in ("exit", "quit"):
#             print("Goodbye!")
#             break

#         # Store any “My name is …” or “My favorite color is …”
#         memory.parse_and_store(user_input)
#         history.append(("user", user_input))

#         # Build the prompt (with the new SYSTEM instruction)
#         prompt_text = build_prompt(history, memory)

#         # Invoke Llama
#         try:
#             reply = bedrock.invoke(
#                 prompt_text,
#                 max_gen_len=512,
#                 temperature=0.5,
#                 top_p=0.9
#             )
#         except BedrockInvocationError as bie:
#             print(f"[Error] Could not get response from Llama: {bie}")
#             continue

#         # Print and store the assistant’s reply
#         print(f"Assistant: {reply}\n")
#         history.append(("assistant", reply))

# if __name__ == "__main__":
#     run_cli()

# main.py

import sys
from bedrock_client import BedrockClient
from errors import ConfigurationError, BedrockInvocationError

def build_prompt_single_turn(user_input: str) -> str:
    """
    Build a prompt that:
      1. Issues a SYSTEM instruction to be concise.
      2. Presents exactly "User: <input>" with no history.
      3. Ends with "Assistant:" so the model knows to reply.
    """
    lines = [
        "System: Respond ONLY with the assistant’s direct reply to the user message. "
        "Do NOT include any previous conversation or extra commentary.",
        f"User: {user_input}",
        "Assistant:"
    ]
    return "\n".join(lines)

def run_cli():
    try:
        bedrock = BedrockClient()
    except ConfigurationError as ce:
        print(f"[Configuration Error] {ce}")
        sys.exit(1)

    print("=== Chat with Llama 3.3 70B Instruct (type 'exit' to quit) ===")
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

        # Build a single-turn prompt with no history or memory
        prompt_text = build_prompt_single_turn(user_input)

        try:
            reply = bedrock.invoke(
                prompt_text,
                max_gen_len=512,
                temperature=0.5,
                top_p=0.9
            )
        except BedrockInvocationError as bie:
            print(f"[Error] Could not get response from Llama: {bie}")
            continue

        print(f"Assistant: {reply}\n")

if __name__ == "__main__":
    run_cli()

