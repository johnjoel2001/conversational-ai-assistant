# app.py

import json
from flask import Flask, request, jsonify
from bedrock_client import BedrockClient
from memory import Memory
from errors import ConfigurationError, BedrockInvocationError

app = Flask(__name__)
global_memory = Memory()
history = []
MAX_CONTEXT_TURNS = 3

def build_prompt(history, memory_obj):
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

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Send JSON like { 'message': 'Hello' }"}), 400

    user_input = data["message"].strip()
    if not user_input:
        return jsonify({"error": "Message cannot be empty."}), 400

    global_memory.parse_and_store(user_input)
    history.append(("user", user_input))

    prompt_text = build_prompt(history, global_memory)

    try:
        bedrock = BedrockClient()
        reply = bedrock.invoke(prompt_text)
    except ConfigurationError as ce:
        return jsonify({"error": f"Configuration error: {ce}"}), 500
    except BedrockInvocationError as be:
        return jsonify({"error": f"Bedrock invocation failed: {be}"}), 502

    history.append(("assistant", reply))
    return jsonify({"reply": reply})

@app.route("/", methods=["GET"])
def index():
    return (
        "<h3>Assistant is running.</h3>"
        "<p>Send POST to <code>/chat</code> with JSON { 'message':'…' }.</p>"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
