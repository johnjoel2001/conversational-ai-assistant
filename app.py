# app.py

from flask import Flask, request, jsonify
from bedrock_client import BedrockClient
from errors import ConfigurationError, BedrockInvocationError

app = Flask(__name__)

def build_prompt_single_turn(user_input: str) -> str:
    """
    Build a prompt that:
      1) Issues a SYSTEM instruction to be concise.
      2) Presents exactly "User: <input>" with no history.
      3) Ends with "Assistant:" so the model knows to reply.
    """
    lines = [
        "System: Respond ONLY with the assistant’s direct reply to the user message. "
        "Do NOT include any previous conversation or extra commentary.",
        f"User: {user_input}",
        "Assistant:"
    ]
    return "\n".join(lines)

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Send JSON like { 'message': 'Hello' }"}), 400

    user_input = data["message"].strip()
    if not user_input:
        return jsonify({"error": "Message cannot be empty."}), 400

    # Build the single-turn prompt
    prompt_text = build_prompt_single_turn(user_input)

    try:
        bedrock = BedrockClient()
        reply = bedrock.invoke(
            prompt_text,
            max_gen_len=512,
            temperature=0.5,
            top_p=0.9
        )
    except ConfigurationError as ce:
        return jsonify({"error": f"Configuration error: {ce}"}), 500
    except BedrockInvocationError as be:
        return jsonify({"error": f"Llama invocation failed: {be}"}), 502

    return jsonify({"reply": reply})

@app.route("/", methods=["GET"])
def index():
    return (
        "<h3>Llama 3.3 70B Instruct is running.</h3>"
        "<p>POST JSON { 'message': '…' } to <code>/chat</code>.</p>"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
