from flask import Flask, request, jsonify, Response
from bedrock_client import BedrockClient
from errors import ConfigurationError, BedrockInvocationError

app = Flask(__name__)

def build_prompt_single_turn(user_input: str) -> str:
    """
    Build a one-off prompt that tells Llama to ignore history and only reply directly:
      System: Respond ONLY with the assistant’s direct reply to the user message. Do NOT include any previous conversation or extra commentary.
      User: <user_input>
      Assistant:
    """
    return "\n".join([
        "System: Respond ONLY with the assistant’s direct reply to the user message. "
        "Do NOT include any previous conversation or extra commentary.",
        f"User: {user_input}",
        "Assistant:"
    ])

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    # 1) Parse JSON body
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Send JSON like { 'message':'Hello' }"}), 400

    user_input = data["message"].strip()
    if not user_input:
        return jsonify({"error": "Message cannot be empty."}), 400

    # 2) Build the Bedrock prompt
    prompt_text = build_prompt_single_turn(user_input)

    # 3) Invoke Bedrock
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

    # 4) Return JSON {"reply": "<assistant_reply>"}
    return jsonify({"reply": reply})

@app.route("/", methods=["GET"])
def index():
    # Return a minimal single-page HTML+JS chat UI
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Chat with Llama 3.3 70B Instruct</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #chat-box {
          border: 1px solid #ccc;
          padding: 10px;
          width: 500px;
          height: 400px;
          overflow-y: scroll;
          margin-bottom: 10px;
        }
        .message.user { color: blue; margin: 5px 0; }
        .message.assistant { color: green; margin: 5px 0; }
        #input-area { display: flex; }
        #user-input { flex: 1; padding: 8px; font-size: 16px; }
        #send-btn { padding: 8px 16px; margin-left: 8px; font-size: 16px; }
      </style>
    </head>
    <body>
      <h2>Chat with Llama 3.3 70B Instruct</h2>
      <div id="chat-box"></div>
      <div id="input-area">
        <input type="text" id="user-input" placeholder="Type your message…" />
        <button id="send-btn">Send</button>
      </div>
      <script>
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');

        function addMessage(role, text) {
          const div = document.createElement('div');
          div.classList.add('message', role);
          div.textContent = (role === 'user' ? 'You: ' : 'Assistant: ') + text;
          chatBox.appendChild(div);
          chatBox.scrollTop = chatBox.scrollHeight;
        }

        async function sendMessage() {
          const msg = userInput.value.trim();
          if (!msg) return;

          addMessage('user', msg);
          userInput.value = '';
          userInput.disabled = true;
          sendBtn.disabled = true;

          try {
            const resp = await fetch('/chat', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message: msg })
            });
            const data = await resp.json();
            if (data.reply) {
              addMessage('assistant', data.reply);
            } else if (data.error) {
              addMessage('assistant', '[Error] ' + data.error);
            }
          } catch (err) {
            addMessage('assistant', '[Network error]');
          }

          userInput.disabled = false;
          sendBtn.disabled = false;
          userInput.focus();
        }

        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keydown', function(e) {
          if (e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
          }
        });

        userInput.focus();
      </script>
    </body>
    </html>
    """
    return Response(html, mimetype='text/html')

if __name__ == "__main__":
    # Listen on port 8080 for App Runner
    app.run(host="0.0.0.0", port=8080)
