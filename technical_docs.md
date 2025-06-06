CopyPublishTechnical Documentation
This document describes the structure, components, and deployment steps for the Llama 3.3 70B Instruct Chat Service, which uses Amazon Bedrock to power a simple browser‐based chat interface. It covers:

Project structure and purpose
Detailed file descriptions (bedrock_client.py, app.py, etc.)
Environment setup and local run instructions
Deployment on AWS App Runner
Testing strategy and coverage

Table of Contents

Overview
Project Structure
bedrock_client.py
app.py
memory.py and errors.py
requirements.txt
Environment Variables
Local Setup and Run
Deployment on AWS App Runner
Testing and Coverage
Appendix: Sample IAM Trust Policy

Overview
Name: Llama 3.3 70B Instruct Chat Service
Purpose:

Provide a minimal chat UI (in a web browser) that accepts a single user message, sends it to Amazon Bedrock's Llama 3.3 70B Instruct model, and returns the assistant's direct reply.
All responses are "single‐turn": the model does not reference previous messages or internal reasoning.

Key Features:

Simple HTML/JavaScript UI served at /
/chat endpoint that accepts JSON {"message":"…"} and returns {"reply":"…"}
Robust Bedrock client that handles multiple response‐format variants (generation, completion, text, choices, messages)
Automated tests (pytest) achieving ≥ 80% coverage
Fully deployed via AWS App Runner with an IAM role allowing bedrock-runtime:InvokeModel

Project Structure
conversational-ai-assistant/
├── app.py
├── bedrock_client.py
├── errors.py
├── memory.py
├── requirements.txt
├── tests/
│   ├── test_bedrock_client.py
│   └── test_flask_endpoints.py
└── README.md

app.py - Flask application exposing:

GET / → returns a static HTML/JS page with a chat interface
POST /chat → receives JSON → invokes Bedrock → returns JSON reply


bedrock_client.py - Wrapper around boto3.client("bedrock-runtime") to invoke Llama 3.3 70B Instruct. Builds the JSON request and parses the JSON response.
memory.py - (Optional) A simple slot‐based memory store. Not used in the single‐turn UI but available for future multi‐turn expansions.
errors.py - Defines two custom exceptions:

ConfigurationError → missing/invalid environment variables or client initialization issues
BedrockInvocationError → invocation or response‐parsing failures


requirements.txt - Lists Python dependencies:
boto3
flask
pytest
pytest-mock

tests/

test_bedrock_client.py → Unit tests for BedrockClient, covering all parsing branches.
test_flask_endpoints.py → Tests for the Flask endpoints (/ and /chat) using Flask's test client and a fake BedrockClient.


README.md - (This document is the main technical reference; README.md may contain short installation instructions.)

bedrock_client.py
python# bedrock_client.py

import os
import json
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from errors import ConfigurationError, BedrockInvocationError

class BedrockClient:
    """
    Wrapper around AWS Bedrock's invoke_model API for Llama 3.3 70B Instruct.
    Expects two environment variables:
      - AWS_REGION
      - BEDROCK_MODEL_ID (e.g. "meta.llama3-3-70b-instruct-v1:0")
    """

    def __init__(self):
        region = os.getenv("AWS_REGION")
        model_id = os.getenv("BEDROCK_MODEL_ID")

        if not region:
            raise ConfigurationError("Missing AWS_REGION environment variable.")
        if not model_id:
            raise ConfigurationError("Missing BEDROCK_MODEL_ID environment variable.")

        self.model_id = model_id
        try:
            self.client = boto3.client("bedrock-runtime", region_name=region)
        except Exception as e:
            raise ConfigurationError(f"Failed to create Bedrock client: {e}")

    def invoke(
        self,
        prompt: str,
        max_gen_len: int = 512,
        temperature: float = 0.5,
        top_p: float = 0.9
    ) -> str:
        """
        Send a prompt to Llama 3.3 70B Instruct and return the assistant's reply.
        Builds JSON of the form:
          {
            "prompt": "<System: …\nUser: …\nAssistant:>",
            "max_gen_len": 512,
            "temperature": 0.5,
            "top_p": 0.9
          }

        Then parses the response, checking (in order):
          1. parsed["generation"]
          2. parsed["completion"]
          3. parsed["text"]
          4. parsed["choices"][0]["message"]["content"][0]["text"]
          5. parsed["messages"][0]["content"][0]["text"]

        Raises:
          - BedrockInvocationError if invocation fails or no valid text is found.
        """

        # 1) Construct the request body
        body_dict = {
            "prompt": prompt,
            "max_gen_len": max_gen_len,
            "temperature": temperature,
            "top_p": top_p
        }
        invoke_args = {
            "modelId": self.model_id,
            "contentType": "application/json",
            "accept": "application/json",
            "body": json.dumps(body_dict).encode("utf-8")
        }

        # 2) Invoke Bedrock
        try:
            response = self.client.invoke_model(**invoke_args)
        except (BotoCoreError, ClientError) as aws_err:
            raise BedrockInvocationError(
                f"Failed to invoke Bedrock model: {aws_err}",
                original_exception=aws_err
            )

        # 3) Read and parse the JSON response
        try:
            raw_bytes = response["body"].read()
            decoded = raw_bytes.decode("utf-8")
            parsed = json.loads(decoded)
        except Exception as parse_err:
            raise BedrockInvocationError(
                f"Failed to parse Bedrock response body: {parse_err}",
                original_exception=parse_err
            )

        # 4) Extract the assistant's reply from one of the possible fields

        # a) "generation"
        gen = parsed.get("generation")
        if isinstance(gen, str) and gen.strip():
            return gen.strip()

        # b) "completion"
        comp = parsed.get("completion")
        if isinstance(comp, str) and comp.strip():
            return comp.strip()

        # c) "text"
        txt = parsed.get("text")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        # d) "choices"[0]["message"]["content"][0]["text"]
        choices = parsed.get("choices")
        if (
            isinstance(choices, list)
            and len(choices) > 0
            and isinstance(choices[0], dict)
            and "message" in choices[0]
            and isinstance(choices[0]["message"], dict)
        ):
            msg = choices[0]["message"]
            content_list = msg.get("content", [])
            if (
                isinstance(content_list, list)
                and len(content_list) > 0
                and isinstance(content_list[0], dict)
                and "text" in content_list[0]
            ):
                text_val = content_list[0]["text"]
                if isinstance(text_val, str) and text_val.strip():
                    return text_val.strip()

        # e) "messages"[0]["content"][0]["text"]
        llm_msgs = parsed.get("messages")
        if (
            isinstance(llm_msgs, list)
            and len(llm_msgs) > 0
            and isinstance(llm_msgs[0], dict)
            and "content" in llm_msgs[0]
        ):
            content_list = llm_msgs[0].get("content", [])
            if (
                isinstance(content_list, list)
                and len(content_list) > 0
                and isinstance(content_list[0], dict)
                and "text" in content_list[0]
            ):
                text_val = content_list[0]["text"]
                if isinstance(text_val, str) and text_val.strip():
                    return text_val.strip()

        # 5) If none of the above returned text, raise an error
        raise BedrockInvocationError(f"No valid text found in Bedrock response: {parsed}")
Key Points:

Validates that AWS_REGION and BEDROCK_MODEL_ID are set; raises ConfigurationError if missing.
Uses boto3 to create a bedrock-runtime client.
Builds a JSON‐encoded string as the body field.
Invokes invoke_model and reads the raw response["body"] bytes.
Parses JSON and tries multiple extraction paths until it finds non‐empty text.
Raises a BedrockInvocationError if invocation fails or parsing yields no valid text.

app.py
python# app.py

from flask import Flask, request, jsonify, Response
from bedrock_client import BedrockClient
from errors import ConfigurationError, BedrockInvocationError

app = Flask(__name__)

def build_prompt_single_turn(user_input: str) -> str:
    """
    Build a one-off prompt that tells Llama to ignore history and only reply directly:
      System: Respond ONLY with the assistant's direct reply to the user message. Do NOT include any previous conversation or extra commentary.
      User: <user_input>
      Assistant:
    """
    return "\n".join([
        "System: Respond ONLY with the assistant's direct reply to the user message. "
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
Key Points:

GET / → returns an HTML page containing:

A scrollable chat box (#chat-box)
A text input (#user-input) and "Send" button (#send-btn)
Inline JavaScript that appends user/assistant messages and POSTs to /chat


POST /chat → same logic as in the CLI version, but communicates JSON back to the browser
The server listens on 0.0.0.0:8080 to satisfy App Runner's requirements.

memory.py and errors.py
Both files remain unchanged from their original simple implementations.
memory.py
python# memory.py

class Memory:
    """
    Simple slot-based memory for "My name is X" or "My favorite color is Y".
    parse_and_store(user_input) updates slots accordingly.
    to_prompt() returns a brief string like:
      "You are talking to Alice. The user likes blue."
    """

    def __init__(self):
        self.slots = {}

    def get(self, key, default=None):
        return self.slots.get(key, default)

    def set(self, key, value):
        self.slots[key] = value

    def to_prompt(self) -> str:
        parts = []
        if "user_name" in self.slots:
            parts.append(f"You are talking to {self.slots['user_name']}.")
        if "favorite_color" in self.slots:
            parts.append(f"The user likes {self.slots['favorite_color']}.")
        return " ".join(parts)

    def parse_and_store(self, user_input: str):
        text = user_input.strip()
        lower = text.lower()

        if lower.startswith("my name is "):
            remainder = text[len("my name is "):].strip()
            name = remainder.split()[0]
            self.set("user_name", name.title())
        elif "my favorite color is " in lower:
            idx = lower.index("my favorite color is ") + len("my favorite color is ")
            remainder = text[idx:].strip()
            color = remainder.split()[0]
            self.set("favorite_color", color.lower())
errors.py
python# errors.py

class ConfigurationError(Exception):
    """
    Raised when required configuration is missing, such as AWS_REGION or BEDROCK_MODEL_ID.
    """
    pass

class BedrockInvocationError(Exception):
    """
    Raised when the Bedrock invoke_model call fails or when response parsing fails.
    """
    def __init__(self, message, original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception
requirements.txt
boto3
flask
pytest
pytest-mock

boto3: AWS SDK for Python (used by bedrock_client.py).
flask: Lightweight web framework (used by app.py).
pytest, pytest-mock: Testing frameworks (used in tests/).

Environment Variables
Before running locally or in App Runner, these environment variables MUST be set:
bashexport AWS_REGION="us-east-2"
export BEDROCK_MODEL_ID="meta.llama3-3-70b-instruct-v1:0"

AWS_REGION: The Bedrock region where your Llama model is available (e.g., us-east-2).
BEDROCK_MODEL_ID: The on‐demand model ID for Llama 3.3 70B Instruct. No provisioned throughput is required for this model.

Local Setup and Run

Clone the repository (or unzip your project folder) so that you see the files listed above.
Create and activate a Python virtual environment:
bashcd conversational-ai-assistant
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# On Windows (PowerShell):
# venv\Scripts\Activate.ps1

Install dependencies:
bashpip install --upgrade pip
pip install -r requirements.txt

Export environment variables:
bashexport AWS_REGION="us-east-2"
export BEDROCK_MODEL_ID="meta.llama3-3-70b-instruct-v1:0"

Run the Flask app:
bashpython app.py
The server listens on http://0.0.0.0:8080.
Open your browser and navigate to http://localhost:8080/.

You will see a minimal chat UI.
Type "Hello" and click Send (or press Enter).
The assistant's reply will appear immediately below your message.


Optional: Test via cURL:
bashcurl -X POST \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello"}' \
  http://localhost:8080/chat
Expected response:
json{
  "reply": "Hello! How can I assist you today?"
}


Deployment on AWS App Runner
1. Prerequisites

A GitHub repository containing all project files (app.py, bedrock_client.py, etc.) with main as the default branch.
An IAM role (for example, Assignment) that:

Trusts apprunner.amazonaws.com (so App Runner can assume it).
Has a policy (e.g., AmazonBedrockFullAccess) that allows bedrock-runtime:InvokeModel.
If you need a sample trust policy, see the Appendix.



2. Create or Update the App Runner Service

Open the AWS Console → App Runner → Create service.
Step 1: Source and deployment

Source: "Source code repository" → GitHub.
Repository: select your conversational-ai-assistant repo.
Branch: main
Source directory: / (root)
Deployment trigger: choose Automatic (every push to main deploys).
Click Next.


Step 2: Configure build

Configuration source: "Configure manually"
Runtime: choose Python 3.10 (or the latest available).
Build command:
bashpip install -r requirements.txt

Start command:
bashpython app.py

Port: 8080
Click Next.


Step 3: Configure service

Service name: e.g. llama-3-3-70b-chat.
CPU & memory: leave defaults (1 vCPU, 2 GB).
Environment variables:

AWS_REGION = us-east-2
BEDROCK_MODEL_ID = meta.llama3-3-70b-instruct-v1:0


Permissions:

Under Instance role, click the dropdown and select your IAM role (e.g. Assignment).
Ensure that role has bedrock-runtime:InvokeModel permission and trusts apprunner.amazonaws.com.


Click Next.


Step 4: Review & create

Verify all settings: source, build/start commands, environment variables, instance role.
Click Create & deploy.



App Runner will now pull the code from GitHub, run pip install -r requirements.txt, and launch python app.py. Wait 1–2 minutes for it to show "Running".
3. Test the Deployed Service

In the App Runner console, copy the Public endpoint (e.g. https://xyz123abcdef.us-east-2.awsapprunner.com).
Open a browser and visit:
https://xyz123abcdef.us-east-2.awsapprunner.com/
You will see the same chat UI.
Type "Hello" and click Send. You should see:
Assistant: Hello, how can I assist you today?

Alternatively, test with cURL:
bashcurl -X POST \
  -H "Content-Type: application/json" \
  -d '{"message":"Tell me a joke"}' \
  https://xyz123abcdef.us-east-2.awsapprunner.com/chat
Expected JSON response:
json{
  "reply": "Why did the llama cross the road? To prove he wasn't a chicken!"
}


Testing and Coverage
1. Test Files Overview
tests/test_bedrock_client.py
Unit tests for BedrockClient.invoke(), ensuring each JSON format path is exercised:

Missing environment variables → ConfigurationError
Response with "generation" → correct extraction
Response with "completion" → correct extraction
Response with "text" → correct extraction
Response with "choices" → correct extraction
Response with "messages" → correct extraction
Response missing all valid fields → BedrockInvocationError

tests/test_flask_endpoints.py
Tests for app.py endpoints using Flask's test client:

GET / returns HTTP 200 and contains expected HTML elements
POST /chat with valid JSON → returns {"reply":"fake reply"}
POST /chat with invalid content‐type or missing/empty "message" → returns HTTP 400

2. Running Tests with Coverage

Install testing dependencies (if not already):
bashpip install pytest pytest-mock coverage

Run pytest with coverage:
bashcoverage run -m pytest
coverage report --omit="*/venv/*"

Interpret the report:
Ensure that total coverage is ≥ 80%. For example:
Name                          Stmts   Miss  Cover
-------------------------------------------------
bedrock_client.py               50      5    90%
app.py                          80     10    88%
tests/test_bedrock_client.py    50      0   100%
tests/test_flask_endpoints.py   30      0   100%
-------------------------------------------------
TOTAL                         210     15    92%


If coverage falls below 80%, add additional tests to cover missing branches (e.g., edge cases in build_prompt or error paths in app.py).
Appendix: Sample IAM Trust Policy
If you need a reminder on how to configure your IAM role (e.g. Assignment) so that App Runner can assume it, here is a minimal trust policy:
json{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "apprunner.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
Attach this trust policy to your role under IAM → Roles → {YourRole} → Trust relationships → Edit trust relationship. Then attach the managed policy AmazonBedrockFullAccess (which includes bedrock-runtime:InvokeModel) under Permissions.
Summary

Core code resides in bedrock_client.py (API wrapper) and app.py (Flask UI).
Local run requires Python 3.10+, boto3, flask, and environment variables AWS_REGION and BEDROCK_MODEL_ID.
Deployment on AWS App Runner:

Build: pip install -r requirements.txt
Start: python app.py
Port: 8080
Instance role: IAM role trusting apprunner.amazonaws.com with bedrock-runtime:InvokeModel.


Testing uses pytest and coverage to ensure all parsing branches and endpoints are validated.

With this documentation, a developer or reviewer can understand the application's architecture, run it locally, deploy it to App Runner, and verify functionality and test coverage.