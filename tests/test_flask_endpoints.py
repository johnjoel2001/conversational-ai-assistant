# tests/test_flask_endpoints.py

import os
import json
import pytest
from app import app

@pytest.fixture
def client():
    os.environ["AWS_REGION"] = "us-east-2"
    os.environ["BEDROCK_MODEL_ID"] = "meta.llama3-3-70b-instruct-v1:0"
    # Monkey‐patch BedrockClient.invoke to avoid actual AWS calls
    import bedrock_client
    class FakeBC:
        def __init__(self):
            pass
        def invoke(self, prompt, max_gen_len=512, temperature=0.5, top_p=0.9):
            return "fake reply"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(bedrock_client, "BedrockClient", FakeBC)

    with app.test_client() as c:
        yield c
    monkeypatch.undo()

def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "<title>Chat with Llama 3.3 70B Instruct</title>" in html
    assert "id=\"chat-box\"" in html
    assert "id=\"user-input\"" in html

def test_chat_endpoint_success(client):
    data = {"message": "Hello"}
    resp = client.post("/chat", data=json.dumps(data), content_type="application/json")
    assert resp.status_code == 200
    parsed = resp.get_json()
    assert "reply" in parsed
    assert parsed["reply"] == "fake reply"

def test_chat_endpoint_bad_requests(client):
    # Missing JSON body
    resp = client.post("/chat", data="not a json", content_type="text/plain")
    assert resp.status_code == 400

    # JSON without "message"
    resp = client.post("/chat", data=json.dumps({}), content_type="application/json")
    assert resp.status_code == 400

    # Empty message
    data = {"message": "   "}
    resp = client.post("/chat", data=json.dumps(data), content_type="application/json")
    assert resp.status_code == 400
