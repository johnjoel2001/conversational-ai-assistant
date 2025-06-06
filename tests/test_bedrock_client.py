# tests/test_bedrock_client.py

import os
import json
import pytest
from bedrock_client import BedrockClient
from errors import ConfigurationError, BedrockInvocationError

class DummyResponse:
    def __init__(self, body_bytes):
        self.body = DummyBody(body_bytes)

class DummyBody:
    def __init__(self, data):
        self._data = data
    def read(self):
        return self._data

class DummyClient:
    def __init__(self, response_payload):
        # response_payload: a dict that will be JSON‐encoded
        self.payload = response_payload

    def invoke_model(self, **kwargs):
        # Always return a dummy response with JSON‐encoded self.payload
        body_bytes = json.dumps(self.payload).encode("utf-8")
        return DummyResponse(body_bytes)

@pytest.fixture(autouse=True)
def no_aws_env(monkeypatch):
    # Ensure AWS_REGION or BEDROCK_MODEL_ID not set by default
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)

def test_missing_env_vars():
    # Neither AWS_REGION nor BEDROCK_MODEL_ID → ConfigurationError
    with pytest.raises(ConfigurationError):
        BedrockClient()

def test_invoke_generation(monkeypatch):
    # Simulate a response with {"generation": "Hello world"}
    os.environ["AWS_REGION"] = "us-east-2"
    os.environ["BEDROCK_MODEL_ID"] = "meta.llama3-3-70b-instruct-v1:0"
    dummy = DummyClient({"generation": " Hello world "})
    monkeypatch.setattr("bedrock_client.boto3.client", lambda *args, **kwargs: dummy)
    bc = BedrockClient()
    result = bc.invoke("prompt")
    assert result == "Hello world"

def test_invoke_completion(monkeypatch):
    # Simulate a response with {"completion": "Bye world"}
    os.environ["AWS_REGION"] = "us-east-2"
    os.environ["BEDROCK_MODEL_ID"] = "meta.llama3-3-70b-instruct-v1:0"
    dummy = DummyClient({"completion": "Bye world"})
    monkeypatch.setattr("bedrock_client.boto3.client", lambda *args, **kwargs: dummy)
    bc = BedrockClient()
    assert bc.invoke("prompt") == "Bye world"

def test_invoke_text(monkeypatch):
    os.environ["AWS_REGION"] = "us-east-2"
    os.environ["BEDROCK_MODEL_ID"] = "meta.llama3-3-70b-instruct-v1:0"
    dummy = DummyClient({"text": "Just text"})
    monkeypatch.setattr("bedrock_client.boto3.client", lambda *args, **kwargs: dummy)
    bc = BedrockClient()
    assert bc.invoke("prompt") == "Just text"

def test_invoke_choices(monkeypatch):
    os.environ["AWS_REGION"] = "us-east-2"
    os.environ["BEDROCK_MODEL_ID"] = "meta.llama3-3-70b-instruct-v1:0"
    payload = {
        "choices": [
            { "message": { "content": [ { "text": "Choice text" } ] } }
        ]
    }
    dummy = DummyClient(payload)
    monkeypatch.setattr("bedrock_client.boto3.client", lambda *args, **kwargs: dummy)
    bc = BedrockClient()
    assert bc.invoke("prompt") == "Choice text"

def test_invoke_messages(monkeypatch):
    os.environ["AWS_REGION"] = "us-east-2"
    os.environ["BEDROCK_MODEL_ID"] = "meta.llama3-3-70b-instruct-v1:0"
    payload = {
        "messages": [
            { "content": [ { "text": "Msg text" } ] }
        ]
    }
    dummy = DummyClient(payload)
    monkeypatch.setattr("bedrock_client.boto3.client", lambda *args, **kwargs: dummy)
    bc = BedrockClient()
    assert bc.invoke("prompt") == "Msg text"

def test_invoke_no_valid_field(monkeypatch):
    # All fields empty or missing → should raise BedrockInvocationError
    os.environ["AWS_REGION"] = "us-east-2"
    os.environ["BEDROCK_MODEL_ID"] = "meta.llama3-3-70b-instruct-v1:0"
    dummy = DummyClient({"nope": "nothing"})
    monkeypatch.setattr("bedrock_client.boto3.client", lambda *args, **kwargs: dummy)
    bc = BedrockClient()
    with pytest.raises(BedrockInvocationError):
        bc.invoke("prompt")
