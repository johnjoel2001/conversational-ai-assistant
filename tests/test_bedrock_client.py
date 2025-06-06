# tests/test_bedrock_client.py

import os
import json
import pytest
import boto3
from botocore.stub import Stubber
from botocore.exceptions import ClientError

from bedrock_client import BedrockClient
from errors import ConfigurationError, BedrockInvocationError

@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "test-model-id")

def test_missing_region(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    with pytest.raises(ConfigurationError):
        BedrockClient()

def test_missing_model(monkeypatch):
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    with pytest.raises(ConfigurationError):
        BedrockClient()

def test_invoke_success(monkeypatch):
    bc = BedrockClient()
    stubber = Stubber(bc.client)

    fake_body = json.dumps({"completion": "Hello from fake Bedrock!"}).encode("utf-8")
    response = {"body": boto3.compat.BytesIO(fake_body)}

    expected_params = {
        "modelId": "test-model-id",
        "contentType": "application/json",
        "accept": "application/json",
        "body": bytes(json.dumps({
            "prompt": "hey",
            "maxTokensToSample": 256,
            "temperature": 0.7
        }), "utf-8")
    }
    stubber.add_response("invoke_model", response, expected_params)
    stubber.activate()

    result = bc.invoke("hey")
    assert result == "Hello from fake Bedrock!"
    stubber.deactivate()

def test_invoke_aws_error(monkeypatch):
    bc = BedrockClient()
    stubber = Stubber(bc.client)
    stubber.add_client_error("invoke_model", service_error_code="InternalFailure", service_message="Oops")
    stubber.activate()

    with pytest.raises(BedrockInvocationError) as excinfo:
        bc.invoke("test")
    assert "Failed to invoke Bedrock model" in str(excinfo.value)
    stubber.deactivate()

def test_invoke_bad_json(monkeypatch):
    bc = BedrockClient()
    stubber = Stubber(bc.client)
    response = {"body": boto3.compat.BytesIO(b"not-a-json")}
    expected_params = {
        "modelId": "test-model-id",
        "contentType": "application/json",
        "accept": "application/json",
        "body": bytes(json.dumps({
            "prompt": "hey",
            "maxTokensToSample": 256,
            "temperature": 0.7
        }), "utf-8")
    }
    stubber.add_response("invoke_model", response, expected_params)
    stubber.activate()

    with pytest.raises(BedrockInvocationError) as excinfo:
        bc.invoke("hey")
    assert "Failed to parse Bedrock response" in str(excinfo.value)
    stubber.deactivate()
