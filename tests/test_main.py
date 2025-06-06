# tests/test_main.py

import pytest
from unittest.mock import patch

import main
from errors import BedrockInvocationError

class DummyBedrockClient:
    def __init__(self):
        pass
    def invoke(self, prompt, max_tokens=256, temperature=0.7):
        return "dummy reply"

class FailingBedrockClient:
    def __init__(self):
        pass
    def invoke(self, prompt, max_tokens=256, temperature=0.7):
        raise BedrockInvocationError("Simulated failure")

@pytest.fixture(autouse=True)
def patch_bedrock(monkeypatch):
    monkeypatch.setattr(main, "BedrockClient", lambda: DummyBedrockClient())

@patch("builtins.input", side_effect=["hello", "exit"])
def test_run_cli_happy_path(mock_input, capsys):
    main.run_cli()
    captured = capsys.readouterr()
    assert "Assistant: dummy reply" in captured.out

@patch("builtins.input", side_effect=["   ", "exit"])
def test_run_cli_skips_empty(mock_input, capsys):
    main.run_cli()
    captured = capsys.readouterr()
    assert "Assistant:" not in captured.out

@patch("builtins.input", side_effect=KeyboardInterrupt())
def test_run_cli_keyboard_interrupt(mock_input, capsys):
    main.run_cli()
    captured = capsys.readouterr()
    assert "Goodbye!" in captured.out

@patch("builtins.input", side_effect=["test error", "exit"])
def test_run_cli_bedrock_fails(mock_input, monkeypatch, capsys):
    monkeypatch.setattr(main, "BedrockClient", lambda: FailingBedrockClient())
    main.run_cli()
    captured = capsys.readouterr()
    assert "[Error] Could not get response from Bedrock" in captured.out
