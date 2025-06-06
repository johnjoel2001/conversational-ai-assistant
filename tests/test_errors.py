# tests/test_errors.py

import pytest
from errors import ConfigurationError, BedrockInvocationError

def test_configuration_error_message():
    msg = "Missing AWS_REGION"
    with pytest.raises(ConfigurationError) as excinfo:
        raise ConfigurationError(msg)
    assert msg in str(excinfo.value)

def test_bedrock_invocation_error_original_exception():
    original = ValueError("Something went wrong at AWS")
    bie = BedrockInvocationError("Failed to call Bedrock", original)
    assert "Failed to call Bedrock" in str(bie)
    assert bie.original_exception == original
