# errors.py

class ConfigurationError(Exception):
    """
    Raised when a required configuration (e.g., AWS_REGION or BEDROCK_MODEL_ID) is missing.
    """
    pass

class BedrockInvocationError(Exception):
    """
    Raised when the Bedrock invoke_model call fails (network error, invalid model, etc.).
    """
    def __init__(self, message, original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception
