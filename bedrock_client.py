# bedrock_client.py

import os
import json
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from errors import ConfigurationError, BedrockInvocationError

class BedrockClient:
    """
    A simple wrapper around AWS Bedrock's invoke_model API.

    Expects environment variables:
      - AWS_REGION
      - BEDROCK_MODEL_ID
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

    def invoke(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """
        Send `prompt` to the Bedrock model. Return the model's reply as a string.
        Raises BedrockInvocationError on failure.
        """
        payload = {
            "prompt": prompt,
            "maxTokensToSample": max_tokens,
            "temperature": temperature
        }

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload).encode("utf-8")
            )
        except (BotoCoreError, ClientError) as aws_err:
            raise BedrockInvocationError(f"Failed to invoke Bedrock model: {aws_err}", original_exception=aws_err)

        try:
            raw_bytes = response["body"].read()
            text = raw_bytes.decode("utf-8")
            parsed = json.loads(text)
            completion = parsed.get("completion") or parsed.get("text") or ""
            return completion.strip()
        except Exception as parse_err:
            raise BedrockInvocationError(f"Failed to parse Bedrock response: {parse_err}", original_exception=parse_err)
