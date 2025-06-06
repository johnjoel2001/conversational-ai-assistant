# embed_client.py

import os
import json
import boto3
from botocore.exceptions import BotoCoreError, ClientError

class EmbedClient:
    """
    Wraps the Titan Embed Text v2 endpoint.
    Expects two environment variables:
      - AWS_REGION
      - BEDROCK_MODEL_ID (e.g. "amazon.titan-embed-text-v2:0")
    """

    def __init__(self):
        region = os.getenv("AWS_REGION")
        model_id = os.getenv("BEDROCK_MODEL_ID")

        if not region:
            raise RuntimeError("Missing AWS_REGION environment variable.")
        if not model_id:
            raise RuntimeError("Missing BEDROCK_MODEL_ID environment variable.")

        self.model_id = model_id
        try:
            self.client = boto3.client("bedrock-runtime", region_name=region)
        except Exception as e:
            raise RuntimeError(f"Failed to create Bedrock client: {e}")

    def embed_text(self, text: str, dimensions: int = 512, normalize: bool = True) -> list[float]:
        """
        Send `text` to Titan Embed Text v2 and return the embedding (list of floats).
        By default, it requests 512 dimensions and applies normalization.
        """
        # Build the JSON body exactly as the API expects:
        payload_body = {
            "inputText": text,
            "dimensions": dimensions,
            "normalize": normalize
        }

        # The API wants "body" as a JSON-string in the outer invocation JSON:
        invoke_args = {
            "modelId": self.model_id,
            "contentType": "application/json",
            "accept": "*/*",
            "body": json.dumps(payload_body).encode("utf-8")
        }

        try:
            response = self.client.invoke_model(**invoke_args)
        except (BotoCoreError, ClientError) as aws_err:
            raise RuntimeError(f"Failed to invoke Bedrock embedding model: {aws_err}")

        # The response body is raw bytes; decode, then parse JSON:
        raw_bytes = response["body"].read()
        text_resp = raw_bytes.decode("utf-8")
        parsed = json.loads(text_resp)

        # Titan embed response returns something like:
        # {
        #   "embedding": [0.123, -0.456, ...] 
        # }
        embedding = parsed.get("embedding")
        if embedding is None:
            raise RuntimeError(f"No 'embedding' field in response: {parsed}")
        return embedding
