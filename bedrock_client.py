# bedrock_client.py

import os
import json
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from errors import ConfigurationError, BedrockInvocationError

class BedrockClient:
    """
    Wrapper around AWS Bedrock's invoke_model API for Llama 3.3 70B Instruct.
    This version first tries "generation" (as seen in on-demand responses),
    then falls back to "completion", "text", "choices", and "messages".
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
        Send a prompt to Llama 3.3 70B Instruct and return the generated text.
        Uses on-demand JSON schema:
          { "prompt": "...", "max_gen_len":512, "temperature":0.5, "top_p":0.9 }
        Then tries, in order:
          1) parsed["generation"]
          2) parsed["completion"]
          3) parsed["text"]
          4) parsed["choices"][0]["message"]["content"][0]["text"]
          5) parsed["messages"][0]["content"][0]["text"]
        """

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

        try:
            response = self.client.invoke_model(**invoke_args)
        except (BotoCoreError, ClientError) as aws_err:
            raise BedrockInvocationError(f"Failed to invoke Bedrock model: {aws_err}", original_exception=aws_err)

        try:
            raw_bytes = response["body"].read()
            decoded = raw_bytes.decode("utf-8")
            parsed = json.loads(decoded)
        except Exception as parse_err:
            raise BedrockInvocationError(f"Failed to parse Bedrock response body: {parse_err}", original_exception=parse_err)

        # 1) Try top-level "generation"
        gen = parsed.get("generation")
        if isinstance(gen, str) and gen.strip():
            return gen.strip()

        # 2) Try top-level "completion"
        comp = parsed.get("completion")
        if isinstance(comp, str) and comp.strip():
            return comp.strip()

        # 3) Try top-level "text"
        txt = parsed.get("text")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        # 4) Try "choices"[0]["message"]["content"][0]["text"]
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
            if isinstance(content_list, list) and len(content_list) > 0:
                first = content_list[0]
                text_val = first.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    return text_val.strip()

        # 5) Try "messages"[0]["content"][0]["text"]
        llm_msgs = parsed.get("messages")
        if (
            isinstance(llm_msgs, list)
            and len(llm_msgs) > 0
            and isinstance(llm_msgs[0], dict)
            and "content" in llm_msgs[0]
        ):
            content_list = llm_msgs[0].get("content", [])
            if isinstance(content_list, list) and len(content_list) > 0:
                first = content_list[0]
                text_val = first.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    return text_val.strip()

        # If none returned non-empty text, throw an error with full parsed output
        raise BedrockInvocationError(f"No valid text found in Bedrock response: {parsed}")
