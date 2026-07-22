from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import boto3
import requests
from botocore.exceptions import ClientError
from langfuse import get_client, observe
from langfuse.model import PromptClient

langfuse = get_client()

MODEL_CONFIG = {
    "sonnet": {
        "model_id": "global.anthropic.claude-sonnet-4-6",
        "inferenceConfig": {"maxTokens": 4096, "temperature": 0},
    },
    "haiku": {
        "model_id": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "inferenceConfig": {"maxTokens": 2048, "temperature": 0},
    },
}

GUARDRAIL_CONFIG = {
    "guardrailIdentifier": "<guardrailid>",  # TODO: Fill with your GuardrailId
    "guardrailVersion": "1",
    "trace": "enabled",
}

# used to invoke the Bedrock Converse API
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)


def convert_to_bedrock_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Convert messages to Bedrock Converse API format."""
    bedrock_messages = []
    system_prompts = []

    for msg in messages:
        if msg["role"] == "system":
            system_prompts.append({"text": msg["content"]})
        else:
            content_list = []
            if isinstance(msg["content"], list):
                for content_item in msg["content"]:
                    if content_item["type"] == "text":
                        content_list.append({"text": content_item["text"]})
                    elif content_item["type"] == "image_url":
                        if "url" not in content_item["image_url"]:
                            raise ValueError("Missing required 'url' field in image_url")
                        url = content_item["image_url"]["url"]
                        if not url:
                            raise ValueError("URL cannot be empty")
                        parsed_url = urlparse(url)
                        if not parsed_url.scheme or not parsed_url.netloc:
                            raise ValueError("Invalid URL format")
                        image_format = parsed_url.path.split(".")[-1].lower()
                        if image_format == "jpg":
                            image_format = "jpeg"
                        response = requests.get(url)
                        image_bytes = response.content
                        content_list.append(
                            {
                                "image": {
                                    "format": image_format,
                                    "source": {"bytes": image_bytes},
                                }
                            }
                        )
            else:
                content_list.append({"text": msg["content"]})
            bedrock_messages.append({"role": msg["role"], "content": content_list})

    return system_prompts, bedrock_messages


@observe(as_type="generation", name="Bedrock Converse")
def converse(
    messages: list[dict[str, Any]],
    model_id: str = "global.anthropic.claude-sonnet-4-6",
    prompt: PromptClient | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> str | None:
    if metadata is None:
        metadata = {}
    kwargs_clone = kwargs.copy()
    model_parameters = {
        **kwargs_clone.pop("inferenceConfig", {}),
        **kwargs_clone.pop("additionalModelRequestFields", {}),
        **kwargs_clone.pop("guardrailConfig", {}),
    }
    langfuse.update_current_generation(
        input=messages,
        model=model_id,
        model_parameters=model_parameters,
        prompt=prompt,
    )

    system_prompts, messages = convert_to_bedrock_messages(messages)

    try:
        response = bedrock_runtime.converse(
            modelId=model_id,
            system=system_prompts,
            messages=messages,
            **kwargs,
        )
    except (ClientError, Exception) as e:
        error_message = f"ERROR: Can't invoke '{model_id}'. Reason: {e}"
        langfuse.update_current_generation(level="ERROR", status_message=error_message)
        print(error_message)
        return

    response_text = response["output"]["message"]["content"][0]["text"]
    langfuse.update_current_generation(
        output=response_text,
        usage_details={
            "input": response["usage"]["inputTokens"],
            "output": response["usage"]["outputTokens"],
            "total": response["usage"]["totalTokens"],
        },
        metadata={
            "ResponseMetadata": response["ResponseMetadata"],
            **metadata,
        },
    )

    return response_text


@observe(as_type="generation", name="Bedrock Converse Tool Use")
def converse_tool_use(
    messages: list[dict[str, str]],
    tools: list[dict[str, str]],
    tool_choice: str = "auto",
    model_id: str = "global.anthropic.claude-sonnet-4-6",
    prompt: PromptClient | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> list[dict] | None:
    if metadata is None:
        metadata = {}
    kwargs_clone = kwargs.copy()
    model_parameters = {
        **kwargs_clone.pop("inferenceConfig", {}),
        **kwargs_clone.pop("additionalModelRequestFields", {}),
        **kwargs_clone.pop("guardrailConfig", {}),
    }

    langfuse.update_current_generation(
        input={"messages": messages, "tools": tools, "tool_choice": tool_choice},
        model=model_id,
        model_parameters=model_parameters,
        prompt=prompt,
    )

    system_prompts, messages = convert_to_bedrock_messages(messages)

    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "inputSchema": {"json": tool["function"]["parameters"]},
                }
            }
            for tool in tools
            if tool["type"] == "function"
        ]
    }

    if tool_choice != "auto":
        tool_config["toolChoice"] = {
            "any": {} if tool_choice == "any" else None,
            "auto": {} if tool_choice == "auto" else None,
            "tool": ({"name": tool_choice} if tool_choice not in ["any", "auto"] else None),
        }

    try:
        response = bedrock_runtime.converse(
            modelId=model_id,
            system=system_prompts,
            messages=messages,
            toolConfig=tool_config,
            **kwargs,
        )
    except (ClientError, Exception) as e:
        error_message = f"ERROR: Can't invoke '{model_id}'. Reason: {e}"
        langfuse.update_current_generation(level="ERROR", status_message=error_message)
        print(error_message)
        return

    output_message = response["output"]["message"]

    tool_calls = []
    if response["stopReason"] == "tool_use":
        for content in output_message["content"]:
            if "toolUse" in content:
                tool = content["toolUse"]
                tool_calls.append(
                    {
                        "index": len(tool_calls),
                        "id": tool["toolUseId"],
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "arguments": json.dumps(tool["input"]),
                        },
                    }
                )

    langfuse.update_current_generation(
        output=tool_calls,
        usage_details={
            "input": response["usage"]["inputTokens"],
            "output": response["usage"]["outputTokens"],
            "total": response["usage"]["totalTokens"],
        },
        metadata={
            "ResponseMetadata": response["ResponseMetadata"],
            **metadata,
        },
    )

    for tc in tool_calls:
        _execute_tool_span(
            tool_id=tc["id"],
            tool_name=tc["function"]["name"],
            tool_input=tc["function"]["arguments"],
        )

    return tool_calls


@observe(as_type="span", name="Tool Execution")
def _execute_tool_span(tool_id: str, tool_name: str, tool_input: str) -> dict:
    langfuse.update_current_span(
        input={"tool_id": tool_id, "tool_name": tool_name, "arguments": tool_input},
        output={"status": "completed", "result": json.loads(tool_input)},
        metadata={"tool_name": tool_name},
    )
    return {"tool_id": tool_id, "status": "executed"}
