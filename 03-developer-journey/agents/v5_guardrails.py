"""
V5 Guardrails Agent - Bedrock Guardrails for content filtering.
- Front gate: screen the raw query with ApplyGuardrail BEFORE building the agent.
  A hard block (content filter / denied topic / blocked PII) returns immediately
  with no model call -> 0 LLM input tokens for that request.
- Defense in depth: the guardrail is ALSO attached to the BedrockModel, so output is
  screened and PII is masked on queries that pass the gate.
- Enforce scope and safety independently of the system prompt.
"""

from __future__ import annotations

import json
import os

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from opentelemetry import trace
from strands import Agent
from strands.models import BedrockModel
from strands.types.content import SystemContentBlock

from utils.agent_config import (
    MODEL_HAIKU,
    MODEL_SONNET,
    SYSTEM_PROMPT_TEXT,
    assistant_text,
    classify_query_complexity,
    model_settings,
    setup_langfuse_telemetry,
)
from utils.langfuse_setup import current_session_id
from utils.tools import get_product_info, get_return_policy, get_technical_support, web_search

setup_langfuse_telemetry()

app = BedrockAgentCoreApp()

GUARDRAIL_ID = os.environ.get("GUARDRAIL_ID")
GUARDRAIL_VERSION = os.environ.get("GUARDRAIL_VERSION", "DRAFT")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# Message returned when the front gate hard-blocks a query (mirrors the guardrail's
# configured blockedInputMessaging).
BLOCK_MESSAGE = (
    "I'm unable to process your request. This may be due to a topic outside my scope "
    "or content that violates our usage policy. Please ask about TechMart products, "
    "returns, or technical support."
)

SYSTEM_PROMPT = [
    SystemContentBlock(text=SYSTEM_PROMPT_TEXT),
    SystemContentBlock(cachePoint={"type": "default"}),
]

_bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)


def guardrail_blocks_input(query: str) -> tuple[bool, list[str], str]:
    """Screen a raw query with ApplyGuardrail without invoking any model.

    Returns (is_hard_block, policies_fired, model_input). A hard block is a content filter or
    denied-topic intervention, or PII with action BLOCKED. PII that is only masked
    (ANONYMIZED) is NOT a hard block. Pass the returned masked text to the model.
    """
    if not GUARDRAIL_ID:
        return False, [], query

    resp = _bedrock_runtime.apply_guardrail(
        guardrailIdentifier=GUARDRAIL_ID,
        guardrailVersion=GUARDRAIL_VERSION,
        source="INPUT",
        content=[{"text": {"text": query}}],
    )
    if resp.get("action") != "GUARDRAIL_INTERVENED":
        return False, [], query

    fired: list[str] = []
    hard_block = False
    anonymized = False
    for assessment in resp.get("assessments", []):
        for topic in assessment.get("topicPolicy", {}).get("topics", []):
            fired.append(f"topic:{topic['name']}")
            hard_block = True
        for filt in assessment.get("contentPolicy", {}).get("filters", []):
            fired.append(f"content:{filt['type']}")
            hard_block = True
        for pii in assessment.get("sensitiveInformationPolicy", {}).get("piiEntities", []):
            fired.append(f"pii:{pii['type']}:{pii['action']}")
            if pii.get("action") == "BLOCKED":
                hard_block = True
            elif pii.get("action") == "ANONYMIZED":
                anonymized = True
    model_input = query
    if anonymized and not hard_block:
        model_input = "\n".join(item["text"] for item in resp.get("outputs", []) if item.get("text"))
        if not model_input:
            raise ValueError("Guardrail reported masking without returning the masked input")
    return hard_block, fired, model_input


@app.entrypoint
def invoke(payload):
    user_input = payload.get("prompt", "")
    session_id = current_session_id(payload)

    telemetry = setup_langfuse_telemetry()

    tracer = trace.get_tracer("customer-support-v5-guardrails")

    # One parent span for the whole request. The front gate runs as a child span so
    # its latency is recorded and rolls up into the overall trace duration -- whether
    # the query is blocked (latency = gate time) or passed (gate time + agent time).
    with tracer.start_as_current_span(
        "customer-support-v5-guardrails",
        attributes={
            "gen_ai.system": "strands",
            "session.id": session_id,
            "gen_ai.agent.name": "customer-support-v5-guardrails",
            "version": "v5-guardrails",
            "guardrails_enabled": bool(GUARDRAIL_ID),
            "langfuse.observation.input": json.dumps({"prompt": user_input}),
        },
    ) as parent_span:
        # Front gate (child span): ApplyGuardrail screen, no model call.
        with tracer.start_as_current_span("guardrail-input-check") as gate_span:
            blocked, policies, model_input = guardrail_blocks_input(user_input)
            gate_span.set_attribute("guardrail.blocked", blocked)
            gate_span.set_attribute("guardrail.blocked_by", ", ".join(policies))

        if blocked:
            # Blocked: no agent, 0 LLM tokens. The parent span's duration is the gate
            # time, and this trace carries the block details into Langfuse.
            parent_span.set_attribute("guardrail.blocked", True)
            parent_span.set_attribute("guardrail.blocked_by", ", ".join(policies))
            parent_span.set_attribute("langfuse.tags", ["guardrails", "guardrail-blocked"])
            parent_span.set_attribute("langfuse.observation.output", json.dumps({"response": BLOCK_MESSAGE}))
            telemetry.flush()
            return {
                "response": BLOCK_MESSAGE,
                "guardrails_enabled": True,
                "blocked": True,
                "blocked_by": policies,
            }

        parent_span.set_attribute("langfuse.observation.input", json.dumps({"prompt": model_input}))
        complexity = classify_query_complexity(user_input)
        model_id = MODEL_HAIKU if complexity == "simple" else MODEL_SONNET
        parent_span.set_attribute("query_complexity", complexity)
        parent_span.set_attribute("langfuse.tags", ["guardrails", complexity])

        model_kwargs = {
            "model_id": model_id,
            **model_settings(model_id, 0.1),
            "max_tokens": 1024,
            "cache_tools": "default",
            "region_name": REGION,
        }

        # Keep the guardrail on the model too: screens OUTPUT and masks PII on queries
        # that pass the front gate (defense in depth).
        if GUARDRAIL_ID:
            model_kwargs["guardrail_id"] = GUARDRAIL_ID
            model_kwargs["guardrail_version"] = GUARDRAIL_VERSION
            model_kwargs["guardrail_trace"] = "enabled"

        agent = Agent(
            model=BedrockModel(**model_kwargs),
            tools=[get_return_policy, get_product_info, web_search, get_technical_support],
            system_prompt=SYSTEM_PROMPT,
            name="customer-support-v5-guardrails",
            trace_attributes={
                "session.id": session_id,
                "version": "v5-guardrails",
                "query_complexity": complexity,
                "guardrails_enabled": bool(GUARDRAIL_ID),
                "langfuse.tags": ["guardrails", complexity],
            },
        )

        # Agent runs inside the parent span, so the trace combines gate + agent latency.
        response = agent(model_input)
        response_text = assistant_text(response)
        parent_span.set_attribute("langfuse.observation.output", json.dumps({"response": response_text}))

    telemetry.flush()

    return {"response": response_text, "guardrails_enabled": bool(GUARDRAIL_ID), "blocked": False}


if __name__ == "__main__":
    app.run()
