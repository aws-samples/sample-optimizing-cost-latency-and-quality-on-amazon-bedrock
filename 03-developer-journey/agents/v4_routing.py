"""
V4 Routing Agent - Model routing based on query complexity.
- Same system prompt as v3 with prompt caching
- LLM-based classification using Haiku
- Simple queries -> Haiku (cheaper)
- Complex queries -> Sonnet (better quality)
"""

from __future__ import annotations

import json
import os

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
    model_settings,
    setup_langfuse_telemetry,
)
from utils.langfuse_setup import current_session_id
from utils.tools import get_product_info, get_return_policy, get_technical_support, web_search

setup_langfuse_telemetry()

app = BedrockAgentCoreApp()

# Classifier prompt for LLM-based routing
CLASSIFIER_PROMPT = """Classify customer support queries for model routing.

SIMPLE queries (route to cheaper model):
- Greetings and general questions
- Single factual lookups (price, policy, hours)
- Direct questions with straightforward answers

COMPLEX queries (route to powerful model):
- Multi-step troubleshooting
- Product comparisons or recommendations
- Questions requiring analysis or reasoning
- Multiple questions in one message

Respond with ONLY one word: simple or complex

Examples:

Query: "What is your return policy for laptops?"
simple

Query: "Tell me about your smartphone options"
simple

Query: "Hello! What can you help me with today?"
simple

Query: "My laptop won't turn on, can you help me troubleshoot?"
complex

Query: "I want to buy a laptop. What are the specs and what's the return policy?"
complex"""

# Provider-agnostic caching using SystemContentBlock
SYSTEM_PROMPT = [
    SystemContentBlock(text=SYSTEM_PROMPT_TEXT),
    SystemContentBlock(cachePoint={"type": "default"}),
]


def classify_query_with_llm(query: str, region: str) -> str:
    """Use Haiku to classify query complexity.

    Defaults to "complex" (route to the more capable Sonnet) on any classifier
    failure or unexpected output. This is a deliberate safe-but-costlier default:
    a misrouted-to-cheap query degrades the answer, whereas a misrouted-to-capable
    query only costs more. Note this is a blanket fallback, not retriable-vs-
    non-retriable error handling — production routers should distinguish 429/503
    (retry/reroute) from 400 (fail) rather than collapsing all failures here.
    """
    classifier_model = BedrockModel(
        model_id=MODEL_HAIKU,
        **model_settings(MODEL_HAIKU, 0),
        max_tokens=50,
        region_name=region,
    )
    classifier = Agent(
        model=classifier_model,
        system_prompt=CLASSIFIER_PROMPT,
        name="customer-support-v4-classifier",
        trace_attributes={
            "session.id": current_session_id(),
            "version": "v4-routing",
            "component": "classifier",
            "langfuse.tags": ["routing", "classifier"],
        },
    )
    try:
        result = classifier(query)
        response_text = assistant_text(result).strip().lower()
    except Exception:
        return "complex"

    return "simple" if "simple" in response_text else "complex"


@app.entrypoint
def invoke(payload):
    user_input = payload.get("prompt", "")
    session_id = current_session_id(payload)

    telemetry = setup_langfuse_telemetry()

    tracer = trace.get_tracer("customer-support-v4-routing")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    with tracer.start_as_current_span(
        "customer-support-v4-routing",
        attributes={
            "gen_ai.system": "strands",
            "session.id": session_id,
            "gen_ai.agent.name": "customer-support-v4-routing",
            "version": "v4-routing",
            "langfuse.observation.input": json.dumps({"prompt": user_input}),
        },
    ) as parent_span:
        complexity = classify_query_with_llm(user_input, region)
        model_id = MODEL_HAIKU if complexity == "simple" else MODEL_SONNET

        parent_span.set_attribute("query_complexity", complexity)
        parent_span.set_attribute("model_used", model_id)

        model = BedrockModel(
            model_id=model_id,
            **model_settings(model_id, 0.1),
            max_tokens=1024,
            cache_tools="default",
            region_name=region,
        )

        agent = Agent(
            model=model,
            tools=[get_return_policy, get_product_info, web_search, get_technical_support],
            system_prompt=SYSTEM_PROMPT,
            name="customer-support-v4-main",
            trace_attributes={
                "session.id": session_id,
                "version": "v4-routing",
                "query_complexity": complexity,
                "model_used": model_id,
                "langfuse.tags": ["routing", complexity],
            },
        )

        response = agent(user_input)
        response_text = assistant_text(response)

        parent_span.set_attribute(
            "langfuse.observation.output",
            json.dumps({"response": response_text, "model_used": model_id, "complexity": complexity}),
        )

    telemetry.flush()

    return {"response": response_text, "model_used": model_id, "complexity": complexity}


if __name__ == "__main__":
    app.run()
