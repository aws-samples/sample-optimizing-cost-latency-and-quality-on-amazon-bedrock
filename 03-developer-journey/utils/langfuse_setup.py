"""Langfuse setup for the existing Journey agents (one client per process)."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache


def current_session_id(payload=None):
    from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

    return BedrockAgentCoreContext.get_session_id() or (payload or {}).get("session_id", "local")


def _usage_patch(*, params):
    """Price model calls once; Strands input totals include cached tokens."""
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    for key, span in params.spans.items():
        attrs = span.attributes
        if span.instrumentation_scope_name != "strands.telemetry.tracer":
            continue
        operation = attrs.get("gen_ai.operation.name")
        usage_keys = tuple(k for k in attrs if k.startswith("gen_ai.usage."))
        if operation == "invoke_agent":
            patches[key] = OtelSpanPatch(
                delete_attributes=usage_keys
                + tuple(k for k in ("gen_ai.request.model", "gen_ai.response.model") if k in attrs),
                set_attributes={"langfuse.observation.type": "span"},
            )
        elif operation == "chat":

            def count(*names, attrs=attrs):
                value = next((attrs[n] for n in names if n in attrs), 0)
                if type(value) is not int or value < 0:
                    raise ValueError("Missing/invalid token counter")
                return value

            try:
                read = count("gen_ai.usage.cache_read_input_tokens", "gen_ai.usage.cache_read.input_tokens")
                write = count("gen_ai.usage.cache_write_input_tokens", "gen_ai.usage.cache_creation.input_tokens")
                total = count("gen_ai.usage.input_tokens")
                output = count("gen_ai.usage.output_tokens")
                if (
                    total < read + write
                    or "gen_ai.usage.input_tokens" not in attrs
                    or "gen_ai.usage.output_tokens" not in attrs
                ):
                    raise ValueError("Inconsistent token counters")
                # All checkpoints in the unchanged Journey use the default 5m TTL.
                details = {
                    "input": total - read - write,
                    "output": output,
                    "cache_read_input_tokens": read,
                    "cache_write_5m_input_tokens": write,
                }
                patches[key] = OtelSpanPatch(
                    delete_attributes=usage_keys,
                    set_attributes={
                        "langfuse.observation.type": "generation",
                        "langfuse.observation.model.name": attrs["gen_ai.request.model"],
                        "langfuse.observation.usage_details": json.dumps(details),
                    },
                )
            except (KeyError, ValueError):
                patches[key] = OtelSpanPatch(
                    delete_attributes=usage_keys
                    + tuple(k for k in ("gen_ai.request.model", "gen_ai.response.model") if k in attrs),
                    set_attributes={
                        "langfuse.observation.type": "span",
                        "langfuse.observation.metadata.usage_error": True,
                    },
                )
    for key, span in params.spans.items():
        environment = getattr(span, "resource_attributes", {}).get("deployment.environment.name")
        if (
            "langfuse.environment" in span.attributes
            or not isinstance(environment, str)
            or not environment.startswith("bedrock-agentcore:")
        ):
            continue
        previous = patches.get(key, OtelSpanPatch())
        patches[key] = OtelSpanPatch(
            delete_attributes=previous.delete_attributes,
            set_attributes={
                **previous.set_attributes,
                "langfuse.environment": "bedrock-agentcore-"
                + re.sub(r"[^a-z0-9_-]", "-", environment.split(":", 1)[1].lower()),
            },
        )
    return MaskOtelSpansResult(span_patches=patches)


def _export_journey_span(span):
    """Keep the original routing and guardrail spans alongside model calls."""
    from langfuse import is_default_export_span

    scope = getattr(span.instrumentation_scope, "name", None)
    return is_default_export_span(span) or scope in {
        "customer-support-v4-routing",
        "customer-support-v5-guardrails",
    }


@lru_cache(maxsize=1)
def setup_langfuse():
    """Keep the Journey's Langfuse-only destination and use the supported SDK exporter."""
    from langfuse import Langfuse

    if not all(os.environ.get(k) for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")):
        raise ValueError("Configure the workshop Langfuse project keys before running the Journey")
    # Self-hosted URLs need not contain "langfuse"; retain content as span attributes.
    modes = [v.strip() for v in os.environ.get("OTEL_SEMCONV_STABILITY_OPT_IN", "").split(",") if v.strip()]
    if "gen_ai_span_attributes_only" not in modes:
        modes.append("gen_ai_span_attributes_only")
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = ",".join(modes)
    return Langfuse(mask_otel_spans=_usage_patch, should_export_span=_export_journey_span)
