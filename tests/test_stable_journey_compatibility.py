"""Regression checks for compatibility fixes to the published Journey."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load("stable_agent_config", "03-developer-journey/utils/agent_config.py")
telemetry = load("stable_telemetry", "03-developer-journey/utils/langfuse_setup.py")
metrics = load("stable_metrics", "03-developer-journey/utils/langfuse_metrics.py")


@pytest.mark.parametrize("model", ["global.anthropic.claude-sonnet-5", "global.anthropic.claude-opus-5"])
def test_unsupported_temperature_omitted(model):
    assert config.model_settings(model, 0.3) == {"additional_request_fields": {"output_config": {"effort": "low"}}}


def test_supported_temperature_and_text_blocks_preserved():
    assert config.model_settings(config.MODEL_HAIKU, 0.3) == {"temperature": 0.3}
    result = SimpleNamespace(
        message={"content": [{"reasoningContent": {}}, {"text": "first"}, {"toolUse": {}}, {"text": " second"}]}
    )
    assert config.assistant_text(result) == "first second"


def test_langfuse_prices_generation_once_with_cache():
    def span(operation, **kwargs):
        return SimpleNamespace(
            instrumentation_scope_name="strands.telemetry.tracer",
            attributes={
                "gen_ai.operation.name": operation,
                "gen_ai.request.model": config.MODEL_SONNET,
                "gen_ai.usage.input_tokens": 1200,
                "gen_ai.usage.output_tokens": 30,
                "gen_ai.usage.cache_read_input_tokens": 1000,
                "gen_ai.usage.cache_write_input_tokens": 100,
                **kwargs,
            },
        )

    result = telemetry._usage_patch(
        params=SimpleNamespace(spans={"agent": span("invoke_agent"), "call": span("chat")})
    )
    assert result.span_patches["agent"].set_attributes["langfuse.observation.type"] == "span"
    patch = result.span_patches["call"]
    details = json.loads(patch.set_attributes["langfuse.observation.usage_details"])
    assert details == {"input": 100, "output": 30, "cache_read_input_tokens": 1000, "cache_write_5m_input_tokens": 100}
    assert "gen_ai.usage.input_tokens" in patch.delete_attributes


def test_unknown_usage_cannot_be_reported_as_free():
    span = SimpleNamespace(
        instrumentation_scope_name="strands.telemetry.tracer", attributes={"gen_ai.operation.name": "chat"}
    )
    patch = telemetry._usage_patch(params=SimpleNamespace(spans={"call": span})).span_patches["call"]
    assert patch.set_attributes["langfuse.observation.metadata.usage_error"] is True


def test_trace_lookup_requires_invocation_identity():
    with pytest.raises(ValueError, match="session ID"):
        metrics.get_latest_trace_metrics(wait_seconds=0)


def test_trace_lookup_filters_session_and_prices_generations(monkeypatch):
    calls = []

    class TraceApi:
        def list(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(id="trace-current", name="")])

        def get(self, trace_id, **kwargs):
            assert trace_id == "trace-current"
            observation = SimpleNamespace(
                type="GENERATION",
                dict=lambda: {
                    "usageDetails": {
                        "input": 10,
                        "output": 20,
                        "cache_read_input_tokens": 100,
                        "cache_write_5m_input_tokens": 50,
                    },
                    "calculatedTotalCost": 0.005,
                },
            )
            return SimpleNamespace(
                id=trace_id,
                name="",
                observations=[SimpleNamespace(type="SPAN", name="invoke_agent customer-support-v1-baseline"), observation],
                latency=1.25,
                html_path="/project/test/traces/" + trace_id,
            )

    monkeypatch.setattr(metrics, "trace_reader", lambda: SimpleNamespace(trace=TraceApi()))
    result = metrics.get_latest_trace_metrics(wait_seconds=0, session_id="this-invocation")
    assert calls[0]["session_id"] == "this-invocation"
    assert result["total_tokens"] == 180
    assert result["model_calls"] == 1 and result["cost_usd"] == 0.005
    assert result["trace_id"] == "trace-current"


def test_runtime_environment_normalized_only_in_langfuse_copy():
    resource = {"deployment.environment.name": "bedrock-agentcore:default"}
    span = SimpleNamespace(
        instrumentation_scope_name="strands.telemetry.tracer",
        attributes={"gen_ai.operation.name": "invoke_agent"},
        resource_attributes=resource,
    )
    patch = telemetry._usage_patch(params=SimpleNamespace(spans={"call": span})).span_patches["call"]
    assert patch.set_attributes["langfuse.environment"] == "bedrock-agentcore-default"
    assert resource == {"deployment.environment.name": "bedrock-agentcore:default"}


def test_original_routing_and_guardrail_spans_are_exported():
    for scope in ("customer-support-v4-routing", "customer-support-v5-guardrails"):
        span = SimpleNamespace(instrumentation_scope=SimpleNamespace(name=scope), attributes={})
        assert telemetry._export_journey_span(span)
    unrelated = SimpleNamespace(instrumentation_scope=SimpleNamespace(name="unrelated-http-client"), attributes={})
    assert not telemetry._export_journey_span(unrelated)


def test_runtime_lookup_reads_all_pages_and_rejects_ambiguity():
    import ast

    source = (ROOT / "03-developer-journey/utils/runtime_helpers.py").read_text()
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "find_agent_by_name")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "runtime_lookup", "exec"), ns)
    calls = []

    def pages(**kwargs):
        calls.append(kwargs)
        if not kwargs:
            return {
                "agentRuntimes": [{"agentRuntimeName": "unrelated", "agentRuntimeArn": "other"}],
                "nextToken": "second",
            }
        return {"agentRuntimes": [{"agentRuntimeName": "customer_support_v6_gateway", "agentRuntimeArn": "wanted"}]}

    client = SimpleNamespace(list_agent_runtimes=pages)
    assert ns["find_agent_by_name"](client, "v6-gateway") == "wanted"
    assert calls == [{}, {"nextToken": "second"}]
    client.list_agent_runtimes = lambda **kwargs: {
        "agentRuntimes": [{"agentRuntimeName": "v1-baseline-" + str(i), "agentRuntimeArn": str(i)} for i in range(2)]
    }
    with pytest.raises(ValueError, match="Multiple Runtimes"):
        ns["find_agent_by_name"](client, "v1-baseline")
