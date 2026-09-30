"""Offline canonical billing meters and the installed public SDK export hook."""

from __future__ import annotations

import json
from types import MappingProxyType, SimpleNamespace
from uuid import uuid4

import pytest

from workshop_utils.bedrock import NormalizedUsage, normalize_usage
from workshop_utils.observability import (
    CONVERSE_SCOPE,
    langfuse_usage_details,
    normalize_langfuse_spans,
    uniform_cache_ttl,
)

CLAUDE = "global.anthropic.claude-sonnet-5"
GPT = "global.openai.gpt-5.6-sol"
USAGE = "langfuse.observation.usage_details"
STATUS = "langfuse.observation.metadata.usage_status"


def test_mixed_claude_ttls_are_disjoint_and_preserve_all_tokens():
    usage = normalize_usage(
        {
            "inputTokens": 7, "outputTokens": 11, "cacheReadInputTokens": 17, "cacheWriteInputTokens": 42,
            "cacheDetails": [{"ttl": "5m", "inputTokens": 19}, {"ttl": "1h", "inputTokens": 23}],
        },
        source="converse",
    )
    details = langfuse_usage_details(usage, model_id=CLAUDE)
    assert details == {
        "input": 7, "output": 11, "cache_read_input_tokens": 17,
        "cache_write_5m_input_tokens": 19, "cache_write_1h_input_tokens": 23,
    }
    assert sum(details.values()) == usage.total_tokens
    assert sum(value for key, value in details.items() if key != "output") == usage.total_input_tokens


@pytest.mark.parametrize(
    "model,api,writes,expected",
    [
        (CLAUDE, "converse", {}, {"cache_read_input_tokens": 17, "cache_write_unpriced_input_tokens": 42}),
        (CLAUDE, "converse", {"30m": 42}, {"cache_read_input_tokens": 17, "cache_write_unpriced_input_tokens": 42}),
        (GPT, "responses", {}, {"cache_read_30m_input_tokens": 17, "cache_write_30m_input_tokens": 42}),
        (GPT, "responses", {"30m": 42}, {"cache_read_30m_input_tokens": 17, "cache_write_30m_input_tokens": 42}),
        (GPT, "responses", {"5m": 42}, {"cache_read_30m_input_tokens": 17, "cache_write_unpriced_input_tokens": 42}),
        (GPT, "converse", {"30m": 42}, {"cache_read_unpriced_input_tokens": 17, "cache_write_unpriced_input_tokens": 42}),
        ("anthropic.claude-future", "converse", {"5m": 42},
         {"cache_read_unpriced_input_tokens": 17, "cache_write_unpriced_input_tokens": 42}),
        ("openai.gpt-future", "responses", {"30m": 42},
         {"cache_read_unpriced_input_tokens": 17, "cache_write_unpriced_input_tokens": 42}),
        (CLAUDE, "unknown-api", {"1h": 42},
         {"cache_read_unpriced_input_tokens": 17, "cache_write_unpriced_input_tokens": 42}),
    ],
)
def test_known_api_semantics_or_unpriced_counts(model, api, writes, expected):
    usage = NormalizedUsage(7, 11, 17, 42, writes)
    original = usage.as_dict()
    details = langfuse_usage_details(usage, model_id=model, api=api)
    assert details == {"input": 7, "output": 11, **expected}
    assert sum(details.values()) == usage.total_tokens
    assert usage.as_dict() == original
    details["input"] = 99
    assert usage.as_dict() == original


def test_zero_usage_does_not_create_phantom_cache_meters():
    assert langfuse_usage_details(NormalizedUsage(0, 0, 0, 0, {"5m": 0}), model_id=CLAUDE) == {
        "input": 0, "output": 0,
    }


@pytest.mark.parametrize(
    "payload,config,expected",
    [
        ({}, None, None),
        ({"system": [{"cachePoint": {"type": "default"}}]}, None, "5m"),
        ({"messages": [{"content": [{"cachePoint": {"type": "default", "ttl": "1h"}}]}]}, None, "1h"),
        ({"toolConfig": {"tools": [{"cachePoint": {"type": "default", "ttl": "1h"}}]}}, None, "1h"),
        ({"system": [{"cachePoint": {"type": "default", "ttl": "1h"}}, {"cachePoint": {"type": "default"}}]},
         None, None),
        ({"system": [{"cachePoint": {"type": "default", "ttl": "unknown"}}]}, None, None),
        ({"system": [{"cachePoint": {"type": "default", "ttl": None}}]}, None, None),
        ({"system": [{"cachePoint": {"type": "future-type"}}]}, None, None),
        ({"toolConfig": {"tools": [{"toolSpec": {"inputSchema": {"json": {"cachePoint": {"ttl": "1h"}}}}}]}},
         None, None),
        ({}, {"strategy": "auto"}, "5m"),
        ({}, {"ttl": "1h", "system_prompt_ttl": True, "tools_ttl": True}, "1h"),
        ({}, {"ttl": "5m", "system_prompt_ttl": "1h"}, None),
        ({}, {"ttl": "1h", "tools_ttl": "5m"}, None),
        ({}, SimpleNamespace(ttl="1h", system_prompt_ttl=True), "1h"),
    ],
)
def test_uniform_request_ttl_without_guessing_mixed_or_schema_fields(payload, config, expected):
    before = json.dumps(payload, sort_keys=True)
    assert uniform_cache_ttl(payload, config) == expected
    assert json.dumps(payload, sort_keys=True) == before


@pytest.mark.parametrize(
    "config_fields,points,expected",
    [
        ({}, [], "5m"),
        ({"ttl": "1h"}, [], "1h"),
        ({"system_prompt_ttl": False, "tools_ttl": False}, [], None),
        ({"ttl": "1h", "system_prompt_ttl": False, "tools_ttl": False}, [], None),
        ({"system_prompt_ttl": False, "tools_ttl": False}, ["1h"], "1h"),
        ({"system_prompt_ttl": False, "tools_ttl": False}, ["1h", "5m"], None),
        ({"system_prompt_ttl": False, "tools_ttl": True}, [], "5m"),
        ({"ttl": "1h", "system_prompt_ttl": False, "tools_ttl": True}, [], "1h"),
        ({"ttl": "1h", "system_prompt_ttl": "5m"}, [], None),
        ({"system_prompt_ttl": False}, [], None),
    ],
)
def test_real_cache_config_defaults_overrides_and_disabled_sections(config_fields, points, expected):
    pytest.importorskip("strands")
    from strands.models.model import CacheConfig

    config = CacheConfig(**config_fields)
    original = vars(config).copy()
    payload = {"system": [{"cachePoint": {"type": "default", "ttl": ttl}} for ttl in points]}
    assert uniform_cache_ttl(payload, config) == expected
    assert vars(config) == original


def batch_span(attributes, *, scope="strands.telemetry.tracer", name="chat", span_id=1):
    types = pytest.importorskip("langfuse.types")
    identifier = types.OtelSpanIdentifier(trace_id="a" * 32, span_id=f"{span_id:016x}")
    span = types.OtelSpanData(
        trace_id=identifier.trace_id, span_id=identifier.span_id, parent_span_id="b" * 16,
        name=name, instrumentation_scope_name=scope, instrumentation_scope_version="fixture",
        attributes=MappingProxyType(attributes), resource_attributes=MappingProxyType({"service.name": "test"}),
    )
    return identifier, span


def apply_patch(span, patch):
    return {**{key: value for key, value in span.attributes.items() if key not in patch.delete_attributes},
            **patch.set_attributes}


def strands_attributes():
    return {
        "gen_ai.operation.name": "chat", "gen_ai.request.model": CLAUDE, "workshop.cache_write_ttl": "1h",
        "gen_ai.usage.input_tokens": 66, "gen_ai.usage.prompt_tokens": 66,
        "gen_ai.usage.output_tokens": 11, "gen_ai.usage.completion_tokens": 11, "gen_ai.usage.total_tokens": 77,
        "gen_ai.usage.cache_read.input_tokens": 17, "gen_ai.usage.cache_read_input_tokens": 17,
        "gen_ai.usage.cache_creation.input_tokens": 42, "gen_ai.usage.cache_write_input_tokens": 42,
        "session.id": "fixture-session", "gen_ai.task.output": "answer",
    }


def patch_batch(*pairs):
    from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult

    result = normalize_langfuse_spans(params=MaskOtelSpansParams(spans=dict(pairs)))
    assert isinstance(result, MaskOtelSpansResult)
    return result.span_patches


def test_strands_inclusive_input_and_aliases_normalize_once():
    attrs = strands_attributes()
    original = dict(attrs)
    identifier, span = batch_span(attrs)
    patch = patch_batch((identifier, span))[identifier]
    output = apply_patch(span, patch)
    assert json.loads(output[USAGE]) == {
        "input": 7, "output": 11, "cache_read_input_tokens": 17, "cache_write_1h_input_tokens": 42,
    }
    assert not any(key.startswith("gen_ai.usage.") for key in output)
    assert output[STATUS] == "canonical"
    assert output["session.id"] == "fixture-session"
    assert output["gen_ai.task.output"] == "answer"
    assert attrs == original
    assert span.attributes == original


def test_strands_without_uniform_ttl_is_marked_unpriced():
    attrs = strands_attributes()
    attrs.pop("workshop.cache_write_ttl")
    identifier, span = batch_span(attrs)
    output = apply_patch(span, patch_batch((identifier, span))[identifier])
    assert json.loads(output[USAGE])["cache_write_unpriced_input_tokens"] == 42
    assert output[STATUS] == "unpriced_cache"
    assert "langfuse.observation.cost_details" not in output


@pytest.mark.parametrize(
    "scope,operation,status",
    [
        ("strands.telemetry.tracer", "invoke_agent", "suppressed_aggregate"),
        ("opentelemetry.instrumentation.botocore.bedrock-runtime", "chat", "suppressed_duplicate"),
        ("opentelemetry.instrumentation.botocore", "chat", "suppressed_duplicate"),
        ("opentelemetry.instrumentation.openai", "chat", "suppressed_duplicate"),
        ("opentelemetry.instrumentation.openai_v2", "chat", "suppressed_duplicate"),
        ("openai", "chat", "suppressed_duplicate"),
    ],
)
def test_aggregate_and_sdk_duplicates_keep_trace_but_cannot_infer_usage_or_cost(scope, operation, status):
    attrs = {
        **strands_attributes(), "gen_ai.operation.name": operation, USAGE: '{"input": 66, "output": 11}',
        "langfuse.observation.cost_details": '{"total": 1.0}', "gen_ai.response.model": CLAUDE,
        "llm.model_name": CLAUDE, "llm.token_count.prompt": 66, "llm.cost.total": 1.0,
        "langfuse.observation.model.name": CLAUDE,
    }
    identifier, span = batch_span(attrs, scope=scope)
    output = apply_patch(span, patch_batch((identifier, span))[identifier])
    assert output[STATUS] == status
    assert output["langfuse.observation.type"] == "span"
    assert output["session.id"] == "fixture-session"
    assert output["gen_ai.task.output"] == "answer"
    assert USAGE not in output
    assert "langfuse.observation.cost_details" not in output
    assert "gen_ai.request.model" not in output
    assert "gen_ai.response.model" not in output
    assert "langfuse.observation.model.name" not in output
    assert not any(key.startswith(("gen_ai.usage.", "llm.token_count.", "llm.cost.")) for key in output)
    assert attrs[USAGE] == '{"input": 66, "output": 11}'


@pytest.mark.parametrize("api,read_meter", [("responses", "cache_read_30m_input_tokens"),
                                          ("converse", "cache_read_unpriced_input_tokens")])
def test_annotated_converse_scope_uses_full_usage_and_api(api, read_meter):
    usage = NormalizedUsage(7, 11, 17, 42, {"30m": 42})
    attrs = {
        "workshop.model_call": True, "workshop.api": api,
        "workshop.normalized_usage": json.dumps(usage.as_dict()), "gen_ai.request.model": GPT,
        "gen_ai.usage.input_tokens": 7,
    }
    identifier, span = batch_span(attrs, scope=CONVERSE_SCOPE)
    output = apply_patch(span, patch_batch((identifier, span))[identifier])
    assert json.loads(output[USAGE]) == langfuse_usage_details(usage, model_id=GPT, api=api)
    assert json.loads(output[USAGE])[read_meter] == 17


def test_explicit_sdk_generation_usage_and_cost_remain_byte_for_byte_exact():
    explicit = '{ "input": 7, "output": 11, "cache_read_30m_input_tokens": 17, "cache_write_30m_input_tokens": 42 }'
    attrs = {
        USAGE: explicit, "langfuse.observation.type": "generation",
        "langfuse.observation.cost_details": '{ "total": 1.25 }',
        "langfuse.observation.model.name": GPT, "gen_ai.usage.input_tokens": 66,
    }
    identifier, span = batch_span(attrs, scope="langfuse-sdk")
    output = apply_patch(span, patch_batch((identifier, span))[identifier])
    assert output[USAGE] == explicit
    assert output["langfuse.observation.cost_details"] == attrs["langfuse.observation.cost_details"]
    assert "gen_ai.usage.input_tokens" not in output


@pytest.mark.parametrize(
    "model,api,ttl,legacy,expected",
    [
        (CLAUDE, "converse", None, {"cache_creation_input_tokens": 42},
         {"cache_read_input_tokens": 17, "cache_write_unpriced_input_tokens": 42}),
        (CLAUDE, "converse", "1h", {"cache_creation_input_tokens": 42, "cache_write_input_tokens": 42},
         {"cache_read_input_tokens": 17, "cache_write_1h_input_tokens": 42}),
        (CLAUDE, "converse", None,
         {"cache_creation_input_tokens": 42, "cache_write_5m_input_tokens": 19, "cache_write_1h_input_tokens": 23},
         {"cache_read_input_tokens": 17, "cache_write_5m_input_tokens": 19, "cache_write_1h_input_tokens": 23}),
        (CLAUDE, "converse", "future", {"cache_write_input_tokens": 42},
         {"cache_read_input_tokens": 17, "cache_write_unpriced_input_tokens": 42}),
        (GPT, "responses", None, {"cache_creation_input_tokens": 42, "cache_read_30m_input_tokens": 17},
         {"cache_read_30m_input_tokens": 17, "cache_write_30m_input_tokens": 42}),
        (GPT, "converse", None, {"cache_creation_input_tokens": 42},
         {"cache_read_unpriced_input_tokens": 17, "cache_write_unpriced_input_tokens": 42}),
        ("unknown-model", "converse", None, {"cache_creation_input_tokens": 42},
         {"cache_read_unpriced_input_tokens": 17, "cache_write_unpriced_input_tokens": 42}),
    ],
)
def test_legacy_sdk_aliases_become_disjoint_canonical_meters(model, api, ttl, legacy, expected):
    raw = {"input": 7, "output": 11, "cache_read_input_tokens": 17, **legacy}
    attrs = {USAGE: json.dumps(raw), "langfuse.observation.model.name": model, "workshop.api": api}
    if ttl is not None:
        attrs["workshop.cache_write_ttl"] = ttl
    identifier, span = batch_span(attrs, scope="langfuse-sdk")
    output = apply_patch(span, patch_batch((identifier, span))[identifier])
    assert json.loads(output[USAGE]) == {"input": 7, "output": 11, **expected}
    assert sum(json.loads(output[USAGE]).values()) == 77
    assert output[STATUS] == ("unpriced_cache" if any("_unpriced_" in key for key in expected) else "canonical")
    assert json.loads(attrs[USAGE]) == raw


@pytest.mark.parametrize(
    "extra",
    [
        {"cache_write_input_tokens": 41},
        {"cache_write_1h_input_tokens": 41},
        {"cache_read_30m_input_tokens": 16},
        {"cache_creation_input_tokens": -42},
        {"cache_creation_input_tokens": True},
    ],
)
def test_conflicting_sdk_aliases_are_unpriced_without_dropping_batch(extra):
    raw = {"input": 7, "output": 11, "cache_read_input_tokens": 17, "cache_creation_input_tokens": 42, **extra}
    bad_id, bad = batch_span(
        {USAGE: json.dumps(raw), "langfuse.observation.model.name": GPT, "workshop.api": "responses"},
        scope="langfuse-sdk",
    )
    good_id, good = batch_span(strands_attributes(), span_id=2)
    patches = patch_batch((bad_id, bad), (good_id, good))
    output = apply_patch(bad, patches[bad_id])
    assert output[STATUS] == "unpriced_invalid_usage"
    assert USAGE not in output
    assert json.loads(json.loads(output["langfuse.observation.metadata.unpriced_usage"])[USAGE]) == raw
    assert json.loads(apply_patch(good, patches[good_id])[USAGE])["input"] == 7


def test_unknown_sdk_meter_schema_is_preserved_without_claiming_canonical_semantics():
    explicit = '{ "input": 7, "output": 11, "cache_creation_input_tokens": 42, "custom_audio_units": 17 }'
    identifier, span = batch_span({USAGE: explicit, "langfuse.observation.model.name": CLAUDE}, scope="langfuse-sdk")
    output = apply_patch(span, patch_batch((identifier, span))[identifier])
    assert output[USAGE] == explicit
    assert output[STATUS] == "unpriced_unknown_usage"
    assert "langfuse.observation.cost_details" not in output


@pytest.mark.parametrize(
    "invalid",
    [
        {"gen_ai.usage.input_tokens": -1},
        {"gen_ai.usage.input_tokens": 1},
        {"gen_ai.usage.output_tokens": None},
        {"gen_ai.usage.cache_read_input_tokens": 9},
        {"workshop.cache_write_ttl": "unknown"},
        {USAGE: "not-json"},
    ],
)
def test_bad_span_cannot_drop_valid_siblings_or_invent_zero_cost(invalid):
    bad_id, bad = batch_span({**strands_attributes(), **invalid})
    good_id, good = batch_span(strands_attributes(), span_id=2)
    unrelated_id, unrelated = batch_span({"session.id": "untouched"}, scope="workshop.journey", span_id=3)
    patches = patch_batch((bad_id, bad), (good_id, good), (unrelated_id, unrelated))
    assert unrelated_id not in patches
    assert json.loads(apply_patch(good, patches[good_id])[USAGE])["input"] == 7
    bad_output = apply_patch(bad, patches[bad_id])
    assert bad_output[STATUS] == "unpriced_invalid_usage"
    assert json.loads(bad_output["langfuse.observation.metadata.unpriced_usage"])
    assert USAGE not in bad_output
    assert "langfuse.observation.cost_details" not in bad_output


def test_real_public_sdk_hook_patches_only_langfuse_export_and_preserves_events_ids(monkeypatch):
    pytest.importorskip("langfuse")
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    for key in ("LANGFUSE_TRACING_ENABLED", "LANGFUSE_TRACING_ENVIRONMENT", "OTEL_SDK_DISABLED"):
        monkeypatch.delenv(key, raising=False)
    aws_export, lf_export = InMemorySpanExporter(), InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(aws_export))
    client = Langfuse(
        public_key=f"pk-offline-{uuid4()}", secret_key="sk-offline",
        base_url="http://127.0.0.1:1", tracer_provider=provider, span_exporter=lf_export,
        mask_otel_spans=normalize_langfuse_spans,
    )
    attrs = strands_attributes()
    try:
        tracer = provider.get_tracer("strands.telemetry.tracer")
        with (
            tracer.start_as_current_span(
                "invoke_agent fixture", attributes={**attrs, "gen_ai.operation.name": "invoke_agent"}
            ),
            tracer.start_as_current_span("chat", attributes=attrs) as span,
        ):
            span.add_event("gen_ai.choice", {"message": "answer"})
        with client.start_as_current_observation(as_type="generation", name="explicit", model=GPT) as generation:
            generation.update(usage_details={"input": 7, "output": 11, "cache_read_30m_input_tokens": 17})
        assert provider.force_flush(timeout_millis=5000)
        aws_spans = {span.name: span for span in aws_export.get_finished_spans()}
        lf_spans = {span.name: span for span in lf_export.get_finished_spans()}
        assert set(aws_spans) == set(lf_spans) == {"chat", "invoke_agent fixture", "explicit"}
        aws_chat, lf_chat = aws_spans["chat"], lf_spans["chat"]
        assert dict(aws_chat.attributes) == attrs
        assert "gen_ai.usage.input_tokens" not in lf_chat.attributes
        assert json.loads(lf_chat.attributes[USAGE]) == {
            "input": 7, "output": 11, "cache_read_input_tokens": 17, "cache_write_1h_input_tokens": 42,
        }
        assert USAGE not in lf_spans["invoke_agent fixture"].attributes
        assert lf_spans["explicit"].attributes[USAGE] == aws_spans["explicit"].attributes[USAGE]
        for name in aws_spans:
            source, exported = aws_spans[name], lf_spans[name]
            assert exported.context == source.context
            assert exported.parent == source.parent
            assert exported.events == source.events
            assert exported.resource == source.resource
            assert exported.start_time == source.start_time
            assert exported.end_time == source.end_time
    finally:
        client.shutdown()
        provider.shutdown()
