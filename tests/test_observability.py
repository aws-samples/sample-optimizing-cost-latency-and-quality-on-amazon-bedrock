"""Offline telemetry checks: exporters are faked and AWS clients are stubbed."""

from __future__ import annotations

import copy
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from workshop_utils import observability as obs
from workshop_utils.pacing import InferencePacer


@pytest.fixture
def telemetry(monkeypatch):
    clock = [100.0]
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    pacer = InferencePacer(clock=lambda: clock[0], sleeper=sleep)
    monkeypatch.setattr("workshop_utils.pacing.get_workshop_pacer", lambda: pacer)
    trace_sdk = pytest.importorskip("opentelemetry.sdk.trace")
    trace_api = pytest.importorskip("opentelemetry.trace")
    resources = pytest.importorskip("opentelemetry.sdk.resources")
    provider = trace_sdk.TracerProvider(resource=resources.Resource({"service.name": "test"}))
    current = [trace_api.ProxyTracerProvider()]
    events = []
    trace = SimpleNamespace(
        ProxyTracerProvider=trace_api.ProxyTracerProvider,
        get_tracer_provider=lambda: current[0],
        set_tracer_provider=lambda item: current.__setitem__(0, item),
        get_tracer=lambda scope: current[0].get_tracer(scope),
    )

    def initialize(**kwargs):
        assert kwargs == {"swallow_exceptions": False}
        events.append("adot")
        current[0] = provider

    def langfuse(**kwargs):
        assert kwargs["tracer_provider"] is current[0]
        assert kwargs["mask_otel_spans"] is obs.normalize_langfuse_spans
        events.append("langfuse")
        return Mock()

    logs = Mock()
    logs.get_paginator.return_value.paginate.return_value = [
        {"logStreams": [{"logStreamName": "runtime-logs"}]}
    ]
    modules = {
        "boto3": SimpleNamespace(client=Mock(return_value=logs)),
        "opentelemetry.trace": trace,
        "opentelemetry.sdk.trace": trace_sdk,
        "opentelemetry.sdk.resources": resources,
        "amazon.opentelemetry.distro": SimpleNamespace(),
        "opentelemetry.instrumentation.auto_instrumentation": SimpleNamespace(initialize=initialize),
        "langfuse": SimpleNamespace(Langfuse=langfuse),
    }

    def dependency(name, package):
        return modules[name] if name in modules else importlib.import_module(name)

    monkeypatch.setattr(obs, "_dependency", dependency)
    monkeypatch.setattr(obs, "_state", None)
    monkeypatch.setattr(obs, "_initialization_failed", False)
    # Undo every environment write from setup after each test.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key in tuple(os.environ):
        if key.startswith(("OTEL_", "LANGFUSE_", "AGENT_OBSERVABILITY")):
            os.environ.pop(key)
    for key in ("WORKSHOP_NOTEBOOK_LOG_GROUP", "WORKSHOP_LOG_GROUP"):
        os.environ.pop(key, None)
    os.environ.update({"LANGFUSE_PUBLIC_KEY": "pk-fixture", "LANGFUSE_SECRET_KEY": "sk-fixture"})
    yield SimpleNamespace(
        events=events,
        provider=provider,
        current=current,
        modules=modules,
        pacer=pacer,
        pacing_clock=clock,
        pacing_sleeps=sleeps,
    )
    provider.shutdown()
    if current[0] is not provider and hasattr(current[0], "shutdown"):
        current[0].shutdown()


def test_import_and_none_need_no_optional_sdks(monkeypatch):
    # A fresh interpreter blocks every optional import, without touching globals
    # of the parent process or relying on its installed packages.
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('boto', 'opentelemetry', 'langfuse', 'bedrock_agentcore', 'pandas')):
        raise AssertionError('optional import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from workshop_utils import observability, evals, metrics
state = observability.setup_observability('none')
assert state.flush()
assert state.collector.get_finished_spans() == ()
"""
    completed = subprocess.run([sys.executable, "-B", "-c", code], text=True, capture_output=True, timeout=15)
    assert completed.returncode == 0, completed.stderr


def test_setup_idempotent_and_adot_before_langfuse(telemetry):
    state = obs.setup_observability("both", service_name="notebook")
    assert state is obs.setup_observability("both", service_name="notebook")
    assert telemetry.modules["boto3"].client.call_count == 1
    assert telemetry.events == ["adot", "langfuse"]
    assert state.provider is telemetry.provider
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in os.environ
    assert "OTEL_EXPORTER_OTLP_TRACES_HEADERS" not in os.environ
    assert state.flush()
    with pytest.raises(RuntimeError, match="restart"):
        obs.setup_observability("langfuse", service_name="notebook")
    with pytest.raises(RuntimeError, match="restart"):
        obs.setup_observability("both", service_name="other")


@pytest.mark.parametrize(
    "canonical,legacy,explicit,expected",
    [
        (None, None, None, "/aws/bedrock-agentcore/workshop/notebook-agents"),
        ("/canonical", None, None, "/canonical"),
        (None, "/legacy", None, "/legacy"),
        ("/canonical", "/legacy", None, "/canonical"),
        ("/canonical", "/legacy", "/explicit", "/explicit"),
        ("", "/legacy", None, "/legacy"),
        ("", "", None, "/aws/bedrock-agentcore/workshop/notebook-agents"),
    ],
)
def test_log_group_resolution_and_exporter_configuration(telemetry, canonical, legacy, explicit, expected):
    if canonical is not None:
        os.environ["WORKSHOP_NOTEBOOK_LOG_GROUP"] = canonical
    if legacy is not None:
        os.environ["WORKSHOP_LOG_GROUP"] = legacy

    state = obs.setup_observability("both", log_group=explicit)

    assert state.log_group == expected
    assert "OTEL_EXPORTER_OTLP_TRACES_HEADERS" not in os.environ
    assert os.environ["OTEL_EXPORTER_OTLP_LOGS_HEADERS"].startswith(f"x-aws-log-group={expected},")
    assert f"aws.log.group.names={expected}" in os.environ["OTEL_RESOURCE_ATTRIBUTES"].split(",")
    # Idempotency uses the resolved value, regardless of how it was supplied.
    assert obs.setup_observability("both", log_group=expected) is state


def test_changed_log_group_environment_requires_restart(telemetry):
    os.environ["WORKSHOP_NOTEBOOK_LOG_GROUP"] = "/first"
    state = obs.setup_observability()
    os.environ["WORKSHOP_NOTEBOOK_LOG_GROUP"] = "/second"
    with pytest.raises(RuntimeError, match="restart"):
        obs.setup_observability()
    assert obs.setup_observability(log_group="/first") is state


def test_explicit_empty_log_group_does_not_fall_back_to_environment(telemetry):
    os.environ["WORKSHOP_NOTEBOOK_LOG_GROUP"] = "/canonical"
    with pytest.raises(ValueError, match="log_group must be nonempty"):
        obs.setup_observability(log_group="")
    assert telemetry.events == []


def test_notebook_setup_reuses_the_canonical_environment_group(telemetry, monkeypatch):
    dotenv = pytest.importorskip("dotenv")
    monkeypatch.setattr(dotenv, "load_dotenv", Mock(return_value=False))
    os.environ["WORKSHOP_NOTEBOOK_LOG_GROUP"] = "/canonical-from-code-editor"
    os.environ["WORKSHOP_LOG_GROUP"] = "/legacy"
    os.environ["OBSERVABILITY_BACKEND"] = "agentcore"
    repo = Path(__file__).resolve().parents[1]
    notebook = json.loads((repo / "01-fundamentals/02-observability-and-evaluation.ipynb").read_text())
    (setup_cell,) = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "telemetry = setup_observability(" in "".join(cell["source"])
    ]
    namespace = {"REPO": repo}

    exec(compile(setup_cell, "<notebook setup>", "exec"), namespace)

    assert namespace["telemetry"].log_group == "/canonical-from-code-editor"
    assert namespace["LOG_GROUP"] == namespace["telemetry"].log_group
    assert "x-aws-log-group=/canonical-from-code-editor," in os.environ["OTEL_EXPORTER_OTLP_LOGS_HEADERS"]


def test_log_routing_headers_are_not_valid_on_xray_trace_endpoint(telemetry):
    os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = "x-aws-log-group=/example,x-aws-log-stream=spans"
    with pytest.raises(ValueError, match="OTEL_EXPORTER_OTLP_TRACES_HEADERS"):
        obs.setup_observability("agentcore")


def test_shutdown_closes_metrics_and_logs_before_trace_provider(telemetry, monkeypatch):
    order = []
    telemetry.modules["opentelemetry.metrics"] = SimpleNamespace(
        get_meter_provider=lambda: SimpleNamespace(shutdown=lambda: order.append("metrics"))
    )
    telemetry.modules["opentelemetry._logs"] = SimpleNamespace(
        get_logger_provider=lambda: SimpleNamespace(shutdown=lambda: order.append("logs"))
    )
    state = obs.setup_observability("agentcore")
    monkeypatch.setattr(state.provider, "shutdown", lambda: order.append("traces"))
    state.shutdown()
    state.shutdown()
    assert order == ["metrics", "logs", "traces"]
    with pytest.raises(RuntimeError, match="restart"):
        obs.setup_observability("agentcore")


@pytest.mark.parametrize("backend", ["agentcore", "langfuse", "none"])
def test_backend_modes(telemetry, backend):
    state = obs.setup_observability(backend)
    assert telemetry.events == {"agentcore": ["adot"], "langfuse": ["langfuse"], "none": []}[backend]
    assert state is obs.setup_observability(backend)
    if backend in {"langfuse", "none"}:
        telemetry.modules["boto3"].client.assert_not_called()
    if backend == "none":
        assert state.provider is None
        with pytest.raises(RuntimeError, match="restart"):
            obs.setup_observability("agentcore")


@pytest.fixture
def log_readiness(telemetry, monkeypatch):
    import boto3
    from botocore.stub import Stubber

    # Explicit fixture credentials avoid the default credential chain/IMDS.
    client = boto3.client(
        "logs", region_name="eu-west-1",
        aws_access_key_id="fixture", aws_secret_access_key="fixture",
    )
    monkeypatch.setattr(
        client._endpoint.http_session, "send", Mock(side_effect=AssertionError("Unexpected live AWS call"))
    )
    close = Mock(wraps=client.close)
    monkeypatch.setattr(client, "close", close)
    factory = Mock(return_value=client)
    telemetry.modules["boto3"] = SimpleNamespace(client=factory)
    with Stubber(client) as stub:
        yield SimpleNamespace(
            stub=stub, client=client, close=close, factory=factory,
            describe={"logGroupName": "/provisioned/notebook", "logStreamNamePrefix": "runtime-logs"},
            create={"logGroupName": "/provisioned/notebook", "logStreamName": "runtime-logs"},
        )
        stub.assert_no_pending_responses()
    client.close()


@pytest.mark.parametrize("backend", ["agentcore", "both"])
@pytest.mark.parametrize("stream_state", ["existing", "missing", "prefix_only", "paginated", "concurrent"])
def test_log_stream_is_ready_before_adot_initializes(telemetry, log_readiness, backend, stream_state):
    readiness = log_readiness
    if stream_state == "paginated":
        readiness.stub.add_response(
            "describe_log_streams", {"logStreams": [], "nextToken": "page2"}, readiness.describe
        )
        readiness.stub.add_response(
            "describe_log_streams", {"logStreams": [{"logStreamName": "runtime-logs"}]},
            readiness.describe | {"nextToken": "page2"},
        )
    else:
        names = {
            "existing": ["runtime-logs"],
            "missing": [],
            "prefix_only": ["runtime-logs-old"],
            "concurrent": [],
        }[stream_state]
        readiness.stub.add_response(
            "describe_log_streams", {"logStreams": [{"logStreamName": name} for name in names]},
            readiness.describe,
        )
    if stream_state in {"missing", "prefix_only"}:
        readiness.stub.add_response("create_log_stream", {}, readiness.create)
    elif stream_state == "concurrent":
        readiness.stub.add_client_error(
            "create_log_stream", "ResourceAlreadyExistsException", "Created by another kernel",
            expected_params=readiness.create,
        )
    auto = telemetry.modules["opentelemetry.instrumentation.auto_instrumentation"]
    initialize = auto.initialize

    def initialize_after_readiness(**kwargs):
        # Check at exporter startup, not just after setup has returned.
        readiness.stub.assert_no_pending_responses()
        readiness.close.assert_called_once()
        assert os.environ["OTEL_EXPORTER_OTLP_LOGS_HEADERS"] == (
            "x-aws-log-group=/provisioned/notebook,x-aws-log-stream=runtime-logs,"
            "x-aws-metric-namespace=bedrock-agentcore"
        )
        initialize(**kwargs)

    auto.initialize = initialize_after_readiness
    state = obs.setup_observability(backend, region="eu-west-1", log_group="/provisioned/notebook")
    assert state is obs.setup_observability(backend, region="eu-west-1", log_group="/provisioned/notebook")
    readiness.factory.assert_called_once_with("logs", region_name="eu-west-1")
    assert telemetry.events == (["adot", "langfuse"] if backend == "both" else ["adot"])
    assert "OTEL_EXPORTER_OTLP_TRACES_HEADERS" not in os.environ


@pytest.mark.parametrize("operation", ["describe_log_streams", "create_log_stream"])
@pytest.mark.parametrize("code", [
    "AccessDeniedException", "ResourceNotFoundException", "ServiceUnavailableException", "ThrottlingException",
])
def test_log_readiness_errors_are_preserved_before_any_initialization(telemetry, log_readiness, operation, code):
    from botocore.exceptions import ClientError

    readiness = log_readiness
    if operation == "create_log_stream":
        readiness.stub.add_response("describe_log_streams", {"logStreams": []}, readiness.describe)
    readiness.stub.add_client_error(
        operation, code, "Original AWS error", response_meta={"RequestId": "request-fixture"},
        expected_params=readiness.describe if operation == "describe_log_streams" else readiness.create,
    )
    environment = dict(os.environ)
    with pytest.raises(ClientError) as error:
        obs.setup_observability("both", region="eu-west-1", log_group="/provisioned/notebook")
    assert error.value.response["Error"] == {"Code": code, "Message": "Original AWS error"}
    assert error.value.response["ResponseMetadata"]["RequestId"] == "request-fixture"
    assert error.value.operation_name == (
        "DescribeLogStreams" if operation == "describe_log_streams" else "CreateLogStream"
    )
    assert telemetry.events == []
    assert os.environ == environment
    assert obs._state is None
    assert not obs._initialization_failed
    readiness.close.assert_called_once()

    # Readiness is retryable after the provisioning/access problem is corrected.
    readiness.stub.add_response(
        "describe_log_streams", {"logStreams": [{"logStreamName": "runtime-logs"}]}, readiness.describe
    )
    assert obs.setup_observability("both", region="eu-west-1", log_group="/provisioned/notebook").provider
    assert telemetry.events == ["adot", "langfuse"]


def test_log_readiness_transport_error_is_not_reported_as_success(telemetry, log_readiness, monkeypatch):
    from botocore.exceptions import EndpointConnectionError

    failure = EndpointConnectionError(endpoint_url="https://logs.eu-west-1.amazonaws.com")
    monkeypatch.setattr(log_readiness.client, "describe_log_streams", Mock(side_effect=failure))
    with pytest.raises(EndpointConnectionError) as error:
        obs.setup_observability("agentcore", region="eu-west-1", log_group="/provisioned/notebook")
    assert error.value is failure
    assert telemetry.events == []
    assert obs._state is None
    assert not obs._initialization_failed


@pytest.mark.parametrize("setting", ["OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_HEADERS"])
def test_generic_otlp_conflict_is_not_silently_rewritten(telemetry, setting):
    os.environ[setting] = "user-config"
    with pytest.raises(ValueError, match=setting):
        obs.setup_observability("both")
    assert os.environ[setting] == "user-config"
    assert telemetry.events == []


def test_preexisting_provider_fails_before_any_initialization(telemetry):
    telemetry.current[0] = telemetry.provider
    with pytest.raises(RuntimeError, match="already exists"):
        obs.setup_observability("both")
    assert telemetry.events == []


def test_partial_init_poisoned_until_restart(telemetry):
    telemetry.modules["langfuse"].Langfuse = Mock(side_effect=RuntimeError("constructor failed"))
    with pytest.raises(RuntimeError, match="initialization failed"):
        obs.setup_observability("both")
    with pytest.raises(RuntimeError, match="partially failed"):
        obs.setup_observability("both")
    assert telemetry.events == ["adot"]


def test_missing_dependency_and_keys_fail_clearly(telemetry, monkeypatch):
    os.environ.pop("LANGFUSE_SECRET_KEY")
    with pytest.raises(ValueError, match="LANGFUSE_SECRET_KEY"):
        obs.setup_observability("both")
    assert not obs._initialization_failed
    monkeypatch.setattr(importlib, "import_module", Mock(side_effect=ModuleNotFoundError("missing")))
    with pytest.raises(ImportError, match="optional dependency 'langfuse'"):
        # Exercise the real lazy import error, not the fixture's dependency shim.
        source = obs.__file__
        spec = importlib.util.spec_from_file_location("workshop_observability_test", source)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        module._dependency("langfuse", "langfuse")


def test_capture_converse_usage_without_ttft_or_mutation(telemetry):
    from opentelemetry import baggage

    state = obs.setup_observability("agentcore")
    response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "30 days."}]}},
        "usage": {"inputTokens": 9, "outputTokens": 4, "cacheReadInputTokens": 100, "cacheWriteInputTokens": 20},
        "metrics": {"latencyMs": 412},
    }
    original = copy.deepcopy(response)
    client = Mock()

    def converse(**kwargs):
        assert baggage.get_baggage("session.id") == "session-fixture"
        return response

    client.converse.side_effect = converse
    result = obs.traced_converse(
        client,
        session_id="session-fixture",
        modelId="model-fixture",
        messages=[{"role": "user", "content": [{"text": "Return policy?"}]}],
        system=[{"text": "Use the supplied policy."}],
    )
    assert response == original
    assert "_trace" not in response
    (document,) = state.collector.span_documents()
    assert result["_trace"]["trace_id"] == document["traceId"]
    assert document["attributes"]["gen_ai.task.output"] == "30 days."
    assert document["attributes"]["gen_ai.usage.cache_creation.input_tokens"] == 20
    assert "gen_ai.server.time_to_first_token" not in document["attributes"]
    assert baggage.get_baggage("session.id") is None


@pytest.mark.parametrize("backend", ["none", "agentcore", "langfuse", "both"])
def test_explicit_converse_records_the_same_response_without_invoking_a_helper(telemetry, monkeypatch, backend):
    from opentelemetry import baggage, context

    state = obs.setup_observability(backend)
    request = {
        "modelId": "model-fixture",
        "system": [{"text": "Use the policy."}, {"cachePoint": {"type": "default"}}],
        "messages": [{"role": "user", "content": [{"text": "Return window?"}]}],
        "inferenceConfig": {"maxTokens": 128},
    }
    response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "30 days"}]}},
        "usage": {
            "inputTokens": 8, "outputTokens": 3, "cacheReadInputTokens": 100,
            "cacheWriteInputTokens": 20,
            "cacheDetails": [{"ttl": "5m", "inputTokens": 7}, {"ttl": "1h", "inputTokens": 13}],
        },
        "metrics": {"latencyMs": 10},
    }
    original_request, original_response = copy.deepcopy(request), copy.deepcopy(response)
    monkeypatch.setattr(obs, "call_converse", Mock(side_effect=AssertionError("Hidden inference call")))
    client = Mock()

    def converse(**sent):
        assert sent == original_request
        assert baggage.get_baggage("session.id") == ("outer-session" if backend == "none" else "native-session")
        return response

    client.converse.side_effect = converse
    token = context.attach(baggage.set_baggage("session.id", "outer-session"))
    try:
        with obs.converse_observation(request, session_id="native-session") as record_response:
            result = client.converse(**request)
            result = record_response(result)
        assert baggage.get_baggage("session.id") == "outer-session"
    finally:
        context.detach(token)
    client.converse.assert_called_once_with(**request)
    assert request == original_request and response == original_response
    assert "_trace" not in response
    if backend == "none":
        assert result is response
        assert state.collector.span_documents() == []
    else:
        (span,) = state.collector.span_documents()
        assert result["_trace"]["trace_id"] == span["traceId"]
        assert result["_trace"]["session_id"] == "native-session"
        attributes = span["attributes"]
        assert attributes["gen_ai.task.output"] == "30 days"
        usage = json.loads(attributes["workshop.normalized_usage"])
        assert usage["cache_write_by_ttl"] == {"5m": 7, "1h": 13}
        if backend in {"langfuse", "both"}:
            assert attributes["langfuse.observation.input"] == "Return window?"
            assert attributes["langfuse.observation.output"] == "30 days"


def test_explicit_converse_releases_baggage_when_sdk_raises(telemetry):
    from opentelemetry import baggage

    state = obs.setup_observability("agentcore")
    client = Mock()
    client.converse.side_effect = RuntimeError("SDK failed")
    request = {"modelId": "model", "messages": []}
    with (
        pytest.raises(RuntimeError, match="SDK failed"),
        obs.converse_observation(request, session_id="failed-session") as record_response,
    ):
        record_response(client.converse(**request))
    assert baggage.get_baggage("session.id") is None
    client.converse.assert_called_once()
    (span,) = state.collector.span_documents()
    assert "gen_ai.task.output" not in span["attributes"]


def test_explicit_converse_retains_attached_per_attempt_pacing(telemetry):
    from botocore.hooks import HierarchicalEmitter

    from workshop_utils.pacing import attach_boto3

    obs.setup_observability("none")
    events = HierarchicalEmitter()
    client = SimpleNamespace(meta=SimpleNamespace(
        events=events, service_model=SimpleNamespace(service_name="bedrock-runtime")
    ))
    response = {"usage": {"inputTokens": 1, "outputTokens": 1}}
    attempts = []

    def converse(**request):
        for attempt in range(3):
            events.emit("before-send.bedrock-runtime.Converse")
            attempts.append((attempt, telemetry.pacing_clock[0]))
        return response

    client.converse = converse
    attach_boto3(client, telemetry.pacer)
    request = {"modelId": "model", "messages": []}
    with obs.converse_observation(request) as record_response:
        result = record_response(client.converse(**request))
    assert result is response
    assert len(attempts) == 3
    assert telemetry.pacing_sleeps == pytest.approx([1.1, 1.1])


@pytest.mark.parametrize(
    "points,cache_details,expected",
    [
        ([{"type": "default"}], None, {"5m": 20}),
        ([{"type": "default", "ttl": "1h"}], None, {"1h": 20}),
        ([{"type": "default", "ttl": "1h"}, {"type": "default"}], None, {}),
        ([], None, {}),
        (
            [{"type": "default"}],
            [{"ttl": "5m", "inputTokens": 7}, {"ttl": "1h", "inputTokens": 13}],
            {"5m": 7, "1h": 13},
        ),
    ],
)
def test_converse_full_usage_retains_details_and_uniform_request_fallback(telemetry, points, cache_details, expected):
    state = obs.setup_observability("both")
    response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "OK"}]}},
        "usage": {"inputTokens": 9, "outputTokens": 4, "cacheReadInputTokens": 100, "cacheWriteInputTokens": 20},
        "metrics": {"latencyMs": 412},
        "ResponseMetadata": {"RequestId": "fixture-request"},
    }
    if cache_details is not None:
        response["usage"]["cacheDetails"] = cache_details
    request = {
        "modelId": "global.anthropic.claude-sonnet-5",
        "messages": [{"role": "user", "content": [{"text": "Question"}]}],
        "system": [{"cachePoint": point} for point in points],
    }
    original_response, original_request = copy.deepcopy(response), copy.deepcopy(request)
    client = Mock()
    client.converse.return_value = response
    result = obs.traced_converse(client, **request)
    (document,) = state.collector.native_span_documents()
    attributes = document["attributes"]
    normalized = json.loads(attributes["workshop.normalized_usage"])
    assert attributes["workshop.api"] == "converse"
    assert normalized == {
        "input_tokens": 9, "output_tokens": 4, "cache_read_tokens": 100, "cache_write_tokens": 20,
        "cache_write_by_ttl": expected, "total_input_tokens": 129, "total_tokens": 133,
    }
    assert attributes["gen_ai.usage.input_tokens"] == 9
    assert attributes["gen_ai.usage.cache_read.input_tokens"] == 100
    assert attributes["gen_ai.usage.cache_creation.input_tokens"] == 20
    assert attributes["gen_ai.server.request.duration"] == 412
    assert response == original_response
    assert request == original_request
    assert {key: value for key, value in result.items() if key != "_trace"} == original_response
    assert result["_trace"]["trace_id"] == document["traceId"]
    assert result["_trace"]["span_id"] == document["spanId"]
    client.converse.assert_called_once_with(**original_request)


def test_invalid_usage_does_not_break_successful_converse(telemetry):
    state = obs.setup_observability("both")
    response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "OK"}]}},
        "usage": {
            "inputTokens": 9, "outputTokens": 4, "cacheWriteInputTokens": 20,
            "cacheDetails": [{"ttl": "future-ttl", "inputTokens": 20}],
        },
    }
    client = Mock()
    client.converse.return_value = response
    result = obs.traced_converse(client, modelId="model-fixture", messages=[])
    assert result["output"] == response["output"]
    (document,) = state.collector.native_span_documents()
    assert document["attributes"]["workshop.usage_status"] == "unpriced_invalid_usage"
    assert document["attributes"]["gen_ai.usage.cache_creation.input_tokens"] == 20


def test_converse_failure_restores_context(telemetry):
    from opentelemetry import baggage

    state = obs.setup_observability("agentcore")
    client = Mock()
    client.converse.side_effect = RuntimeError("model rejected")
    with pytest.raises(RuntimeError, match="model rejected"):
        obs.traced_converse(client, session_id="failed-session", modelId="model", messages=[])
    assert baggage.get_baggage("session.id") is None
    (document,) = state.collector.span_documents()
    assert document["status"]["code"] == "ERROR"
    assert "gen_ai.task.output" not in document["attributes"]


@pytest.mark.parametrize("backend", ["langfuse", "both"])
def test_langfuse_receives_input_output_without_losing_aws_evaluation_fields(telemetry, backend):
    state = obs.setup_observability(backend)
    client = Mock()
    client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "A receipt is required."}]}},
        "usage": {"inputTokens": 12, "outputTokens": 6},
    }
    obs.traced_converse(
        client,
        session_id="langfuse-fixture",
        modelId="model-fixture",
        messages=[{"role": "user", "content": [{"text": "What proof is needed?"}]}],
    )
    (document,) = state.collector.span_documents()
    attributes = document["attributes"]
    assert attributes["langfuse.observation.input"] == attributes["gen_ai.task.input"] == "What proof is needed?"
    assert attributes["langfuse.observation.output"] == attributes["gen_ai.task.output"] == "A receipt is required."
    assert attributes["langfuse.trace.input"] == attributes["gen_ai.task.input"]
    assert attributes["langfuse.trace.output"] == attributes["gen_ai.task.output"]
    assert attributes["gen_ai.usage.input_tokens"] == 12


@pytest.mark.parametrize("backend", ["langfuse", "both"])
def test_env_file_backend_overrides_kernel_default_without_overriding_credentials(tmp_path, monkeypatch, backend):
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "agentcore")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "supplied-credential-fixture")
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"OBSERVABILITY_BACKEND={backend}\nAWS_ACCESS_KEY_ID=file-credential-fixture\n"
        "LANGFUSE_BASE_URL=https://langfuse.example\n"
    )
    assert obs.resolve_notebook_backend(env_file) == backend
    assert os.environ["AWS_ACCESS_KEY_ID"] == "supplied-credential-fixture"
    assert os.environ["LANGFUSE_BASE_URL"] == "https://langfuse.example"


def test_disabled_dotenv_and_missing_file_preserve_explicit_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "none")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    env_file = tmp_path / ".env"
    env_file.write_text("OBSERVABILITY_BACKEND=both\n")
    assert obs.resolve_notebook_backend(env_file) == "none"
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED")
    assert obs.resolve_notebook_backend(tmp_path / "missing.env") == "none"


@pytest.mark.parametrize("invalid", ["", "typo", "${SOME_BACKEND}"])
def test_invalid_file_backend_is_not_silently_replaced_by_kernel_default(tmp_path, monkeypatch, invalid):
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "agentcore")
    env_file = tmp_path / ".env"
    env_file.write_text(f"OBSERVABILITY_BACKEND={invalid}\n")
    with pytest.raises(ValueError, match="OBSERVABILITY_BACKEND"):
        obs.resolve_notebook_backend(env_file)


def test_none_passes_through_and_never_captures_remote_spans(telemetry):
    state = obs.setup_observability("none")
    response = {"output": "remote response"}
    client = Mock()
    client.converse.return_value = response
    assert obs.traced_converse(client, modelId="model", messages=[]) is response
    assert state.collector.get_finished_spans() == ()


@pytest.mark.parametrize("backend", ["none", "agentcore"])
def test_traced_converse_paces_default_and_explicit_pacer_with_unchanged_requests(telemetry, backend):
    obs.setup_observability(backend)
    response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "OK"}]}},
        "usage": {"inputTokens": 2, "outputTokens": 3},
        "metrics": {"latencyMs": 15},
    }
    client = Mock()
    client.converse.return_value = response
    request = {"modelId": "fixture", "messages": [{"role": "user", "content": [{"text": "question"}]}]}
    obs.traced_converse(client, **request)
    result = obs.traced_converse(client, pacer=telemetry.pacer, **request)
    assert telemetry.pacing_sleeps == pytest.approx([1.1])
    assert all(call.kwargs == request for call in client.converse.call_args_list)
    assert result["metrics"]["latencyMs"] == 15
    # An unconstrained Mock invents .events; it must not masquerade as a session
    # and swallow registrations without pacing the callable.
    assert "events" not in client._mock_children


def test_local_capture_is_bounded_and_overflow_is_explicit():
    collector = obs.LocalSpanCollector(2)
    collector.on_end("first")
    collector.on_end("second")
    collector.on_end("third")
    assert collector.get_finished_spans() == ("second", "third")
    assert collector.dropped_spans == 1
    with pytest.raises(RuntimeError, match="overflowed"):
        collector.span_documents()
    with pytest.raises(RuntimeError, match="overflowed"):
        collector.native_span_documents()
    collector.clear()
    assert collector.span_documents() == []
    assert collector.dropped_spans == 0


def test_response_span_is_explicit_response_only():
    document = obs.response_span(
        "Return policy?",
        "30 days.",
        session_id="session",
        trace_id="a" * 32,
        span_id="b" * 16,
        start_time_ns=10,
        end_time_ns=100,
    )
    assert document["durationNano"] == 90
    assert document["attributes"]["workshop.response_only"]
    assert not any(key.startswith("gen_ai.usage") for key in document["attributes"])
    with pytest.raises(ValueError, match="hex IDs"):
        obs.response_span(
            "Q", "A", session_id="session", trace_id="bad", span_id="b" * 16, start_time_ns=10, end_time_ns=20
        )


def test_strands_uses_sdk_conversion_including_log_records(telemetry, monkeypatch):
    serializer = SimpleNamespace(convert_strands_to_adot=Mock(return_value=[{"spanId": "span"}, {"body": {}}]))
    original = obs._dependency
    monkeypatch.setattr(
        obs,
        "_dependency",
        lambda name, package: serializer if name.endswith("span_to_adot_serializer") else original(name, package),
    )
    state = obs.setup_observability("agentcore")
    tool_message = json.dumps([{"toolUse": {"name": "support___get_technical_support", "input": {"issue": "power"}}}])
    with telemetry.provider.get_tracer("strands.telemetry.tracer").start_as_current_span("chat") as span:
        span.add_event("gen_ai.tool.message", {"content": "Observed prior skills result"})
        span.add_event("gen_ai.choice", {"message": tool_message})
    assert state.collector.span_documents() == [{"spanId": "span"}, {"body": {}}]
    native = state.collector.native_span_documents()
    assert len(native) == 1
    assert [event["name"] for event in native[0]["events"]] == ["gen_ai.tool.message", "gen_ai.choice"]
    assert native[0]["events"][1]["attributes"]["message"] == tool_message
    serializer.convert_strands_to_adot.assert_called_once()


def test_observed_otel_arrays_are_json_documents_for_recommendations(telemetry):
    from botocore.session import Session
    from botocore.validate import validate_parameters

    state = obs.setup_observability("agentcore")
    with telemetry.provider.get_tracer(obs.CONVERSE_SCOPE).start_as_current_span("invoke_agent fixture") as span:
        span.set_attribute("gen_ai.response.finish_reasons", ("stop",))
        span.add_event("gen_ai.choice", {"roles": ("assistant", "tool")})
    document = state.collector.native_span_documents()[0]
    assert document["attributes"]["gen_ai.response.finish_reasons"] == ["stop"]
    assert document["events"][0]["attributes"]["roles"] == ["assistant", "tool"]
    assert isinstance(document["startTimeUnixNano"], int)
    # The source observation retains OTEL's tuple representation; only its wire copy changes.
    assert state.collector.get_finished_spans()[0].attributes["gen_ai.response.finish_reasons"] == ("stop",)
    service = Session().get_service_model("bedrock-agentcore")
    shape = service.operation_model("StartRecommendation").input_shape
    request = {
        "name": "WorkshopFixture",
        "type": "SYSTEM_PROMPT_RECOMMENDATION",
        "recommendationConfig": {"systemPromptRecommendationConfig": {
            "systemPrompt": {"text": "Use the supplied policy."},
            "agentTraces": {"sessionSpans": [document]},
            "evaluationConfig": {"evaluators": [{"evaluatorArn": "arn:aws:bedrock-agentcore:::evaluator/Builtin.GoalSuccessRate"}]},
        }},
    }
    validate_parameters(request, shape)
