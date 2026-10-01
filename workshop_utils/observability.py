"""Explicit notebook telemetry setup and bounded, process-local span capture.

Call setup before importing instrumented agents or creating a Langfuse client.
Supports AWS-native and Langfuse export with explicit initialization.
Lab 8 requires ``agentcore`` or ``both``.
This module neither creates AWS resources nor loads credentials on import.
"""

from __future__ import annotations

import importlib
import json
import os
import threading
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from workshop_utils.bedrock import NormalizedUsage, normalize_usage
from workshop_utils.models import UnknownModelError, get_model
from workshop_utils.pacing import InferencePacer, call_converse

CONVERSE_SCOPE = "opentelemetry.instrumentation.workshop_converse"
_STRANDS_SCOPE = "strands.telemetry.tracer"
_USAGE_DETAILS = "langfuse.observation.usage_details"
_USAGE_STATUS = "langfuse.observation.metadata.usage_status"
_WRITE_METERS = tuple(f"cache_write_{ttl}_input_tokens" for ttl in ("5m", "1h", "30m", "unpriced"))
_READ_METERS = ("cache_read_input_tokens", "cache_read_30m_input_tokens", "cache_read_unpriced_input_tokens")
_LEGACY_WRITE_METERS = ("cache_creation_input_tokens", "cache_write_input_tokens")
DEFAULT_NOTEBOOK_LOG_GROUP = "/aws/bedrock-agentcore/workshop/notebook-agents"
_NOTEBOOK_LOG_STREAM = "runtime-logs"
_lock = threading.Lock()
_state = None
_initialization_failed = False


def uniform_cache_ttl(request: Mapping[str, Any], cache_config: Any = None) -> str | None:
    """Return a single observed/configured TTL, or None for absent/mixed TTLs.

    Inspect only Converse content blocks, never tool schemas or tool input.
    Omitted checkpoint TTLs mean Bedrock's 5m default. Optional Strands
    CacheConfig (or a mapping of its fields) includes auto-injected checkpoints;
    section overrides must agree with the conversation TTL before using it.
    This describes configuration, not proof of a cache hit.
    """
    blocks = list(request.get("system", []))
    blocks.extend(request.get("toolConfig", {}).get("tools", []))
    for message in request.get("messages", []):
        blocks.extend(message.get("content", []))
    ttls = set()
    default_ttl = "5m"
    if cache_config is not None:
        config = cache_config if isinstance(cache_config, Mapping) else vars(cache_config)
        default_ttl = config.get("ttl") or "5m"
        auto_enabled = (
            config.get("system_prompt_ttl", True) is not False
            or config.get("tools_ttl") not in (None, False)
        )
        if auto_enabled:
            ttls.add(default_ttl)
        for section in ("system_prompt_ttl", "tools_ttl"):
            value = config.get(section)
            if isinstance(value, str):
                ttls.add(value)
    for block in blocks:
        if isinstance(block, Mapping) and "cachePoint" in block:
            point = block["cachePoint"]
            if not isinstance(point, Mapping) or point.get("type") != "default":
                return None
            ttl = point.get("ttl", default_ttl)
            if not isinstance(ttl, str):
                return None
            ttls.add(ttl)
    return next(iter(ttls)) if len(ttls) == 1 and ttls <= {"5m", "1h", "30m"} else None


def langfuse_usage_details(
    usage: NormalizedUsage, *, model_id: str, api: str = "converse"
) -> dict[str, int]:
    """Disjoint Langfuse meters; register every supported TTL price once.

    ``input`` is uncached input and ``output`` already includes reasoning.
    No total, aggregate cache-write, or duplicate alias is exported. Unknown
    models/APIs/TTLs retain counts in unpriced meters. GPT's fixed 30m cache
    semantics apply only to Responses, never inferred from Converse counts.
    """
    try:
        model = get_model(model_id)
    except UnknownModelError:
        model = None
    claude = model is not None and model.provider == "anthropic" and api == "converse"
    gpt_responses = model is not None and model.provider == "openai" and api == "responses"
    details = {"input": usage.input_tokens, "output": usage.output_tokens}
    if usage.cache_read_tokens:
        meter = (
            "cache_read_input_tokens" if claude
            else "cache_read_30m_input_tokens" if gpt_responses
            else "cache_read_unpriced_input_tokens"
        )
        details[meter] = usage.cache_read_tokens
    writes = dict(usage.cache_write_by_ttl)
    if not writes and usage.cache_write_tokens:
        writes = {"30m" if gpt_responses else "unknown": usage.cache_write_tokens}
    for ttl, count in writes.items():
        if not count:
            continue
        supported = (claude and ttl in model.capabilities.cache_ttls) or (gpt_responses and ttl == "30m")
        meter = f"cache_write_{ttl}_input_tokens" if supported else "cache_write_unpriced_input_tokens"
        details[meter] = details.get(meter, 0) + count
    return details


def _usage_attributes(attributes: Mapping) -> tuple[str, ...]:
    """Aliases that could make Langfuse count or price the same call twice."""
    return tuple(
        key for key in attributes
        if key.startswith(("gen_ai.usage.", "gen_ai.cost.", "llm.usage.", "llm.token_count.", "llm.cost."))
        or key in {_USAGE_DETAILS, "langfuse.observation.cost_details"}
    )


def _unpriced_attributes(details: Mapping) -> dict:
    unpriced = [key for key, value in details.items() if "_unpriced_" in key and value]
    return {
        _USAGE_STATUS: "unpriced_cache" if unpriced else "canonical",
        "langfuse.observation.metadata.unpriced_meters": json.dumps(unpriced),
    }


def _normalize_sdk_usage(details: dict, attributes: Mapping) -> dict | None:
    """Recognize only the workshop's complete uncached-input SDK usage shape.

    Legacy aggregate writes are aliases of the full canonical write breakdown,
    not additional counts. Custom schemas are left to their owner. Returning
    the unchanged mapping lets the caller preserve canonical JSON byte-for-byte.
    """
    known = {"input", "output", *_READ_METERS, *_WRITE_METERS, *_LEGACY_WRITE_METERS}
    if not {"input", "output"} <= details.keys() or details.keys() - known:
        return None
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in details.values()):
        raise ValueError("Usage counts must be nonnegative integers")
    legacy_writes = [details[key] for key in _LEGACY_WRITE_METERS if key in details]
    typed_reads = {key: details[key] for key in _READ_METERS[1:] if key in details}
    duplicate_read = "cache_read_input_tokens" in details and bool(typed_reads)
    if not legacy_writes and not duplicate_read:
        return details
    if len(set(legacy_writes)) > 1:
        raise ValueError("Conflicting legacy cache write aliases")
    writes = {key: details[key] for key in _WRITE_METERS if key in details}
    if legacy_writes and writes and legacy_writes[0] != sum(writes.values()):
        raise ValueError("Aggregate cache writes conflict with the TTL breakdown")
    if duplicate_read and details["cache_read_input_tokens"] != sum(typed_reads.values()):
        raise ValueError("Aggregate cache reads conflict with canonical reads")

    write_count = legacy_writes[0] if legacy_writes and not writes else 0
    ttl = attributes.get("workshop.cache_write_ttl")
    usage = NormalizedUsage(
        details["input"], details["output"],
        0 if typed_reads else details.get("cache_read_input_tokens", 0),
        write_count, {ttl: write_count} if ttl in {"5m", "1h", "30m"} else {},
    )
    normalized = langfuse_usage_details(
        usage,
        model_id=attributes.get("langfuse.observation.model.name", attributes.get("gen_ai.request.model", "")),
        api=attributes.get("workshop.api", "converse"),
    )
    normalized.update(typed_reads)
    normalized.update(writes)
    return normalized


def _suppressed_usage_patch(attributes: Mapping, status: str) -> Any:
    from langfuse.types import OtelSpanPatch

    # Remove model inference inputs too: missing usage must not trigger a
    # tokenizer estimate on an aggregate or duplicate. IDs/events remain intact.
    models = ("gen_ai.request.model", "gen_ai.response.model", "llm.model_name", "langfuse.observation.model.name")
    return OtelSpanPatch(
        delete_attributes=(*_usage_attributes(attributes), *(key for key in models if key in attributes)),
        set_attributes={
            "langfuse.observation.type": "span",
            _USAGE_STATUS: status,
            "langfuse.observation.metadata.source_model": str(
                attributes.get("gen_ai.request.model", attributes.get("langfuse.observation.model.name", ""))
            ),
        },
    )


def normalize_langfuse_spans(*, params: Any) -> Any:
    """Public Langfuse ``mask_otel_spans`` hook; affects only its export copy.

    Register on every Langfuse constructor, including Runtime. Strands ``chat``
    spans should carry ``workshop.cache_write_ttl`` only when configuration is
    uniform. Canonical SDK generation ``usage_details`` are preserved exactly;
    recognized legacy workshop aliases are deduplicated. Unknown SDK meter
    schemas stay unchanged and are marked unpriced. Botocore/OpenAI spans are secondary in this
    workshop: their owner is a traced Converse, Strands chat, or explicit SDK
    generation. Keep those secondary spans but suppress their billing meters.

    Each span is isolated: malformed usage annotates an unpriced observation,
    never raises from the callback and drops a whole SDK export batch.
    """
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    for identifier, span in params.spans.items():
        attributes = span.attributes
        try:
            scope = span.instrumentation_scope_name or ""
            duplicate = any(
                scope == prefix or scope.startswith(prefix + ".")
                for prefix in (
                    "opentelemetry.instrumentation.botocore",
                    "opentelemetry.instrumentation.openai",
                    "opentelemetry.instrumentation.openai_v2",
                    "openai",
                )
            )
            if duplicate:
                patches[identifier] = _suppressed_usage_patch(attributes, "suppressed_duplicate")
                continue
            if scope == _STRANDS_SCOPE and attributes.get("gen_ai.operation.name") == "invoke_agent":
                patches[identifier] = _suppressed_usage_patch(attributes, "suppressed_aggregate")
                continue
            if _USAGE_DETAILS in attributes:
                details = json.loads(attributes[_USAGE_DETAILS])
                if not isinstance(details, dict):
                    raise ValueError("usage_details must be an object")
                normalized = _normalize_sdk_usage(details, attributes)
                if normalized is None:
                    export_attributes = {
                        _USAGE_STATUS: "unpriced_unknown_usage",
                        "langfuse.observation.metadata.unpriced_meters": json.dumps(list(details)),
                    }
                else:
                    export_attributes = _unpriced_attributes(normalized)
                    if normalized != details:
                        export_attributes[_USAGE_DETAILS] = json.dumps(normalized)
                patches[identifier] = OtelSpanPatch(
                    delete_attributes=tuple(
                        key for key in _usage_attributes(attributes)
                        if key not in {_USAGE_DETAILS, "langfuse.observation.cost_details"}
                    ),
                    set_attributes=export_attributes,
                )
                continue
            if scope == CONVERSE_SCOPE and attributes.get("workshop.model_call"):
                raw = json.loads(attributes["workshop.normalized_usage"])
                usage = NormalizedUsage(
                    raw["input_tokens"], raw["output_tokens"], raw["cache_read_tokens"],
                    raw["cache_write_tokens"], raw["cache_write_by_ttl"],
                )
            elif scope == _STRANDS_SCOPE and attributes.get("gen_ai.operation.name") == "chat":
                usage = normalize_usage(
                    attributes, source="strands", cache_ttl=attributes.get("workshop.cache_write_ttl")
                )
            else:
                continue
            details = langfuse_usage_details(
                usage, model_id=attributes.get("gen_ai.request.model", ""), api=attributes.get("workshop.api", "converse")
            )
            patches[identifier] = OtelSpanPatch(
                delete_attributes=_usage_attributes(attributes),
                set_attributes={
                    _USAGE_DETAILS: json.dumps(details),
                    "langfuse.observation.type": "generation",
                    "langfuse.observation.model.name": attributes.get("gen_ai.request.model", ""),
                    **_unpriced_attributes(details),
                },
            )
        except Exception:
            # No exception text or prompt content is added to telemetry. Keep
            # malformed source meters as metadata for diagnosis, without pricing.
            patch = _suppressed_usage_patch(attributes, "unpriced_invalid_usage")
            patches[identifier] = OtelSpanPatch(
                delete_attributes=patch.delete_attributes,
                set_attributes={
                    **patch.set_attributes,
                    "langfuse.observation.metadata.unpriced_usage": json.dumps(
                        {key: attributes[key] for key in _usage_attributes(attributes)}, default=str
                    ),
                },
            )
    return MaskOtelSpansResult(span_patches=patches)


def resolve_notebook_backend(env_file: str | Path) -> str:
    """Load notebook settings and honor an explicit backend choice in ``.env``.

    Code Editor supplies a default backend through its service and kernelspec.
    Only this participant-selectable setting takes precedence over that default;
    other environment settings, including supplied AWS credentials, are retained.
    ``PYTHON_DOTENV_DISABLED`` disables file loading for isolated execution.
    Calling this function does not initialize telemetry or make AWS requests.
    """
    disabled = os.getenv("PYTHON_DOTENV_DISABLED", "").lower() in {"1", "true", "yes", "t", "y"}
    choice = os.getenv("OBSERVABILITY_BACKEND", "agentcore")
    if not disabled:
        from dotenv import dotenv_values, load_dotenv

        path = Path(env_file)
        values = dotenv_values(path, interpolate=False)
        load_dotenv(path, override=False, interpolate=False)
        if "OBSERVABILITY_BACKEND" in values:
            choice = values["OBSERVABILITY_BACKEND"]
    if not isinstance(choice, str) or choice.strip().lower() not in {"agentcore", "langfuse", "both", "none"}:
        raise ValueError("OBSERVABILITY_BACKEND must be agentcore, langfuse, both, or none")
    return choice.strip().lower()


def _dependency(module: str, package: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"Observability requires optional dependency {package!r}; "
            "install the workshop's pinned observability dependencies."
        ) from exc


class LocalSpanCollector:
    """A bounded OTEL span processor; captures only this Python process.

    It cannot see model/tool spans inside a remotely invoked Runtime. Overflow
    discards oldest spans, increments ``dropped_spans``, and makes document
    conversion fail until clear() so an incomplete local trace is not evaluated.
    """

    def __init__(self, max_spans: int = 2048):
        if isinstance(max_spans, bool) or not isinstance(max_spans, int) or max_spans < 1:
            raise ValueError("max_spans must be a positive integer")
        self._spans: deque = deque(maxlen=max_spans)
        self._lock = threading.Lock()
        self.dropped_spans = 0

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        """Only finished spans are retained."""

    def _on_ending(self, span: Any) -> None:
        """OTEL's pre-end hook: wait for on_end to capture the final ReadableSpan."""

    def on_end(self, span: Any) -> None:
        with self._lock:
            if len(self._spans) == self._spans.maxlen:
                self.dropped_spans += 1
            self._spans.append(span)

    def get_finished_spans(self) -> tuple:
        with self._lock:
            return tuple(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()
            self.dropped_spans = 0

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        self.clear()

    def span_documents(self) -> list[dict]:
        """Convert finished spans; Strands conversion includes conversation logs.

        Strands' events need the AgentCore serializer, not merely span.to_json().
        Raw botocore spans alone are not supported evaluation inputs.
        """
        with self._lock:
            if self.dropped_spans:
                raise RuntimeError("Local spans overflowed; clear and rerun with a larger max_spans")
            spans = tuple(self._spans)
        documents = []
        for span in spans:
            scope = getattr(span.instrumentation_scope, "name", "")
            if scope == "strands.telemetry.tracer":
                serializer = _dependency("bedrock_agentcore.evaluation.span_to_adot_serializer", "bedrock-agentcore")
                converted = serializer.convert_strands_to_adot([span])
                if not converted:
                    raise ValueError("AgentCore could not serialize a captured Strands span")
                documents.extend(_json_document(document) for document in converted)
            else:
                documents.append(_span_document(span))
        return documents

    def native_span_documents(self) -> list[dict]:
        """Keep native events for Recommendations, including multi-step tool calls.

        The evaluation serializer emits separate conversation log records; that
        representation is retained by span_documents() for Evaluate callers.
        """
        with self._lock:
            if self.dropped_spans:
                raise RuntimeError("Local spans overflowed; clear and rerun with a larger max_spans")
            spans = tuple(self._spans)
        return [_span_document(span) for span in spans]


def _json_document(document: dict) -> dict:
    """Preserve observed values while converting OTEL tuple arrays to JSON lists."""
    return json.loads(json.dumps(document, allow_nan=False))


def _span_document(span: Any) -> dict:
    """Serialize an OTEL ReadableSpan to the ADOT document format."""
    if span.start_time is None or span.end_time is None:
        raise ValueError("Only completed spans can be serialized")
    document = {
        "traceId": format(span.context.trace_id, "032x"),
        "spanId": format(span.context.span_id, "016x"),
        "name": span.name,
        "kind": span.kind.name,
        "startTimeUnixNano": span.start_time,
        "endTimeUnixNano": span.end_time,
        "durationNano": span.end_time - span.start_time,
        "status": {"code": span.status.status_code.name},
        "resource": {"attributes": dict(span.resource.attributes)},
        "scope": {"name": span.instrumentation_scope.name},
        "attributes": dict(span.attributes or {}),
        "events": [
            {"name": event.name, "timeUnixNano": event.timestamp, "attributes": dict(event.attributes or {})}
            for event in span.events
        ],
    }
    if span.parent:
        document["parentSpanId"] = format(span.parent.span_id, "016x")
    return _json_document(document)


@dataclass(frozen=True)
class Observability:
    backend: str
    service_name: str
    region: str
    log_group: str
    max_spans: int
    collector: LocalSpanCollector
    provider: Any = None
    langfuse: Any = None
    _closed: bool = False

    def flush(self, timeout_ms: int = 5000) -> bool:
        """Flush trace processors within their SDK timeout; not a delivery proof.

        This flushes Langfuse's attached trace processor too. It does not flush
        non-trace Langfuse API work (datasets/scores) or wait for log ingestion.
        """
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        return True if self.provider is None else self.provider.force_flush(timeout_millis=timeout_ms)

    def shutdown(self) -> None:
        """Stop owned exporters before deleting their log resources.

        This ends telemetry for the process; restart the kernel to initialize it
        again. Closing only the trace provider leaves ADOT's metrics exporter
        alive, which can recreate a log group during process shutdown.
        """
        if self._closed:
            return
        object.__setattr__(self, "_closed", True)
        if self.backend in {"agentcore", "both"}:
            for module, getter in (
                ("opentelemetry.metrics", "get_meter_provider"),
                ("opentelemetry._logs", "get_logger_provider"),
            ):
                provider = getattr(_dependency(module, "opentelemetry-sdk"), getter)()
                close = getattr(provider, "shutdown", None)
                if callable(close):
                    close()
        if self.langfuse is not None:
            self.langfuse.shutdown()
        if self.provider is not None:
            self.provider.shutdown()


def _ensure_notebook_log_stream(region: str, log_group: str) -> None:
    """Reuse the provisioned group and prepare ADOT's stream before export starts.

    Existing streams need only DescribeLogStreams; creation needs CreateLogStream.
    A missing group is a provisioning error, not permission to create a new group
    with different retention/ownership. Only a concurrent stream creation is safe
    to accept; all other AWS errors propagate to the caller.
    """
    boto3 = _dependency("boto3", "boto3")
    with closing(boto3.client("logs", region_name=region)) as logs:
        pages = logs.get_paginator("describe_log_streams").paginate(
            logGroupName=log_group, logStreamNamePrefix=_NOTEBOOK_LOG_STREAM
        )
        for page in pages:
            if any(stream["logStreamName"] == _NOTEBOOK_LOG_STREAM for stream in page.get("logStreams", [])):
                return
        # Another notebook kernel may have created this stream after our read.
        with suppress(logs.exceptions.ResourceAlreadyExistsException):
            logs.create_log_stream(logGroupName=log_group, logStreamName=_NOTEBOOK_LOG_STREAM)


def setup_observability(
    backend: str = "agentcore",
    *,
    service_name: str = "workshop-notebook",
    region: str = "us-east-1",
    log_group: str | None = None,
    max_spans: int = 2048,
) -> Observability:
    """Initialize once; identical calls reuse the handle, changes require restart.

    An explicit log_group wins. Otherwise resolve WORKSHOP_NOTEBOOK_LOG_GROUP,
    then the legacy WORKSHOP_LOG_GROUP alias, then DEFAULT_NOTEBOOK_LOG_GROUP.
    Empty environment values are ignored. Resolution happens at setup time, so
    load .env first. Changing the resolved group after initialization needs restart.

    ``none`` imports no SDKs and installs no provider. Enabled modes capture local
    finished spans. AWS's X-Ray OTLP endpoint delivers traces to aws/spans;
    log_group receives logs/metrics and must already exist. Its runtime-logs stream
    is checked/created before ADOT starts. Readiness errors leave setup retryable
    and preserve the original AWS error. Transaction Search and its X-Ray Logs
    resource policy must be enabled. Langfuse reads LANGFUSE_*
    configuration only when explicitly initialized. Generic OTLP endpoint/headers
    are rejected in AWS modes, never silently rewritten to a Langfuse endpoint.

    An independently initialized global provider (including an auto-instrumented
    Runtime entrypoint) is rejected: this helper owns notebook initialization.
    Runtime entrypoints already configured by ADOT should not call this helper.
    """
    global _state, _initialization_failed
    if backend not in {"agentcore", "langfuse", "both", "none"}:
        raise ValueError("backend must be agentcore, langfuse, both, or none")
    if log_group is None:
        log_group = (
            os.getenv("WORKSHOP_NOTEBOOK_LOG_GROUP") or os.getenv("WORKSHOP_LOG_GROUP") or DEFAULT_NOTEBOOK_LOG_GROUP
        )
    for name, value in (("service_name", service_name), ("region", region), ("log_group", log_group)):
        if not isinstance(value, str) or not value.strip() or any(c in value for c in ",=\r\n"):
            raise ValueError(f"{name} must be nonempty and contain no commas, equals signs or newlines")
    collector = LocalSpanCollector(max_spans)
    configuration = (backend, service_name, region, log_group, max_spans)
    with _lock:
        if _initialization_failed:
            raise RuntimeError("Observability initialization partially failed; restart the Python process")
        if _state is not None:
            if _state._closed:
                raise RuntimeError("Observability was shut down; restart the process before initializing again")
            previous = (_state.backend, _state.service_name, _state.region, _state.log_group, _state.max_spans)
            if configuration != previous:
                raise RuntimeError("Observability configuration is already initialized; restart to change it")
            return _state
        if backend == "none":
            _state = Observability(*configuration, collector)
            return _state

        trace = _dependency("opentelemetry.trace", "opentelemetry-api")
        sdk = _dependency("opentelemetry.sdk.trace", "opentelemetry-sdk")
        resources = _dependency("opentelemetry.sdk.resources", "opentelemetry-sdk")
        if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
            raise RuntimeError("A tracer provider already exists; restart and call setup_observability first")
        if backend in {"agentcore", "both"}:
            for name in (
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "OTEL_EXPORTER_OTLP_HEADERS",
                "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
            ):
                if os.environ.get(name):
                    raise ValueError(f"Unset {name} before AWS telemetry setup; use backend-specific configuration")
            _dependency("amazon.opentelemetry.distro", "aws-opentelemetry-distro")
            auto = _dependency("opentelemetry.instrumentation.auto_instrumentation", "aws-opentelemetry-distro")
        if backend in {"langfuse", "both"}:
            # Importing the module does not create a client/provider.
            langfuse_module = _dependency("langfuse", "langfuse")
            if not all(os.environ.get(key) for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")):
                raise ValueError("Langfuse requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
            if os.environ.get("LANGFUSE_TRACING_ENABLED", "true").lower() == "false":
                raise ValueError("LANGFUSE_TRACING_ENABLED=false conflicts with the selected backend")

        if backend in {"agentcore", "both"}:
            # Fail before changing the environment or initializing any providers.
            _ensure_notebook_log_stream(region, log_group)

        try:
            if backend in {"agentcore", "both"}:
                attributes = dict(
                    item.split("=", 1)
                    for item in os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "").split(",")
                    if "=" in item
                )
                attributes.update({"service.name": service_name, "aws.log.group.names": log_group})
                os.environ.update(
                    {
                        "AGENT_OBSERVABILITY_ENABLED": "true",
                        "OTEL_PYTHON_DISTRO": "aws_distro",
                        "OTEL_PYTHON_CONFIGURATOR": "aws_configurator",
                        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
                        "OTEL_TRACES_EXPORTER": "otlp",
                        "OTEL_SERVICE_NAME": service_name,
                        "OTEL_RESOURCE_ATTRIBUTES": ",".join(f"{k}={v}" for k, v in attributes.items()),
                        "OTEL_EXPORTER_OTLP_LOGS_HEADERS": (
                            f"x-aws-log-group={log_group},x-aws-log-stream={_NOTEBOOK_LOG_STREAM},"
                            "x-aws-metric-namespace=bedrock-agentcore"
                        ),
                        "AWS_GENAI_CONTENT_EXTRACTION_OPT_OUT": "true",
                        "AWS_REGION": region,
                        "AWS_DEFAULT_REGION": region,
                    }
                )
                auto.initialize(swallow_exceptions=False)
                provider = trace.get_tracer_provider()
                if not isinstance(provider, sdk.TracerProvider):
                    raise RuntimeError("ADOT did not install an SDK tracer provider")
            else:
                provider = sdk.TracerProvider(resource=resources.Resource.create({"service.name": service_name}))
                trace.set_tracer_provider(provider)
            langfuse_client = None
            if backend in {"langfuse", "both"}:
                langfuse_client = langfuse_module.Langfuse(
                    tracer_provider=provider, mask_otel_spans=normalize_langfuse_spans
                )
            provider.add_span_processor(collector)
            _state = Observability(*configuration, collector, provider, langfuse_client)
        except Exception as exc:
            _initialization_failed = True
            raise RuntimeError("Observability initialization failed; restart before retrying") from exc
        return _state


def shutdown_observability() -> None:
    """End this process's workshop telemetry before account-resource teardown."""
    if _state is not None:
        _state.shutdown()


def response_span(
    question: str,
    answer: str,
    *,
    session_id: str,
    trace_id: str,
    span_id: str,
    start_time_ns: int,
    end_time_ns: int,
    service_name: str = "workshop-response-check",
) -> dict:
    """Describe an observed Q/A for immediate response quality evaluation.

    Caller supplies correlation IDs and observation times. For a remote response,
    this represents only the caller's observation, never the remote trajectory.
    No model-call usage, TTFT or remote tool execution is inferred.
    """
    if not all(isinstance(text, str) and text for text in (question, answer, session_id)):
        raise ValueError("question, answer and session_id must be nonempty strings")
    for value, size in ((trace_id, 32), (span_id, 16)):
        if (
            not isinstance(value, str)
            or len(value) != size
            or any(c not in "0123456789abcdef" for c in value)
            or int(value, 16) == 0
        ):
            raise ValueError("trace_id and span_id must be nonzero lowercase hex IDs (32/16 characters)")
    if (
        not isinstance(start_time_ns, int)
        or not isinstance(end_time_ns, int)
        or start_time_ns < 0
        or end_time_ns < start_time_ns
    ):
        raise ValueError("Observation times must be ordered nonnegative nanosecond integers")
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "name": "invoke_agent response_check",
        "kind": "INTERNAL",
        "startTimeUnixNano": start_time_ns,
        "endTimeUnixNano": end_time_ns,
        "durationNano": end_time_ns - start_time_ns,
        "status": {"code": "OK"},
        "scope": {"name": CONVERSE_SCOPE},
        "resource": {"attributes": {"service.name": service_name}},
        "attributes": {
            "session.id": session_id,
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.task.input": question,
            "gen_ai.task.output": answer,
            "workshop.response_only": True,
        },
    }


@contextmanager
def converse_observation(
    request: Mapping[str, Any], *, session_id: str | None = None
) -> Iterator[Callable[[dict], dict]]:
    """Record an explicitly invoked Converse response without making an API call.

    The caller invokes its paced Runtime client inside this context, then passes
    the response to the yielded recorder. Recording returns a copy with _trace;
    backend=none returns the original response. Client pacing remains the
    caller's responsibility. Baggage and the active span cover the actual SDK
    call, including its retries, and are released even when that call fails.
    """
    if _state is None:
        raise RuntimeError("Call setup_observability before observing Converse")
    if _state.backend == "none":
        yield lambda response: response
        return
    trace = _dependency("opentelemetry.trace", "opentelemetry-api")
    baggage = _dependency("opentelemetry.baggage", "opentelemetry-api")
    context = _dependency("opentelemetry.context", "opentelemetry-api")
    sid = session_id or str(uuid4())
    token = context.attach(baggage.set_baggage("session.id", sid))
    try:
        with trace.get_tracer(CONVERSE_SCOPE).start_as_current_span("invoke_agent converse") as span:
            attributes = {
                "session.id": sid,
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.request.model": request["modelId"],
                "workshop.model_call": True,
                "workshop.response_only": True,
                "workshop.api": "converse",
                "gen_ai.task.input": "\n".join(
                    block["text"]
                    for message in request["messages"]
                    if message["role"] == "user"
                    for block in message["content"]
                    if "text" in block
                ),
            }
            span.set_attributes(attributes)
            if _state.backend in {"langfuse", "both"}:
                # Langfuse's input/output fields are separate from AWS evaluation
                # attributes. Keep both representations of the observed response.
                span.set_attribute("langfuse.observation.input", attributes["gen_ai.task.input"])
                span.set_attribute("langfuse.trace.input", attributes["gen_ai.task.input"])
            if request.get("system"):
                span.set_attribute(
                    "gen_ai.system_instructions",
                    "\n".join(block["text"] for block in request["system"] if "text" in block),
                )

            def record_response(response: dict) -> dict:
                answer = "".join(
                    block["text"] for block in response["output"]["message"]["content"] if "text" in block
                )
                span.set_attribute("gen_ai.task.output", answer)
                if _state.backend in {"langfuse", "both"}:
                    span.set_attribute("langfuse.observation.output", answer)
                    span.set_attribute("langfuse.trace.output", answer)
                for key, attr in (
                    ("inputTokens", "input_tokens"),
                    ("outputTokens", "output_tokens"),
                    ("cacheReadInputTokens", "cache_read.input_tokens"),
                    ("cacheWriteInputTokens", "cache_creation.input_tokens"),
                ):
                    if key in response.get("usage", {}):
                        span.set_attribute(f"gen_ai.usage.{attr}", response["usage"][key])
                try:
                    raw_usage = response.get("usage", {})
                    # Service-provided details win; request TTL is only a fallback.
                    ttl = None if raw_usage.get("cacheDetails") else uniform_cache_ttl(request)
                    usage = normalize_usage(raw_usage, source="converse", cache_ttl=ttl)
                    span.set_attribute("workshop.normalized_usage", json.dumps(usage.as_dict()))
                    if ttl is not None:
                        span.set_attribute("workshop.cache_write_ttl", ttl)
                except (KeyError, TypeError, ValueError):
                    # A successful model response must survive malformed telemetry.
                    span.set_attribute("workshop.usage_status", "unpriced_invalid_usage")
                if "latencyMs" in response.get("metrics", {}):
                    span.set_attribute("gen_ai.server.request.duration", response["metrics"]["latencyMs"])
                sc = span.get_span_context()
                return {
                    **response,
                    "_trace": {
                        "trace_id": format(sc.trace_id, "032x"),
                        "span_id": format(sc.span_id, "016x"),
                        "session_id": sid,
                    },
                }

            yield record_response
    finally:
        context.detach(token)


def traced_converse(
    client: Any, *, session_id: str | None = None, pacer: InferencePacer | None = None, **kwargs: Any
) -> dict:
    """Converse with pacing and response telemetry for callers using the wrapper.

    Notebooks can instead use converse_observation around a visible SDK call.
    Neither form executes tool requests or measures time to first token.
    """
    with converse_observation(kwargs, session_id=session_id) as record_response:
        return record_response(call_converse(client, pacer=pacer, **kwargs))
