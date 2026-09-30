"""Normalize supplied ADOT span documents to one row per attributed model call.

The input is shared by local collectors and CloudWatch/Langfuse OTEL exports.
This module does not query backends or import their SDKs. Strands 1.57 input
tokens include cache read/write tokens; Converse inputTokens excludes them.
Unknown values remain None, including unmeasured TTFT and unpriced cost.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from .observability import CONVERSE_SCOPE

_STRANDS = "strands.telemetry.tracer"
_BOTOCORE = "opentelemetry.instrumentation.botocore.bedrock-runtime"
_COLUMNS = [
    "backend",
    "service_name",
    "session_id",
    "trace_id",
    "span_id",
    "parent_span_id",
    "scope",
    "model_id",
    "start_time",
    "n_llm_calls",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "ttft_ms",
    "llm_ms",
    "cost_usd",
]
_TOKEN_ATTRIBUTES = {
    "input_tokens": ("gen_ai.usage.input_tokens",),
    "output_tokens": ("gen_ai.usage.output_tokens",),
    "cache_read_tokens": ("gen_ai.usage.cache_read.input_tokens", "gen_ai.usage.cache_read_input_tokens"),
    "cache_write_tokens": ("gen_ai.usage.cache_creation.input_tokens", "gen_ai.usage.cache_write_input_tokens"),
}


def _number(value: Any, name: str, *, integer: bool = False) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    if integer and int(value) != value:
        raise ValueError(f"{name} must be an integer")
    return int(value) if integer else value


def normalize_usage(attributes: Mapping, scope: str) -> dict[str, int | None]:
    """Return uncached input/output/cache counts without inventing missing usage.

    Missing cache counts in workshop Converse/Strands usage are zero: these fields
    are optional and omitted when unused. Botocore instrumentation does not emit
    cache metrics, so missing counts there remain unknown. A span with no usage
    has unknown counts. Explicit null cache counts remain unknown. Inclusive
    Strands usage less than reported cache counts is invalid.
    """
    usage = {}
    has_usage = any(key in attributes for keys in _TOKEN_ATTRIBUTES.values() for key in keys)
    for name, aliases in _TOKEN_ATTRIBUTES.items():
        found = [attributes[key] for key in aliases if key in attributes]
        if found and any(value != found[0] for value in found):
            raise ValueError(f"Conflicting aliases for {name}")
        cache_default = 0 if has_usage and scope != _BOTOCORE and name.startswith("cache_") else None
        value = found[0] if found else cache_default
        usage[name] = _number(value, name, integer=True)
    if scope == _STRANDS:
        counts = [usage[name] for name in ("input_tokens", "cache_read_tokens", "cache_write_tokens")]
        if any(value is None for value in counts):
            usage["input_tokens"] = None
        else:
            uncached = counts[0] - counts[1] - counts[2]
            if uncached < 0:
                raise ValueError("Strands inclusive input tokens are smaller than cache usage")
            usage["input_tokens"] = uncached
    elif scope not in {CONVERSE_SCOPE, _BOTOCORE}:
        raise ValueError(f"Unsupported token semantics for scope {scope!r}")
    return usage


def _rank(document: Mapping) -> int:
    scope = document.get("scope", {}).get("name")
    attributes = document.get("attributes", {})
    operation = attributes.get("gen_ai.operation.name")
    if scope == _STRANDS and operation == "chat":
        return 3
    if (
        scope == CONVERSE_SCOPE
        and operation == "invoke_agent"
        and (
            attributes.get("workshop.model_call")
            or any(key in attributes for key in _TOKEN_ATTRIBUTES["input_tokens"])
        )
    ):
        return 2
    if scope == _BOTOCORE and operation == "chat":
        return 1
    return 0


def _span_index(documents: Sequence[Mapping]) -> dict[tuple[str, str], dict]:
    spans = {}
    for document in documents:
        if not isinstance(document, Mapping):
            raise ValueError("Span documents must be mappings")
        # Conversation log records intentionally share trace/span IDs with spans.
        if "startTimeUnixNano" not in document:
            continue
        trace_id, span_id = document.get("traceId"), document.get("spanId")
        if not trace_id or not span_id:
            raise ValueError("A metric span must have traceId and spanId")
        key = (trace_id, span_id)
        previous = spans.get(key)
        if previous is None:
            spans[key] = dict(document)
            continue
        # Unified/split copies may add fields, but conflicting observations must
        # not choose whichever happened to be listed first.
        merged = dict(previous)
        for name, value in document.items():
            if name == "attributes":
                attributes = dict(merged.get(name, {}))
                for attr, attr_value in value.items():
                    if attr in attributes and attributes[attr] != attr_value:
                        raise ValueError(f"Conflicting attributes for span {key}")
                    attributes[attr] = attr_value
                merged[name] = attributes
            elif name in merged and merged[name] != value:
                raise ValueError(f"Conflicting copies of span {key}")
            else:
                merged[name] = value
        spans[key] = merged
    return spans


def metrics_records(
    span_docs: Sequence[Mapping],
    *,
    backend: str = "memory",
    pricing: Callable[..., float | None] | None = None,
) -> list[dict]:
    """Return one normalized record per model call, suitable for groupby/sums.

    Repeated traceId/spanId copies count once. A nested botocore model span is
    suppressed only if its actual ancestor is a richer Strands chat or workshop
    Converse span. Independent calls in the same trace are retained. Aggregate
    invoke_agent usage is never added to chat usage. Missing ancestry cannot prove
    deduplication: supply the full trace rather than a filtered list of chat spans.

    pricing is optional and called with model_id plus the four token counts.
    It should return USD or None when unknown; KeyError means unpriced model.
    Other pricing errors propagate. No cost is computed with missing model/usage.
    Values cover only supplied calls: retries must be included by the caller.
    """
    if backend not in {"memory", "cloudwatch", "langfuse"}:
        raise ValueError("backend must be memory, cloudwatch, or langfuse")
    spans = _span_index(span_docs)
    records = []
    for (trace_id, span_id), document in spans.items():
        rank = _rank(document)
        if not rank:
            continue
        parent_id = document.get("parentSpanId")
        ancestor_id = parent_id
        visited = {span_id}
        duplicate = False
        while ancestor_id:
            if ancestor_id in visited:
                raise ValueError(f"Cycle in trace {trace_id} parent links")
            visited.add(ancestor_id)
            ancestor = spans.get((trace_id, ancestor_id))
            if ancestor is None:
                break
            if _rank(ancestor) > rank:
                duplicate = True
            ancestor_id = ancestor.get("parentSpanId")
        if duplicate:
            continue
        attributes = document.get("attributes", {})
        scope = document["scope"]["name"]
        usage = normalize_usage(attributes, scope)
        start = _number(document.get("startTimeUnixNano"), "startTimeUnixNano", integer=True)
        end = _number(document.get("endTimeUnixNano"), "endTimeUnixNano", integer=True)
        if start is None or end is None:
            raise ValueError("Model-call metrics require finished spans")
        if end is not None and start is not None and end < start:
            raise ValueError("Span end time precedes start time")
        model_id = attributes.get("gen_ai.request.model") or attributes.get("gen_ai.response.model")
        cost = None
        if pricing is not None and model_id and all(value is not None for value in usage.values()):
            try:
                cost = pricing(model_id=model_id, **usage)
            except KeyError:
                cost = None
            cost = _number(cost, "cost_usd")
        records.append(
            {
                "backend": backend,
                "service_name": document.get("resource", {}).get("attributes", {}).get("service.name"),
                "session_id": attributes.get("session.id"),
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": parent_id,
                "scope": scope,
                "model_id": model_id,
                "start_time": datetime.fromtimestamp(start / 1e9, UTC) if start is not None else None,
                "n_llm_calls": 1,
                **usage,
                "ttft_ms": _number(attributes.get("gen_ai.server.time_to_first_token"), "ttft_ms"),
                # These SDK attributes are explicitly milliseconds. DurationNano is
                # not substituted: span overhead is not provider-reported latency.
                "llm_ms": _number(attributes.get("gen_ai.server.request.duration"), "llm_ms"),
                "cost_usd": cost,
            }
        )
    return sorted(
        records,
        key=lambda row: (
            row["start_time"] or datetime.min.replace(tzinfo=UTC),
            row["trace_id"],
            row["span_id"],
        ),
    )


def metrics_dataframe(
    span_docs: Sequence[Mapping],
    *,
    backend: str = "memory",
    pricing: Callable[..., float | None] | None = None,
) -> Any:
    """Return a pandas DataFrame if installed, otherwise the same list of records."""
    records = metrics_records(span_docs, backend=backend, pricing=pricing)
    try:
        import pandas as pd
    except ImportError:
        return records
    return pd.DataFrame.from_records(records, columns=_COLUMNS)
