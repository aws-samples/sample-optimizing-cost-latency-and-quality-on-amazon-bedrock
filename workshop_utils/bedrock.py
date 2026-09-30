"""Validated Converse requests and explicit token-accounting semantics.

This helper implements one API. It never reroutes to Responses, changes providers,
drops controls, creates clients, or retries inference. Use provider-native APIs in
separately labelled experiments when their features are needed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal

from .models import (
    CONVERSE,
    RUNTIME,
    ConverseClient,
    ResolvedModel,
    UnknownCapabilityError,
    UnsupportedFeatureError,
    resolve_model,
)
from .pacing import InferencePacer, call_converse


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class NormalizedUsage:
    """Disjoint billed meters; output includes reasoning, never add it again.

    input_tokens is ALWAYS uncached input. cache_write_by_ttl, when present, is
    a breakdown of cache_write_tokens, not an additional meter. Empty breakdowns
    preserve unknown TTLs rather than assuming a cheap write rate.
    """

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_by_ttl: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            _count(getattr(self, name), name)
        details = dict(self.cache_write_by_ttl)
        for ttl, count in details.items():
            if ttl not in {"5m", "1h", "30m"}:
                raise ValueError(f"Unknown cache TTL: {ttl}")
            _count(count, f"cache_write_by_ttl[{ttl}]")
        if details and sum(details.values()) != self.cache_write_tokens:
            raise ValueError("Cache TTL details must sum to cache_write_tokens")
        object.__setattr__(self, "cache_write_by_ttl", MappingProxyType(details))

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_write_by_ttl": dict(self.cache_write_by_ttl),
            "total_input_tokens": self.total_input_tokens,
            "total_tokens": self.total_tokens,
        }


def _attribute(attrs: Mapping[str, Any], names: tuple[str, ...], *, required: bool = False) -> int:
    values = [_count(attrs[name], name) for name in names if name in attrs]
    if not values:
        if required:
            raise ValueError(f"Missing usage field: {names[0]}")
        return 0
    if len(set(values)) != 1:
        raise ValueError(f"Conflicting usage fields: {names}")
    return values[0]


def normalize_usage(
    usage: Mapping[str, Any],
    *,
    source: Literal["converse", "strands"],
    cache_ttl: str | None = None,
) -> NormalizedUsage:
    """Normalize a Converse usage dict or Strands 1.57+ OTEL span attributes.

    Strands means cache-inclusive telemetry attributes, NOT a raw Bedrock usage
    dict or an arbitrary Strands version's accumulatedUsage. Missing required
    meters and impossible inclusive totals raise rather than being clamped to 0.
    Converse cacheDetails follows the AWS CacheDetail inputTokens/ttl schema.
    """
    details: dict[str, int] = {}
    if source == "converse":
        inp = _attribute(usage, ("inputTokens",), required=True)
        out = _attribute(usage, ("outputTokens",), required=True)
        read = _attribute(usage, ("cacheReadInputTokens",))
        write = _attribute(usage, ("cacheWriteInputTokens",))
        for detail in usage.get("cacheDetails", []):
            ttl = detail["ttl"]
            details[ttl] = details.get(ttl, 0) + _count(detail["inputTokens"], "cacheDetails.inputTokens")
    elif source == "strands":
        inp = _attribute(usage, ("gen_ai.usage.input_tokens",), required=True)
        out = _attribute(usage, ("gen_ai.usage.output_tokens",), required=True)
        read = _attribute(
            usage,
            (
                "gen_ai.usage.cache_read.input_tokens",
                "gen_ai.usage.cache_read_input_tokens",
            ),
        )
        write = _attribute(
            usage,
            (
                "gen_ai.usage.cache_creation.input_tokens",
                "gen_ai.usage.cache_write_input_tokens",
            ),
        )
        inp -= read + write
        if inp < 0:
            raise ValueError("Strands cache counts exceed its cache-inclusive input total")
    else:
        raise ValueError("source must be 'converse' or 'strands'; semantics cannot be guessed")
    if cache_ttl is not None:
        if cache_ttl not in {"5m", "1h", "30m"}:
            raise ValueError(f"Unknown cache TTL: {cache_ttl}")
        if details and any(ttl != cache_ttl and count for ttl, count in details.items()):
            raise ValueError("cache_ttl conflicts with the response cacheDetails")
        if not details:
            details = {cache_ttl: write}
    return NormalizedUsage(inp, out, read, write, details)


def _selected(model: str | ResolvedModel) -> ResolvedModel:
    selected = resolve_model(model) if isinstance(model, str) else model
    if (selected.api, selected.endpoint) != (CONVERSE, RUNTIME):
        raise UnknownCapabilityError("This request helper implements Converse on bedrock-runtime only")
    return selected


def _range(value: Any, name: str, minimum: float, maximum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _effort_key(model: ResolvedModel) -> str:
    """Field names for the catalogued Claude/GPT effort controls."""
    if model.provider == "anthropic":
        return "output_config"
    return "reasoning"


def _validate_thinking(
    thinking: Mapping[str, Any],
    model: ResolvedModel,
    effort: str | None,
    max_tokens: int,
) -> None:
    if not isinstance(thinking, Mapping):
        raise ValueError("thinking must be an object")
    capability = model.capabilities
    kind = thinking.get("type")
    if kind not in capability.thinking_types:
        if model.provider == "openai":
            raise UnsupportedFeatureError(f"{model.model_id} uses {_effort_key(model)}, not Claude thinking")
        raise UnsupportedFeatureError(f"Thinking type {kind!r} is unsupported for {model.model_id}")
    allowed = {"type"}
    if kind == "adaptive":
        allowed.add("display")
        if "display" in thinking and thinking["display"] not in {"omitted", "summarized"}:
            raise ValueError("thinking.display must be omitted or summarized")
    if kind == "enabled":
        allowed.add("budget_tokens")
        budget = _count(thinking.get("budget_tokens"), "thinking.budget_tokens")
        if not 1024 <= budget < max_tokens:
            raise ValueError("thinking.budget_tokens must be >=1024 and below max_tokens")
    if set(thinking) - allowed:
        raise UnsupportedFeatureError(f"Unsupported thinking fields: {sorted(set(thinking) - allowed)}")
    if kind == "disabled" and effort in {"xhigh", "max"}:
        raise UnsupportedFeatureError("xhigh/max effort cannot be used with thinking disabled")


def _validate_provider_fields(
    fields: dict[str, Any],
    model: ResolvedModel,
    *,
    max_tokens: int,
) -> None:
    effort_key = _effort_key(model)
    allowed = {effort_key}
    if model.provider == "anthropic":
        allowed |= {"thinking", "top_k"}
    unknown = set(fields) - allowed
    if unknown:
        raise UnsupportedFeatureError(
            f"Unsupported provider fields for {model.model_id}: {sorted(unknown)}. Effort belongs in {effort_key}."
        )
    effort = None
    if effort_key in fields:
        model.capabilities.require("effort")
        value = fields[effort_key]
        if not isinstance(value, Mapping) or set(value) != {"effort"}:
            raise UnsupportedFeatureError(f"Only {effort_key}.effort is implemented")
        effort = value["effort"]
        if effort not in model.capabilities.effort_values:
            raise UnsupportedFeatureError(f"Unsupported effort {effort!r} for {model.model_id}")
    if "thinking" in fields:
        _validate_thinking(fields["thinking"], model, effort, max_tokens)
    if "top_k" in fields:
        model.capabilities.require("top_k")
        top_k = _count(fields["top_k"], "top_k")
        if not 1 <= top_k <= 500:
            raise ValueError("top_k must be between 1 and 500")


def _cache_blocks(request: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    # Inspect only API content blocks, never schema keys or arbitrary tool data.
    # Bedrock assembles tools, then system, then messages.
    blocks = list(request.get("toolConfig", {}).get("tools", []))
    blocks.extend(request.get("system", []))
    for message in request["messages"]:
        blocks.extend(message["content"])
    return [block for block in blocks if "cachePoint" in block]


def _validate_cache(request: Mapping[str, Any], model: ResolvedModel) -> None:
    blocks = _cache_blocks(request)
    if not blocks:
        return
    capability = model.capabilities
    capability.require("cache_points")
    if len(blocks) > (capability.max_cache_checkpoints or 0):
        raise ValueError("Too many cache checkpoints for this model")
    seen_short_ttl = False
    for block in blocks:
        point = block["cachePoint"]
        if set(block) != {"cachePoint"}:
            raise ValueError("cachePoint must be its own content block")
        if not isinstance(point, Mapping):
            raise ValueError("cachePoint must be an object")
        if point.get("type") != "default" or set(point) - {"type", "ttl"}:
            raise UnsupportedFeatureError("Only default cachePoint with an optional ttl is implemented")
        if point.get("ttl", "5m") not in capability.cache_ttls:
            raise UnsupportedFeatureError(f"Unsupported cache TTL for {model.model_id}")
        if point.get("ttl", "5m") == "1h" and seen_short_ttl:
            raise ValueError("1h cache checkpoints must precede 5m checkpoints in prompt assembly order")
        seen_short_ttl |= point.get("ttl", "5m") == "5m"
    # Token minimums require the actual model tokenizer; never estimate them
    # from character counts or turn a successful response into a claimed hit.


def build_converse_request(
    model: str | ResolvedModel,
    messages: Sequence[Mapping[str, Any]],
    *,
    system: Sequence[Mapping[str, Any]] | None = None,
    max_tokens: int = 1024,
    effort: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    stop_sequences: Sequence[str] | None = None,
    thinking: Mapping[str, Any] | None = None,
    tool_config: Mapping[str, Any] | None = None,
    output_config: Mapping[str, Any] | None = None,
    guardrail_config: Mapping[str, Any] | None = None,
    service_tier: str | None = None,
    additional_model_request_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return boto3 kwargs after validation. Caller-owned inputs are not mutated.

    output_config is Converse's outputConfig (textFormat only). Native Claude
    output_config.effort and GPT-5.6 reasoning.effort go into AMRF via effort=,
    subject to exact-model capability verification.
    Supplied AMRF is validated too; it is not an escape hatch around capabilities.
    service_tier="standard" uses the boto3 wire value "default".
    """
    selected = _selected(model)
    capability = selected.capabilities
    capability.require("invocation")
    if _count(max_tokens, "max_tokens") == 0:
        raise ValueError("max_tokens must be positive")
    if not messages or isinstance(messages, (str, bytes)):
        raise ValueError("messages must be a nonempty sequence of Converse messages")
    request: dict[str, Any] = {
        "modelId": selected.model_id,
        "messages": deepcopy(list(messages)),
        "inferenceConfig": {"maxTokens": max_tokens},
    }
    for message in request["messages"]:
        if message.get("role") not in {"user", "assistant"} or not message.get("content"):
            raise ValueError("Each Converse message needs a user/assistant role and content")
    if request["messages"][-1]["role"] == "assistant":
        capability.require("prefill")
    for name, value, wire in (("temperature", temperature, "temperature"), ("top_p", top_p, "topP")):
        if value is not None:
            capability.require(name)
            _range(value, name, 0, 1)
            request["inferenceConfig"][wire] = value
    if selected.provider == "anthropic" and temperature is not None and top_p is not None:
        raise UnsupportedFeatureError("Select either temperature or top_p for these Claude models")
    fields = deepcopy(dict(additional_model_request_fields or {}))
    effort_key = _effort_key(selected)
    if effort is not None:
        if effort_key in fields:
            raise ValueError("effort is specified both explicitly and in provider fields")
        fields[effort_key] = {"effort": effort}
    for name, value in (("thinking", thinking), ("top_k", top_k)):
        if value is not None:
            if name in fields:
                raise ValueError(f"{name} is specified twice")
            fields[name] = deepcopy(value)
    _validate_provider_fields(fields, selected, max_tokens=max_tokens)
    if fields.get("thinking", {}).get("type") in {"adaptive", "enabled"} and (
        any(value is not None for value in (temperature, top_p, top_k)) or "top_k" in fields
    ):
        raise UnsupportedFeatureError("Sampling controls with enabled thinking are not supported here")
    if stop_sequences is not None:
        sonnet_disabled = (
            selected.base_model_id == "anthropic.claude-sonnet-5"
            and fields.get("thinking", {}).get("type") == "disabled"
        )
        if not sonnet_disabled:
            capability.require("stop_sequences")
        if (
            isinstance(stop_sequences, str)
            or not stop_sequences
            or any(not isinstance(value, str) or not value for value in stop_sequences)
        ):
            raise ValueError("stop_sequences must be a nonempty sequence of nonempty strings")
        request["inferenceConfig"]["stopSequences"] = list(stop_sequences)
    if fields:
        request["additionalModelRequestFields"] = fields
    if system is not None:
        request["system"] = deepcopy(list(system))
    if tool_config is not None:
        capability.require("tools")
        if set(tool_config) - {"tools", "toolChoice"}:
            raise UnsupportedFeatureError("Only tools and toolChoice are implemented in tool_config")
        choice = tool_config.get("toolChoice", {"auto": {}})
        if len(choice) != 1 or not set(choice) <= {"auto", "tool", "any"}:
            raise UnsupportedFeatureError("Unknown Converse toolChoice")
        if "tool" in choice or "any" in choice:
            capability.require("forced_tool_choice")
        for tool in tool_config.get("tools", []):
            if "strict" in tool.get("toolSpec", {}):
                capability.require("strict_tools")
        if fields.get("thinking", {}).get("type") == "enabled" and set(choice) != {"auto"}:
            raise UnsupportedFeatureError("Extended thinking requires auto tool choice")
        request["toolConfig"] = deepcopy(dict(tool_config))
    if output_config is not None:
        capability.require("structured_output")
        if set(output_config) != {"textFormat"}:
            raise UnsupportedFeatureError("Only Converse outputConfig.textFormat is implemented")
        request["outputConfig"] = deepcopy(dict(output_config))
    if guardrail_config is not None:
        capability.require("guardrails")
        request["guardrailConfig"] = deepcopy(dict(guardrail_config))
    if service_tier is not None:
        if service_tier not in capability.service_tiers:
            raise UnsupportedFeatureError(f"Unsupported service tier {service_tier} for {selected.model_id}")
        request["serviceTier"] = {"type": "default" if service_tier == "standard" else service_tier}
    _validate_cache(request, selected)
    return request


@dataclass(frozen=True)
class ConverseResult:
    model: ResolvedModel
    request: Mapping[str, Any]
    response: Mapping[str, Any]
    usage: NormalizedUsage

    @property
    def text(self) -> str:
        content = self.response.get("output", {}).get("message", {}).get("content", [])
        return "".join(block["text"] for block in content if "text" in block)

    @property
    def latency_ms(self) -> float | None:
        return self.response.get("metrics", {}).get("latencyMs")


def converse(
    client: ConverseClient,
    model: str | ResolvedModel,
    messages: Sequence[Mapping[str, Any]],
    *,
    pacer: InferencePacer | None = None,
    **controls: Any,
) -> ConverseResult:
    """Make one helper call, pacing SDK send attempts/retries by default.

    Pass an explicit pacer for a custom scope or an offline clock. Pacing does
    not alter model controls, usage or service latency measurements.
    """
    selected = _selected(model)
    client_region = getattr(getattr(client, "meta", None), "region_name", None)
    if selected.region and client_region and selected.region != client_region:
        raise ValueError("Injected client region differs from the resolved experiment")
    if selected.region is None and client_region:
        selected = replace(selected, region=client_region)
    request = build_converse_request(selected, messages, **controls)
    response = call_converse(client, pacer=pacer, **request)
    # An explicit single-TTL request gives an unambiguous write attribution.
    # Otherwise keep raw cache meters and use only service-provided cacheDetails.
    # Implicit cache metadata (including historical observations) cannot establish
    # the TTL or billing of this response, or override explicit response details.
    ttls = {block["cachePoint"].get("ttl", "5m") for block in _cache_blocks(request)}
    cache_ttl = next(iter(ttls)) if len(ttls) == 1 else None
    usage = normalize_usage(response["usage"], source="converse", cache_ttl=cache_ttl)
    return ConverseResult(selected, request, response, usage)
