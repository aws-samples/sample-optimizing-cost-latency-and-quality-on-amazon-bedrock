"""Small, dated Bedrock capability catalog and run-scoped model selection.

Only Converse on bedrock-runtime is covered by this catalog. Missing evidence is
UNKNOWN, not UNSUPPORTED. Sources describe service capabilities, not access in a
participant account. Importing this module never creates a client or calls AWS.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol

VERIFIED_ON = "2026-09-24"
RUNTIME = "bedrock-runtime"
CONVERSE = "converse"


class Support(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class UnknownModelError(ValueError):
    """The exact model/profile is absent from the workshop catalog."""


class UnsupportedFeatureError(ValueError):
    """Evidence says the requested feature is unsupported."""


class UnknownCapabilityError(ValueError):
    """There is insufficient evidence to enable the requested feature."""


class SelectionFrozenError(ValueError):
    """A resolved experiment cannot change models or API surfaces."""


class ModelUnavailableError(RuntimeError):
    """An access probe found no available candidate."""


class ProbeFailedError(RuntimeError):
    """A transient or unclassified probe failure cannot justify fallback."""

    def __init__(self, result: ProbeResult):
        self.result = result
        super().__init__(f"{result.status.value}: {result.error_code}: {result.message}")


class ConverseClient(Protocol):
    """Implemented by boto3's bedrock-runtime client and offline test clients."""

    def converse(self, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ModelCapabilities:
    features: Mapping[str, Support]
    sources: tuple[str, ...]
    verified_on: str = VERIFIED_ON
    api: str = CONVERSE
    endpoint: str = RUNTIME
    effort_values: tuple[str, ...] = ()
    default_effort: str | None = None
    thinking_types: tuple[str, ...] = ()
    cache_mode: str = "unknown"
    cache_min_tokens: int | None = None
    cache_ttls: tuple[str, ...] = ()
    max_cache_checkpoints: int | None = None
    service_tiers: tuple[str, ...] = ("standard",)
    output_quota_factor: int | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))

    def support(self, feature: str) -> Support:
        return self.features.get(feature, Support.UNKNOWN)

    def require(self, feature: str) -> None:
        status = self.support(feature)
        if status is Support.UNKNOWN:
            raise UnknownCapabilityError(
                f"{feature} is unverified for {self.endpoint}/{self.api} as of {self.verified_on}"
            )
        if status is Support.UNSUPPORTED:
            raise UnsupportedFeatureError(
                f"{feature} is unsupported for {self.endpoint}/{self.api} as of {self.verified_on}"
            )


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    name: str
    provider: str
    profiles: tuple[str, ...]
    capabilities: ModelCapabilities


def _capabilities(
    *,
    supported: str,
    unsupported: str,
    sources: tuple[str, ...],
    **kwargs: Any,
) -> ModelCapabilities:
    """Construct explicit records; never infer capabilities from an ID prefix."""
    features = dict.fromkeys(supported.split(), Support.SUPPORTED)
    features.update(dict.fromkeys(unsupported.split(), Support.UNSUPPORTED))
    return ModelCapabilities(features=features, sources=sources, **kwargs)


_AWS = "https://docs.aws.amazon.com/bedrock/latest/userguide/"
_CACHING = _AWS + "prompt-caching.html"
_QUOTAS = _AWS + "quotas-token-burndown.html"
_BATCH = _AWS + "batch-inference-supported.html"
_CLAUDE_EFFORT = "https://platform.claude.com/docs/en/build-with-claude/effort"
_GPT_LAUNCH = (
    "https://aws.amazon.com/blogs/machine-learning/"
    "bring-more-intelligence-to-everyday-work-with-gpt-6-sol-and-gpt-6-luna-on-amazon-bedrock/"
)
# Records are intentionally per model. Do not replace these with "Claude 5" or
# "GPT" family defaults: cache minimums, strict tools and Guardrails differ.
_SPECS = (
    ModelSpec(
        "anthropic.claude-sonnet-5",
        "Claude Sonnet 5",
        "anthropic",
        ("global", "us", "eu", "au"),
        _capabilities(
            supported="invocation effort tools forced_tool_choice cache_points guardrails",
            unsupported="temperature top_p top_k structured_output strict_tools batch count_tokens prefill",
            sources=(_AWS + "model-card-anthropic-claude-sonnet-5.html", _CACHING, _CLAUDE_EFFORT, _QUOTAS, _BATCH),
            effort_values=("low", "medium", "high", "xhigh", "max"),
            default_effort="high",
            thinking_types=("adaptive", "disabled"),
            cache_mode="explicit",
            cache_min_tokens=1024,
            cache_ttls=("5m", "1h"),
            max_cache_checkpoints=4,
            output_quota_factor=10,
            notes=(
                "Stop sequences verified only with thinking disabled.",
                "Implicit cache hits are best effort, not guaranteed.",
            ),
        ),
    ),
    ModelSpec(
        "anthropic.claude-opus-5",
        "Claude Opus 5",
        "anthropic",
        ("global", "us"),
        _capabilities(
            supported="invocation effort tools cache_points guardrails batch",
            unsupported="temperature top_p top_k structured_output strict_tools count_tokens prefill",
            sources=(_AWS + "model-card-anthropic-claude-opus-5.html", _CACHING, _CLAUDE_EFFORT, _QUOTAS, _BATCH),
            effort_values=("low", "medium", "high", "xhigh", "max"),
            default_effort="high",
            thinking_types=("adaptive", "disabled"),
            cache_mode="explicit",
            cache_min_tokens=512,
            cache_ttls=("5m", "1h"),
            max_cache_checkpoints=4,
            output_quota_factor=10,
            notes=("xhigh/max require thinking enabled.",),
        ),
    ),
    ModelSpec(
        "anthropic.claude-opus-5-5",
        "Claude Opus 5.5",
        "anthropic",
        ("global", "us", "eu", "au", "jp"),
        _capabilities(
            supported="invocation effort tools cache_points guardrails",
            unsupported="temperature top_p top_k structured_output strict_tools forced_tool_choice batch count_tokens prefill",
            sources=(_AWS + "model-card-anthropic-claude-opus-5-5.html", _CACHING, _CLAUDE_EFFORT, _QUOTAS, _BATCH),
            effort_values=("low", "medium", "high", "xhigh", "max"),
            default_effort="medium",
            thinking_types=("adaptive",),
            cache_mode="explicit",
            cache_min_tokens=512,
            cache_ttls=("5m", "1h"),
            max_cache_checkpoints=4,
            output_quota_factor=10,
            notes=(
                "Thinking is always on; replay thinking blocks unchanged.",
                "History must remain append-only when replaying preserved thinking.",
            ),
        ),
    ),
    ModelSpec(
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        "Claude Haiku 4.5",
        "anthropic",
        ("global", "us"),
        _capabilities(
            supported="invocation temperature top_p top_k tools forced_tool_choice structured_output strict_tools cache_points guardrails batch stop_sequences",
            unsupported="effort",
            sources=(
                _AWS + "model-card-anthropic-claude-haiku-4-5.html",
                _AWS + "claude-messages-structured-outputs.html",
                _CACHING,
                _CLAUDE_EFFORT,
                _QUOTAS,
                _BATCH,
            ),
            thinking_types=("enabled", "disabled"),
            cache_mode="explicit",
            cache_min_tokens=4096,
            cache_ttls=("5m", "1h"),
            max_cache_checkpoints=4,
            service_tiers=("standard", "reserved"),
            output_quota_factor=5,
            notes=(
                "Extended thinking requires budget_tokens; effort is unsupported.",
                "Anthropic platform retirement dates are not Bedrock retirement dates.",
            ),
        ),
    ),
    ModelSpec(
        "anthropic.claude-sonnet-4-6",
        "Claude Sonnet 4.6",
        "anthropic",
        ("global", "us"),
        _capabilities(
            supported="invocation temperature top_p top_k effort tools structured_output strict_tools cache_points guardrails batch count_tokens",
            unsupported="prefill",
            sources=(
                _AWS + "model-card-anthropic-claude-sonnet-4-6.html",
                _AWS + "claude-messages-adaptive-thinking.html",
                _CACHING,
                _CLAUDE_EFFORT,
                _QUOTAS,
                _BATCH,
            ),
            verified_on="2026-09-30",
            effort_values=("low", "medium", "high", "max"),
            default_effort="high",
            thinking_types=("adaptive", "disabled", "enabled"),
            cache_mode="explicit",
            cache_min_tokens=1024,
            cache_ttls=("5m", "1h"),
            max_cache_checkpoints=4,
            service_tiers=("standard", "reserved"),
            output_quota_factor=5,
        ),
    ),
    ModelSpec(
        "anthropic.claude-opus-4-8",
        "Claude Opus 4.8",
        "anthropic",
        ("global", "us"),
        _capabilities(
            supported="invocation effort tools cache_points guardrails",
            unsupported="temperature top_p top_k batch count_tokens prefill structured_output",
            sources=(_AWS + "model-card-anthropic-claude-opus-4-8.html", _CACHING, _CLAUDE_EFFORT, _QUOTAS, _BATCH),
            effort_values=("low", "medium", "high"),
            default_effort="high",
            thinking_types=("adaptive", "disabled"),
            cache_mode="explicit",
            cache_min_tokens=1024,
            cache_ttls=("5m", "1h"),
            max_cache_checkpoints=4,
            output_quota_factor=15,
            notes=("This catalog exposes only verified effort levels.",),
        ),
    ),
    ModelSpec(
        "openai.gpt-6-sol",
        "GPT-6 Sol",
        "openai",
        ("global", "us"),
        _capabilities(
            supported="invocation effort tools guardrails",
            unsupported="temperature top_p top_k cache_points implicit_caching structured_output count_tokens batch",
            sources=(
                _AWS + "model-card-openai-gpt-6-sol.html",
                _GPT_LAUNCH,
                _AWS + "model-parameters-openai.html",
                _CACHING,
            ),
            verified_on="2026-09-30",
            effort_values=("none", "low", "medium", "high", "xhigh", "max"),
            default_effort="medium",
            cache_mode="none",
            output_quota_factor=10,
            notes=(
                "Converse effort/tool observations from research on 2026-09-24 are retained.",
                "Implicit/explicit caching is documented for Responses only, not Converse.",
                "Preserve observed Converse cache meters; they do not verify cache support, TTL or billing.",
                "Retained for historical comparisons; not a default workshop model.",
            ),
        ),
    ),
    ModelSpec(
        "openai.gpt-6-luna",
        "GPT-6 Luna",
        "openai",
        ("global", "us"),
        _capabilities(
            supported="invocation effort tools forced_tool_choice structured_output guardrails",
            unsupported="temperature top_p top_k cache_points implicit_caching count_tokens batch",
            sources=(
                _AWS + "model-card-openai-gpt-6-luna.html",
                _GPT_LAUNCH,
                _AWS + "model-parameters-openai.html",
                _AWS + "structured-output.html",
                _CACHING,
            ),
            verified_on="2026-09-30",
            effort_values=("none", "low", "medium", "high", "xhigh", "max"),
            default_effort="medium",
            cache_mode="none",
            output_quota_factor=10,
            notes=(
                "Structured output/forced tool choice observed in research on 2026-09-24.",
                "Guardrails returned 500s on 2026-09-24; the current AWS card documents Converse support.",
                "Implicit/explicit caching is documented for Responses only, not Converse.",
                "Preserve observed Converse cache meters; they do not verify cache support, TTL or billing.",
                "Retained for historical comparisons; not a default workshop model.",
            ),
        ),
    ),
    ModelSpec(
        "openai.gpt-5.6-sol",
        "GPT-5.6 Sol",
        "openai",
        ("global", "us"),
        _capabilities(
            supported="invocation guardrails",
            unsupported="cache_points implicit_caching structured_output count_tokens",
            sources=(_AWS + "model-card-openai-gpt-56-sol.html", _CACHING),
            verified_on="2026-09-25",
            cache_mode="none",
            output_quota_factor=10,
            notes=(
                "Converse is supported on bedrock-runtime with global/us profiles.",
                "Implicit/explicit caching is documented for Responses only, not Converse.",
                "Converse effort, sampling, client tools and forced/strict tools remain unverified.",
                "Do not inherit request fields from other OpenAI model families.",
            ),
        ),
    ),
    ModelSpec(
        "openai.gpt-5.6-luna",
        "GPT-5.6 Luna",
        "openai",
        ("global", "us", "in"),
        _capabilities(
            supported="invocation effort tools structured_output guardrails",
            unsupported="temperature top_p top_k cache_points implicit_caching batch count_tokens",
            sources=(
                _AWS + "model-card-openai-gpt-56-luna.html",
                _AWS + "model-parameters-openai.html",
                _CACHING,
                _QUOTAS,
            ),
            effort_values=("none", "low", "medium", "high", "xhigh", "max"),
            default_effort="medium",
            verified_on="2026-09-25",
            cache_mode="none",
            output_quota_factor=10,
            notes=(
                "Implicit/explicit caching is documented for Responses only, not Converse.",
                "Preserve any observed Converse cache meters; they do not verify cache support, TTL or billing.",
                "Converse effort uses nested reasoning.effort.",
            ),
        ),
    ),
    ModelSpec(
        "amazon.nova-2-lite-v1:0",
        "Amazon Nova 2 Lite",
        "amazon",
        ("us",),
        _capabilities(
            supported="invocation",
            unsupported="",
            sources=(
                _AWS + "model-card-amazon-nova-2-lite.html",
                _AWS + "service-tiers-inference.html",
                "https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html",
                "https://docs.aws.amazon.com/nova/latest/nova2-userguide/getting-started-console.html",
            ),
            verified_on="2026-09-28",
            service_tiers=("standard", "priority", "flex"),
            notes=(
                "Only the US cross-region profile is catalogued for the small service-tier exercise.",
                "Use us.amazon.nova-2-lite-v1:0 from us-east-1; in-region invocation is unavailable.",
                "Only plain-text invocation and service tiers are exposed here; other controls remain unverified.",
            ),
        ),
    ),
)
MODEL_CATALOG: Mapping[str, ModelSpec] = MappingProxyType({model.model_id: model for model in _SPECS})
DEFAULT_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "workhorse": "global.anthropic.claude-sonnet-5",
        "small": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "deep": "global.anthropic.claude-opus-5",
        "gpt-workhorse": "global.openai.gpt-5.6-sol",
        "gpt-small": "global.openai.gpt-5.6-luna",
    }
)


def get_model(model_id: str) -> ModelSpec:
    """Look up an exact base ID or a documented profile ID; no fuzzy matching."""
    if model_id in MODEL_CATALOG:
        return MODEL_CATALOG[model_id]
    prefix, _, base_id = model_id.partition(".")
    if prefix != "regional" and base_id in MODEL_CATALOG and prefix in MODEL_CATALOG[base_id].profiles:
        return MODEL_CATALOG[base_id]
    raise UnknownModelError(f"Unknown model/profile: {model_id}")


def caps(model_id: str, *, api: str = CONVERSE, endpoint: str = RUNTIME) -> ModelCapabilities:
    model = get_model(model_id)
    if (api, endpoint) != (CONVERSE, RUNTIME):
        raise UnknownCapabilityError(
            f"No capability record for {model.model_id} on {endpoint}/{api}; select and validate that API explicitly."
        )
    return model.capabilities


@dataclass(frozen=True)
class ResolvedModel:
    model_id: str
    base_model_id: str
    provider: str
    profile: str
    capabilities: ModelCapabilities
    api: str = CONVERSE
    endpoint: str = RUNTIME
    region: str | None = None
    alias: str | None = None
    probe_status: ProbeStatus | None = None

    def as_dict(self) -> dict[str, Any]:
        """Metadata for experiment manifests; no credentials or prompt content."""
        return {
            "model_id": self.model_id,
            "base_model_id": self.base_model_id,
            "provider": self.provider,
            "profile": self.profile,
            "api": self.api,
            "endpoint": self.endpoint,
            "region": self.region,
            "alias": self.alias,
            "capabilities_verified_on": self.capabilities.verified_on,
            "probe_status": self.probe_status.value if self.probe_status else None,
        }


def resolve_model(
    model_id: str,
    *,
    api: str = CONVERSE,
    endpoint: str = RUNTIME,
    region: str | None = None,
) -> ResolvedModel:
    """Resolve an exact invocation ID offline; require a profile when catalogued."""
    model = get_model(model_id)
    profile = "regional" if model_id == model.model_id else model_id.split(".", 1)[0]
    if profile not in model.profiles:
        raise UnknownModelError(
            f"{model_id} requires an explicit inference profile on {endpoint}; "
            f"catalogued profiles: {', '.join(model.profiles)}"
        )
    return ResolvedModel(
        model_id,
        model.model_id,
        model.provider,
        profile,
        caps(model_id, api=api, endpoint=endpoint),
        api,
        endpoint,
        region,
    )


class ProbeStatus(StrEnum):
    """Request outcomes only; AVAILABLE does not mean durable model access."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    TRANSIENT_ERROR = "transient_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass(frozen=True)
class ProbeKey:
    account_id: str
    region: str
    model_id: str
    api: str
    endpoint: str
    sdk_version: str
    checked_on: str


@dataclass(frozen=True)
class ProbeResult:
    key: ProbeKey
    status: ProbeStatus
    error_code: str | None = None
    message: str = ""


def probe(
    client: ConverseClient,
    model: str | ResolvedModel,
    *,
    account_id: str,
    region: str,
    sdk_version: str,
    checked_on: str | None = None,
    cache: MutableMapping[ProbeKey, ProbeResult] | None = None,
) -> ProbeResult:
    """Explicit invocation-only probe; caller owns the client and optional cache.

    A success proves acceptance of this small request, not caching, tool use,
    Guardrails, useful output, or Workshop Studio enablement. AWS documents that
    requests can succeed for up to 15 minutes while Marketplace subscription is
    pending, then fail with AccessDeniedException. Even cached AVAILABLE results
    are invocation observations only (see model-access.html), not durable access.
    Transient/unclassified errors are never cached.
    The SDK may retry internally; this helper adds no retries or inference calls.
    """
    from .bedrock import build_converse_request

    selected = resolve_model(model, region=region) if isinstance(model, str) else model
    if not account_id or not region or not sdk_version:
        raise ValueError("account_id, region and sdk_version are required for probe isolation")
    if selected.region is not None and selected.region != region:
        raise ValueError("Probe region differs from the resolved model region")
    client_region = getattr(getattr(client, "meta", None), "region_name", region)
    if client_region != region:
        raise ValueError("Injected client region differs from the probe cache key")
    checked_on = checked_on or date.today().isoformat()
    date.fromisoformat(checked_on)
    endpoint = getattr(getattr(client, "meta", None), "endpoint_url", selected.endpoint)
    key = ProbeKey(account_id, region, selected.model_id, selected.api, endpoint, sdk_version, checked_on)
    if cache is not None and key in cache:
        return cache[key]
    request = build_converse_request(
        selected,
        [{"role": "user", "content": [{"text": "Reply OK."}]}],
        max_tokens=256,
        effort="low" if selected.capabilities.support("effort") is Support.SUPPORTED else None,
    )
    try:
        client.converse(**request)
        result = ProbeResult(
            key,
            ProbeStatus.AVAILABLE,
            message=(
                "Invocation accepted; not proof of durable subscription or Workshop Studio approval. "
                "Marketplace setup can temporarily accept requests for up to 15 minutes. "
                "Other capabilities were not probed."
            ),
        )
    except Exception as exc:
        # Avoid importing botocore merely to classify its public error response.
        error = getattr(exc, "response", {}).get("Error", {})
        code = error.get("Code", type(exc).__name__)
        if code in {"AccessDeniedException", "ResourceNotFoundException"}:
            status = ProbeStatus.UNAVAILABLE
        elif code in {
            "ThrottlingException",
            "TooManyRequestsException",
            "ServiceUnavailableException",
            "InternalServerException",
            "ModelNotReadyException",
            "ModelTimeoutException",
            "ReadTimeoutError",
            "ConnectTimeoutError",
            "EndpointConnectionError",
            "ConnectionClosedError",
            "TimeoutError",
            "ConnectionError",
        }:
            status = ProbeStatus.TRANSIENT_ERROR
        else:
            # ValidationException could be a malformed probe, not a missing model.
            status = ProbeStatus.UNKNOWN_ERROR
        result = ProbeResult(key, status, code, error.get("Message", str(exc)))
    if cache is not None and result.status in {ProbeStatus.AVAILABLE, ProbeStatus.UNAVAILABLE}:
        cache[key] = result
    return result


class ModelRegistry:
    """One registry per experiment. Offline by default, with no automatic fallback.

    Overrides are exact invocation IDs. Optional candidate lists and probe_fn are
    explicit; only UNAVAILABLE results can advance a candidate list. Cross-provider
    fallback also requires opt-in before resolution, and the selection is recorded.
    """

    def __init__(
        self,
        *,
        overrides: Mapping[str, str] | None = None,
        candidates: Mapping[str, Sequence[str]] | None = None,
        probe_fn: Callable[[ResolvedModel], ProbeResult] | None = None,
        region: str | None = None,
        allow_cross_provider_fallback: bool = False,
    ) -> None:
        self._overrides = dict(overrides or {})
        self._candidates = {key: tuple(value) for key, value in (candidates or {}).items()}
        self._probe_fn = probe_fn
        self._region = region
        self._allow_cross_provider_fallback = allow_cross_provider_fallback
        self._resolved: dict[str, ResolvedModel] = {}
        self._lock = RLock()

    def resolve(
        self,
        alias: str = "workhorse",
        *,
        override: str | None = None,
        api: str = CONVERSE,
        endpoint: str = RUNTIME,
    ) -> ResolvedModel:
        with self._lock:
            existing = self._resolved.get(alias)
            if existing:
                if (override is not None and override != existing.model_id) or (api, endpoint) != (
                    existing.api,
                    existing.endpoint,
                ):
                    raise SelectionFrozenError("Create a new ModelRegistry for a changed experiment")
                return existing
            explicit = override if override is not None else self._overrides.get(alias)
            candidates = (
                (explicit,)
                if explicit is not None
                else self._candidates.get(alias, (DEFAULT_ALIASES.get(alias, alias),))
            )
            if not candidates:
                raise ValueError(f"No candidates configured for {alias}")
            primary_provider = get_model(candidates[0]).provider
            for candidate in candidates:
                selected = replace(
                    resolve_model(candidate, api=api, endpoint=endpoint, region=self._region),
                    alias=alias,
                )
                if selected.provider != primary_provider and not self._allow_cross_provider_fallback:
                    raise SelectionFrozenError(
                        "Cross-provider fallback requires explicit opt-in before resolution; "
                        "use a separately labelled comparison."
                    )
                if self._probe_fn:
                    result = self._probe_fn(selected)
                    if result.status is ProbeStatus.UNAVAILABLE:
                        continue
                    if result.status is not ProbeStatus.AVAILABLE:
                        raise ProbeFailedError(result)
                    selected = replace(selected, probe_status=result.status)
                self._resolved[alias] = selected
                return selected
            raise ModelUnavailableError(f"No available model for {alias}: {', '.join(candidates)}")
