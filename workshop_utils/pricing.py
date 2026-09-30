"""Dated token price estimates for exact workshop models and inference profiles.

Prices are USD per million tokens, dated per record. This is an offline snapshot,
not a live billing API. Price each request separately before summing costs: the
GPT long-context threshold applies per request, including cached input.
Quota burndown changes throughput; it is never a token-bill multiplier.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from .bedrock import NormalizedUsage
from .models import MODEL_CATALOG, VERIFIED_ON, ResolvedModel, UnknownModelError, get_model


class UnknownPriceError(ValueError):
    """No price is recorded for this exact model, profile, region or service tier."""


class InferredPriceError(ValueError):
    """An inferred rate needs explicit opt-in."""


class AmbiguousCacheUsageError(ValueError):
    """Cache writes lack enough TTL information to price them accurately."""


@dataclass(frozen=True)
class PriceRecord:
    model_id: str
    profile: str
    input_per_million: float
    output_per_million: float
    cache_read_per_million: float | None
    cache_write_per_million: Mapping[str, float]
    evidence: str
    sources: tuple[str, ...]
    verified_on: str = VERIFIED_ON
    currency: str = "USD"
    service_tier: str = "standard"
    region: str | None = None
    profile_factor: float = 1.0
    tier_factor: float = 1.0
    long_context_threshold: int | None = None
    long_context_input_factor: float = 1.0
    long_context_output_factor: float = 1.0
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_write_per_million", MappingProxyType(dict(self.cache_write_per_million)))

    @property
    def inferred(self) -> bool:
        return self.evidence == "inferred"


_AWS = "https://docs.aws.amazon.com/bedrock/latest/userguide/"
_PRICES = "https://aws.amazon.com/bedrock/pricing/"
_CLAUDE_PRICES = "https://platform.claude.com/docs/en/about-claude/pricing"
_CLAUDE_BEDROCK = "https://platform.claude.com/docs/en/build-with-claude/claude-in-amazon-bedrock"
_CACHING = _AWS + "prompt-caching.html"
_NOVA_PRICES = "https://aws.amazon.com/nova/pricing/"
_PUBLIC_PRICE_MAP = "https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/bedrock/USD/current/bedrock.json"
# Exact US geo-profile input/output rates, not a family-wide tier discount.
_NOVA_TIER_RATES = MappingProxyType({"flex": (0.165, 1.375, 0.5), "priority": (0.5775, 4.8125, 1.75)})

# Exact-model rates captured in the September research. No unknown-model default.
_BASE_PRICES = (
    PriceRecord(
        "anthropic.claude-sonnet-5",
        "global",
        2,
        10,
        0.2,
        {"5m": 2.5, "1h": 4},
        "price-list",
        (_PRICES, _CLAUDE_PRICES, _CLAUDE_BEDROCK),
    ),
    PriceRecord(
        "anthropic.claude-opus-5",
        "global",
        5,
        25,
        0.5,
        {"5m": 6.25, "1h": 10},
        "price-list",
        (_PRICES, _CLAUDE_PRICES, _CLAUDE_BEDROCK),
    ),
    PriceRecord(
        "anthropic.claude-opus-5-5",
        "global",
        4,
        20,
        0.2,
        {"5m": 5, "1h": 8},
        "price-list",
        (_PRICES, _CLAUDE_PRICES, _CLAUDE_BEDROCK),
        notes=("Cache read is 0.05x input, not the usual 0.1x.",),
    ),
    PriceRecord(
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        "global",
        1,
        5,
        0.1,
        {"5m": 1.25, "1h": 2},
        "price-list",
        (_PRICES, _CLAUDE_PRICES, _CLAUDE_BEDROCK),
    ),
    PriceRecord(
        "anthropic.claude-sonnet-4-6",
        "global",
        3,
        15,
        0.3,
        {"5m": 3.75, "1h": 6},
        "price-list",
        (_PRICES, _CLAUDE_PRICES, _CLAUDE_BEDROCK),
    ),
    PriceRecord(
        "anthropic.claude-opus-4-8",
        "global",
        5,
        25,
        0.5,
        {"5m": 6.25, "1h": 10},
        "price-list",
        (_PRICES, _CLAUDE_PRICES, _CLAUDE_BEDROCK),
    ),
    PriceRecord(
        "openai.gpt-6-sol",
        "global",
        2,
        10,
        None,
        {},
        "documented",
        (
            _AWS + "model-card-openai-gpt-6-sol.html",
            "https://developers.openai.com/api/docs/models/gpt-6-sol",
            "https://developers.openai.com/api/docs/guides/amazon-bedrock",
            _CACHING,
        ),
        verified_on="2026-09-30",
        long_context_threshold=272000,
        long_context_input_factor=2,
        long_context_output_factor=1.5,
        notes=(
            "Rates inferred on 2026-09-24 are confirmed by the AWS model card fetched 2026-09-30.",
            "Exact Global CRIS Standard rates; US Geo CRIS includes the documented 10% premium.",
            "Published Global CRIS short-context Responses cache rates are $0.20/M read and $2.50/M write.",
            "The historical 2026-09-24 30m write estimate does not establish a documented TTL.",
            "Cache reads/writes remain unpriced: Converse caching is unsupported and pricing does not identify the API.",
            "Responses cache pricing requires separate API/TTL verification before adding executable cache rates.",
            "Price evidence does not establish Workshop Studio model enablement.",
        ),
    ),
    PriceRecord(
        "openai.gpt-6-luna",
        "global",
        0.1,
        0.5,
        None,
        {},
        "documented",
        (
            _AWS + "model-card-openai-gpt-6-luna.html",
            "https://developers.openai.com/api/docs/models/gpt-6-luna",
            "https://developers.openai.com/api/docs/guides/amazon-bedrock",
            _CACHING,
        ),
        verified_on="2026-09-30",
        long_context_threshold=272000,
        long_context_input_factor=2,
        long_context_output_factor=1.5,
        notes=(
            "Rates inferred on 2026-09-24 are confirmed by the AWS model card fetched 2026-09-30.",
            "Exact Global CRIS Standard rates; US Geo CRIS includes the documented 10% premium.",
            "Published Global CRIS short-context Responses cache rates are $0.01/M read and $0.125/M write.",
            "The historical 2026-09-24 30m write estimate does not establish a documented TTL.",
            "Cache reads/writes remain unpriced: Converse caching is unsupported and pricing does not identify the API.",
            "Responses cache pricing requires separate API/TTL verification before adding executable cache rates.",
            "Price evidence does not establish Workshop Studio model enablement.",
        ),
    ),
    PriceRecord(
        "openai.gpt-5.6-sol",
        "global",
        4,
        20,
        0.4,
        {"30m": 5},
        "documented",
        (_AWS + "model-card-openai-gpt-56-sol.html", _CACHING),
        verified_on="2026-09-25",
        long_context_threshold=272000,
        long_context_input_factor=2,
        long_context_output_factor=1.5,
        notes=(
            "Exact Global CRIS Standard rates; Geo CRIS rates include the documented 10% premium.",
            "Cache rates price reported usage; caching support is documented only for Responses.",
            "Price availability does not establish API capabilities or Workshop Studio enablement.",
        ),
    ),
    PriceRecord(
        "openai.gpt-5.6-luna",
        "global",
        0.2,
        1.2,
        0.02,
        {"30m": 0.25},
        "documented",
        (_AWS + "model-card-openai-gpt-56-luna.html", _CACHING),
        long_context_threshold=272000,
        long_context_input_factor=2,
        long_context_output_factor=1.5,
        notes=(
            "Cache rates are documented for Responses; observed Converse cache billing remains unverified.",
            "Pricing reported cache meters is an estimate, not confirmation of API support, TTL or invoice charges.",
        ),
    ),
    PriceRecord(
        "amazon.nova-2-lite-v1:0",
        "us",
        0.33,
        2.75,
        None,
        {},
        "price-list",
        (_NOVA_PRICES, _PUBLIC_PRICE_MAP, _AWS + "model-card-amazon-nova-2-lite.html"),
        verified_on="2026-09-28",
        region="us-east-1",
        profile_factor=1.1,
        notes=(
            "US geo cross-region rates from the public AWS pricing table, selected region us-east-1.",
            "Global rates differ; this snapshot does not price other profiles or source regions.",
            "Cache pricing is outside this small service-tier exercise; cache meters remain unpriced.",
        ),
    ),
)


def _profile_prices() -> Mapping[tuple[str, str], PriceRecord]:
    records: dict[tuple[str, str], PriceRecord] = {}
    for base in _BASE_PRICES:
        records[(base.model_id, base.profile)] = base
        if base.profile != "global":
            continue
        # Source rules explicitly cover geo profiles. Store each known profile
        # as its own record rather than stripping prefixes during billing.
        for profile in MODEL_CATALOG[base.model_id].profiles:
            if profile == "global":
                continue
            evidence = base.evidence
            notes = base.notes
            if base.model_id in {
                "anthropic.claude-sonnet-5",
                "anthropic.claude-opus-5",
                "anthropic.claude-opus-5-5",
            }:
                evidence = "inferred"
                notes += ("Non-global Claude 5 SKU to geo-profile mapping is inferred.",)
            records[(base.model_id, profile)] = replace(
                base,
                profile=profile,
                input_per_million=base.input_per_million * 1.1,
                output_per_million=base.output_per_million * 1.1,
                cache_read_per_million=(
                    base.cache_read_per_million * 1.1 if base.cache_read_per_million is not None else None
                ),
                cache_write_per_million={ttl: rate * 1.1 for ttl, rate in base.cache_write_per_million.items()},
                profile_factor=1.1,
                evidence=evidence,
                notes=notes,
            )
    return MappingProxyType(records)


PRICE_CATALOG = _profile_prices()


def get_price(
    model_id: str | ResolvedModel,
    *,
    profile: str | None = None,
    allow_inferred: bool = False,
    service_tier: str = "standard",
    region: str | None = None,
) -> PriceRecord:
    """Get an exact rate. A bare model ID requires profile= unless in-region only.

    A prefixed ID already specifies its profile; a conflicting profile is an
    error. 'us' and 'global' are different billing records. The Nova 2 Lite snapshot
    is us-east-1 only and therefore requires a matching region.
    """
    if isinstance(model_id, ResolvedModel):
        if region is not None and model_id.region and region != model_id.region:
            raise UnknownPriceError("Price region conflicts with the resolved model")
        region = region or model_id.region
        model_id = model_id.model_id
    try:
        model = get_model(model_id)
    except UnknownModelError as exc:
        raise UnknownPriceError(str(exc)) from exc
    if model_id != model.model_id:
        actual_profile = model_id.split(".", 1)[0]
        if profile is not None and profile != actual_profile:
            raise UnknownPriceError("Price profile conflicts with the invocation model ID")
        profile = actual_profile
    elif profile is None and model.profiles == ("regional",):
        profile = "regional"
    if profile is None:
        raise UnknownPriceError("A base model ID requires an explicit pricing profile")
    try:
        record = PRICE_CATALOG[(model.model_id, profile)]
    except KeyError as exc:
        raise UnknownPriceError(f"No price for {model.model_id} with profile {profile}") from exc
    if record.region is not None and region != record.region:
        raise UnknownPriceError(f"This price snapshot requires region={record.region!r}")
    if service_tier != "standard":
        if model.model_id != "amazon.nova-2-lite-v1:0" or service_tier not in _NOVA_TIER_RATES:
            raise UnknownPriceError(f"No {service_tier} price for {model.model_id}")
        input_rate, output_rate, factor = _NOVA_TIER_RATES[service_tier]
        record = replace(
            record,
            input_per_million=input_rate,
            output_per_million=output_rate,
            service_tier=service_tier,
            tier_factor=factor,
        )
    if record.inferred and not allow_inferred:
        raise InferredPriceError(
            f"{model_id}/{profile} has inferred prices as of {record.verified_on}; "
            "pass allow_inferred=True to use an explicitly labelled estimate."
        )
    return record


@dataclass(frozen=True)
class CostBreakdown:
    model_id: str
    profile: str
    usage: NormalizedUsage
    price: PriceRecord
    input_cost: float
    output_cost: float
    cache_read_cost: float
    cache_write_cost: float
    long_context: bool

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost + self.cache_read_cost + self.cache_write_cost

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "profile": self.profile,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "cache_read_cost": self.cache_read_cost,
            "cache_write_cost": self.cache_write_cost,
            "total_cost": self.total_cost,
            "currency": self.price.currency,
            "pricing_date": self.price.verified_on,
            "price_evidence": self.price.evidence,
            "price_sources": list(self.price.sources),
            "price_notes": list(self.price.notes),
            "inferred": self.price.inferred,
            "service_tier": self.price.service_tier,
            "long_context": self.long_context,
            "usage": self.usage.as_dict(),
        }


def calculate_cost(
    model_id: str | ResolvedModel,
    usage: NormalizedUsage,
    *,
    profile: str | None = None,
    cache_ttl: str | None = None,
    allow_inferred: bool = False,
    service_tier: str = "standard",
    region: str | None = None,
) -> CostBreakdown:
    """Estimate one request using disjoint token meters and exact-profile prices.

    Unknown cache write TTLs require cache_ttl= or a response-derived breakdown.
    Mixed 5m/1h cache writes are charged separately. Never pass aggregated GPT
    usage here: sum per-request results instead, preserving the 272K threshold.
    """
    if not isinstance(usage, NormalizedUsage):
        raise TypeError("usage must be NormalizedUsage; normalize with an explicit source first")
    record = get_price(
        model_id,
        profile=profile,
        allow_inferred=allow_inferred,
        service_tier=service_tier,
        region=region,
    )
    writes = dict(usage.cache_write_by_ttl)
    if cache_ttl is not None:
        if cache_ttl not in record.cache_write_per_million:
            raise UnknownPriceError(f"No {cache_ttl} cache write price for {record.model_id}")
        if writes and any(ttl != cache_ttl and count for ttl, count in writes.items()):
            raise AmbiguousCacheUsageError("cache_ttl conflicts with the usage breakdown")
        if not writes:
            writes = {cache_ttl: usage.cache_write_tokens}
    if usage.cache_write_tokens and not writes:
        raise AmbiguousCacheUsageError("Cache writes need cache_ttl= or cache_write_by_ttl")
    if usage.cache_read_tokens and record.cache_read_per_million is None:
        raise UnknownPriceError(f"No cache read price for {record.model_id}")
    for ttl, count in writes.items():
        if count and ttl not in record.cache_write_per_million:
            raise UnknownPriceError(f"No {ttl} cache write price for {record.model_id}")
    long_context = (
        record.long_context_threshold is not None and usage.total_input_tokens > record.long_context_threshold
    )
    input_factor = record.long_context_input_factor if long_context else 1
    output_factor = record.long_context_output_factor if long_context else 1
    identifier = model_id.model_id if isinstance(model_id, ResolvedModel) else model_id
    return CostBreakdown(
        identifier,
        record.profile,
        usage,
        record,
        usage.input_tokens * record.input_per_million * input_factor / 1_000_000,
        usage.output_tokens * record.output_per_million * output_factor / 1_000_000,
        usage.cache_read_tokens * (record.cache_read_per_million or 0) * input_factor / 1_000_000,
        sum(count * record.cache_write_per_million.get(ttl, 0) for ttl, count in writes.items())
        * input_factor
        / 1_000_000,
        long_context,
    )
