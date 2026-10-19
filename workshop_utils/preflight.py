"""Explicit, principal-scoped workshop model preflight. No AWS calls on import.

These small billable probes establish invocation acceptance only. Run them before
initializing experiment telemetry. A saved run must reuse its resolved IDs; this
module does not silently replace models during an experiment.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import asdict, dataclass
from datetime import date
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from .models import ModelRegistry, ProbeKey, ProbeResult, ProbeStatus, ResolvedModel, get_model, probe
from .pricing import get_price

DEFAULT_CANDIDATES = MappingProxyType(
    {
        "workhorse": ("global.anthropic.claude-sonnet-5", "global.anthropic.claude-sonnet-4-6"),
        "small": ("global.anthropic.claude-haiku-4-5-20251001-v1:0",),
    }
)


class PrincipalProbeCache(MutableMapping[ProbeKey, ProbeResult]):
    """Persist only available/unavailable outcomes, with the STS principal in each key."""

    def __init__(self, path: Path, principal_arn: str, *, refresh: bool = False):
        self.path, self.principal_arn, self.refresh = Path(path), principal_arn, refresh
        self.entries = []
        if self.path.exists():
            payload = json.loads(self.path.read_text())
            if payload.get("schema_version") != 1 or not isinstance(payload.get("entries"), list):
                raise ValueError(f"Unsupported probe-cache format: {self.path}; archive it before refreshing")
            self.entries = payload["entries"]
        self.hits: set[ProbeKey] = set()

    def __getitem__(self, key: ProbeKey) -> ProbeResult:
        if self.refresh:
            raise KeyError(key)
        scope = {"principal_arn": self.principal_arn, **asdict(key)}
        for entry in reversed(self.entries):
            if entry.get("key") == scope:
                status = ProbeStatus(entry["status"])
                if status not in {ProbeStatus.AVAILABLE, ProbeStatus.UNAVAILABLE}:
                    raise KeyError(key)
                self.hits.add(key)
                return ProbeResult(key, status, entry.get("error_code"), entry.get("message", ""))
        raise KeyError(key)

    def __setitem__(self, key: ProbeKey, value: ProbeResult) -> None:
        if value.key != key or value.status not in {ProbeStatus.AVAILABLE, ProbeStatus.UNAVAILABLE}:
            raise ValueError("Only matching available/unavailable results may enter the persistent cache")
        scope = {"principal_arn": self.principal_arn, **asdict(key)}
        self.entries = [entry for entry in self.entries if entry.get("key") != scope]
        self.entries.append(
            {"key": scope, "status": value.status.value, "error_code": value.error_code, "message": value.message}
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + "." + uuid4().hex + ".tmp")
        temporary.write_text(json.dumps({"schema_version": 1, "entries": self.entries}, indent=2) + "\n")
        temporary.replace(self.path)

    def __delitem__(self, key: ProbeKey) -> None:
        raise TypeError("Use refresh=True for an explicit new probe after permissions change")

    def __iter__(self) -> Iterator[ProbeKey]:
        for entry in self.entries:
            if entry["key"].get("principal_arn") == self.principal_arn:
                yield ProbeKey(**{key: value for key, value in entry["key"].items() if key != "principal_arn"})

    def __len__(self) -> int:
        return sum(1 for _ in self)


@dataclass(frozen=True)
class PreflightResult:
    account_id: str
    principal_arn: str
    region: str
    sdk_version: str
    checked_on: str
    models: Mapping[str, ResolvedModel]
    probes: tuple[dict[str, Any], ...]
    pricing: Mapping[str, dict[str, Any]]

    def __post_init__(self):
        object.__setattr__(self, "models", MappingProxyType(dict(self.models)))
        object.__setattr__(self, "pricing", MappingProxyType(dict(self.pricing)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "principal_arn": self.principal_arn,
            "region": self.region,
            "sdk_version": self.sdk_version,
            "checked_on": self.checked_on,
            "models": {role: model.as_dict() for role, model in self.models.items()},
            "probes": list(self.probes),
            "pricing": dict(self.pricing),
            "scope": "billable invocation-acceptance preflight; excluded from baseline; not capability verification",
        }


def preflight_models(
    bedrock_client: Any,
    *,
    sts_client: Any,
    region: str,
    overrides: Mapping[str, str] | None = None,
    allow_gpt: bool = False,
    allow_inferred: bool = False,
    cache_path: str | Path = ".models.json",
    sdk_version: str | None = None,
    checked_on: str | None = None,
    refresh: bool = False,
) -> PreflightResult:
    """Resolve once using explicit candidates; transient failures never cause fallback.

    Default: Sonnet 5 → Sonnet 4.6; Haiku 4.5 for small. Overrides are exact
    invocation IDs. Any OpenAI override needs allow_gpt=True; inferred prices
    separately require allow_inferred=True. Identity comes from the injected STS
    client; account alone is insufficient for permission-sensitive probe caching.
    """
    overrides = dict(overrides or {})
    if set(overrides) - set(DEFAULT_CANDIDATES):
        raise ValueError("Preflight accepts only workhorse and small overrides")
    checked_on = checked_on or date.today().isoformat()
    date.fromisoformat(checked_on)
    sdk_version = sdk_version or f"boto3-{version('boto3')}/botocore-{version('botocore')}"
    identity = sts_client.get_caller_identity()
    account, principal = identity["Account"], identity["Arn"]
    if not principal.startswith("arn:") or principal.split(":")[4] != account:
        raise ValueError("STS principal ARN and account do not match")
    candidates = {
        role: (overrides[role],) if role in overrides else defaults for role, defaults in DEFAULT_CANDIDATES.items()
    }
    # Validate opt-ins and price knowledge before spending any inference tokens.
    for options in candidates.values():
        for model_id in options:
            if get_model(model_id).provider == "openai" and not allow_gpt:
                raise ValueError("OpenAI comparisons require explicit allow_gpt=True")
            get_price(model_id, allow_inferred=allow_inferred, region=region)
    cache = PrincipalProbeCache(Path(cache_path), principal, refresh=refresh)
    evidence = []

    def invocation_probe(selected):
        result = probe(
            bedrock_client,
            selected,
            account_id=account,
            region=region,
            sdk_version=sdk_version,
            checked_on=checked_on,
            cache=cache,
        )
        evidence.append(
            {
                "principal_arn": principal,
                **asdict(result.key),
                "status": result.status.value,
                "error_code": result.error_code,
                "message": result.message,
                "cache_hit": result.key in cache.hits,
            }
        )
        return result

    registry = ModelRegistry(candidates=candidates, region=region, probe_fn=invocation_probe)
    models = {role: registry.resolve(role) for role in DEFAULT_CANDIDATES}
    pricing = {}
    for role, selected in models.items():
        price = get_price(selected, allow_inferred=allow_inferred, region=region)
        pricing[role] = {
            "model_id": selected.model_id,
            "profile": selected.profile,
            "verified_on": price.verified_on,
            "sources": list(price.sources),
            "evidence": price.evidence,
            "inferred": price.inferred,
            "currency": price.currency,
            "input_per_million": price.input_per_million,
            "output_per_million": price.output_per_million,
            "cache_read_per_million": price.cache_read_per_million,
            "cache_write_per_million": dict(price.cache_write_per_million),
        }
    return PreflightResult(account, principal, region, sdk_version, checked_on, models, tuple(evidence), pricing)
