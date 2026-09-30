"""Create exact workshop model definitions through Langfuse's public API.

Run from the repository root after saving project keys in .env. Imports and
--dry-run are offline. Prices come exclusively from workshop_utils.pricing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_ALIASES = ("workhorse", "small", "deep", "gpt-small", "gpt-workhorse")
# Disjoint canonical meters emitted by observability.langfuse_usage_details.
# Unpriced cache tokens still contribute to the per-request context threshold.
INPUT_USAGE_KEYS = (
    "input",
    "cache_read_input_tokens",
    "cache_write_5m_input_tokens",
    "cache_write_1h_input_tokens",
    "cache_read_30m_input_tokens",
    "cache_write_30m_input_tokens",
    "cache_read_unpriced_input_tokens",
    "cache_write_unpriced_input_tokens",
)
REQUEST_OPTIONS = {"max_retries": 0, "timeout_in_seconds": 20}


class ModelSetupError(ValueError):
    """Setup needs review; existing definitions have not been overwritten."""


def build_definitions() -> list[dict[str, Any]]:
    """Build v3-compatible public API bodies without credentials or network calls."""
    # Support direct execution from a source checkout without an installed root package.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from workshop_utils.models import DEFAULT_ALIASES, get_model
    from workshop_utils.pricing import UnknownPriceError, get_price

    definitions = []
    for model_id in dict.fromkeys(DEFAULT_ALIASES[alias] for alias in DEFAULT_MODEL_ALIASES):
        model = get_model(model_id)
        rate = get_price(model_id)  # Reject unknown/inferred rates; no fallback.
        if not model_id.startswith("global.") or rate.currency != "USD" or rate.service_tier != "standard":
            raise ModelSetupError("Setup requires global profiles with Standard USD prices")
        prices = {"input": rate.input_per_million / 1_000_000, "output": rate.output_per_million / 1_000_000}
        if model.provider == "anthropic":
            read_key, ttls = "cache_read_input_tokens", {"5m", "1h"}
        elif model.provider == "openai":
            read_key, ttls = "cache_read_30m_input_tokens", {"30m"}
        else:
            raise ModelSetupError(f"No canonical cache meters for {model_id}")
        if rate.cache_read_per_million is None or set(rate.cache_write_per_million) != ttls:
            raise UnknownPriceError(f"Incomplete or unsupported cache prices for {model_id}")
        prices[read_key] = rate.cache_read_per_million / 1_000_000
        for ttl, write_rate in rate.cache_write_per_million.items():
            prices[f"cache_write_{ttl}_input_tokens"] = write_rate / 1_000_000
        tiers = [{"name": "Standard", "isDefault": True, "priority": 0, "conditions": [], "prices": prices}]
        if rate.long_context_threshold is not None:
            pattern = "^(" + "|".join(re.escape(key) for key in INPUT_USAGE_KEYS) + ")$"
            tiers.append({
                "name": "Long context", "isDefault": False, "priority": 1,
                "conditions": [{
                    "usageDetailPattern": pattern, "operator": "gt",
                    "value": rate.long_context_threshold, "caseSensitive": True,
                }],
                "prices": {
                    key: price * (
                        rate.long_context_output_factor if key == "output" else rate.long_context_input_factor
                    )
                    for key, price in prices.items()
                },
            })
        definitions.append({
            "modelName": model_id,
            # Postgres accepts escaped dots; hyphens need no escaping outside [].
            "matchPattern": "^" + model_id.replace(".", r"\.") + "$",
            "unit": "TOKENS", "tokenizerId": None, "tokenizerConfig": None,
            "startDate": None, "pricingTiers": tiers,
        })
    return definitions


def _model_request(body: dict[str, Any]) -> Any:
    """Validate API bodies with public SDK types and their Python field names."""
    from langfuse.api import CreateModelRequest, PricingTierInput, PricingTierUsageConditionInput

    return CreateModelRequest(
        model_name=body["modelName"], match_pattern=body["matchPattern"],
        unit=body["unit"], start_date=body["startDate"],
        tokenizer_id=body["tokenizerId"], tokenizer_config=body["tokenizerConfig"],
        pricing_tiers=[
            PricingTierInput(
                name=tier["name"], is_default=tier["isDefault"], priority=tier["priority"],
                conditions=[
                    PricingTierUsageConditionInput(
                        usage_detail_pattern=condition["usageDetailPattern"],
                        operator=condition["operator"], value=condition["value"],
                        case_sensitive=condition["caseSensitive"],
                    )
                    for condition in tier["conditions"]
                ],
                prices=tier["prices"],
            )
            for tier in body["pricingTiers"]
        ],
    )


def _signature(model: Any) -> dict[str, Any]:
    """Compare all behavior fields, ignoring generated model/tier IDs and timestamps."""
    return {
        "name": model.model_name, "pattern": model.match_pattern, "unit": model.unit,
        "start_date": model.start_date, "tokenizer_id": model.tokenizer_id,
        "tokenizer_config": model.tokenizer_config,
        "tiers": sorted([{
            "name": tier.name, "default": tier.is_default, "priority": tier.priority,
            "conditions": [condition.model_dump() for condition in tier.conditions],
            "prices": tier.prices,
        } for tier in model.pricing_tiers or []], key=lambda tier: tier["priority"]),
    }


def setup_models(api: Any, definitions: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Preflight every custom match before creating anything; never update/delete.

    Requests have retries disabled. If a create response is lost, an explicit
    rerun lists the project again and reuses the committed definition.
    """
    desired = [_model_request(body) for body in definitions]
    existing = []
    for page in range(1, 1001):
        response = api.models.list(page=page, limit=100, request_options=REQUEST_OPTIONS)
        if response.meta.page != page:
            raise ModelSetupError("Unexpected pagination; no definitions created")
        existing.extend(response.data)
        if page >= response.meta.total_pages:
            break
    else:
        raise ModelSetupError("Model list exceeded 1000 pages; no definitions created")

    plan = []
    conflicts = []
    for request in desired:
        matches = []
        for model in existing:
            if model.is_langfuse_managed:
                continue  # Project custom models take precedence; managed records are untouched.
            try:
                relevant = (
                    model.model_name == request.model_name
                    or model.match_pattern == request.match_pattern
                    or re.search(model.match_pattern, request.model_name) is not None
                )
            except re.error:
                # Public API patterns use Postgres syntax, which Python cannot always interpret.
                raise ModelSetupError(
                    f"Cannot check custom model {model.id}'s regex; review it in Settings → Models"
                ) from None
            if relevant:
                matches.append(model)
        if len(matches) > 1 or (matches and _signature(matches[0]) != _signature(request)):
            conflicts.append(f"{request.model_name}: existing IDs {', '.join(model.id for model in matches)}")
        else:
            plan.append((request, matches[0] if matches else None))
    if conflicts:
        raise ModelSetupError(
            "Conflicting custom definitions; no definitions created. Review Settings → Models:\n"
            + "\n".join(conflicts)
        )

    result = []
    for request, match in plan:
        if match:
            status = "reused"
        else:
            try:
                match = api.models.create(
                    model_name=request.model_name, match_pattern=request.match_pattern,
                    unit=request.unit, start_date=request.start_date,
                    tokenizer_id=request.tokenizer_id, tokenizer_config=request.tokenizer_config,
                    pricing_tiers=request.pricing_tiers, request_options=REQUEST_OPTIONS,
                )
            except Exception as exc:
                raise ModelSetupError(
                    f"Create outcome uncertain for {request.model_name} ({type(exc).__name__}). "
                    "No automatic retry. Rerun this command to list and reconcile existing definitions."
                ) from None
            persisted = api.models.get(match.id, request_options=REQUEST_OPTIONS)
            if _signature(persisted) != _signature(request):
                raise ModelSetupError(f"Read-back differs for model ID {match.id}; review before continuing")
            status = "created"
        result.append({"model": request.model_name, "id": match.id, "status": status})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print API bodies offline; no keys needed")
    args = parser.parse_args(argv)
    try:
        definitions = build_definitions()
        if args.dry_run:
            print(json.dumps(definitions, indent=2))
            return 0

        import httpx
        from dotenv import load_dotenv
        from langfuse.api import LangfuseAPI

        load_dotenv(ROOT / ".env", override=False)
        base_url = os.environ.get("LANGFUSE_BASE_URL", "").strip().rstrip("/")
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
        url = urlsplit(base_url)
        if (
            url.scheme != "https" or not url.netloc or url.path or url.query or url.fragment
            or url.username or url.password or not public_key or not secret_key
        ):
            raise ModelSetupError("Set LANGFUSE_BASE_URL to an HTTPS server root and both project keys in .env")
        with httpx.Client(timeout=20, follow_redirects=False) as client:
            api = LangfuseAPI(
                base_url=base_url, username=public_key, password=secret_key,
                httpx_client=client, follow_redirects=False,
            )
            result = setup_models(api, definitions)
        created = sum(row["status"] == "created" for row in result)
        reused = sum(row["status"] == "reused" for row in result)
        print(f"Models: {created} created, {reused} reused.")
        return 0
    except ModelSetupError as exc:
        print(f"Setup stopped: {exc}", file=sys.stderr)
    except Exception as exc:
        # Never echo exception bodies, request headers, or environment values.
        print(
            f"Setup stopped ({type(exc).__name__}). Check the locked Langfuse extra, connection, "
            "project permissions and Settings → Models; then rerun to reconcile.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
