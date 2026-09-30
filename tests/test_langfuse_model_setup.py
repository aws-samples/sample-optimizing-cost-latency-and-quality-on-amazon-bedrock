"""Public model API contract tests; synthetic HTTP only, no inference or AWS."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from langfuse.api import LangfuseAPI

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "01-fundamentals/utils/setup_langfuse_models.py"
spec = importlib.util.spec_from_file_location("langfuse_model_setup", SCRIPT)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)
BASE = "https://langfuse.example.test"
CLAUDE_IDS = [
    "global.anthropic.claude-sonnet-5",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "global.anthropic.claude-opus-5",
]
GPT_IDS = ["global.openai.gpt-5.6-luna", "global.openai.gpt-5.6-sol"]
UNPRICED_KEYS = {"cache_read_unpriced_input_tokens", "cache_write_unpriced_input_tokens"}
LEGACY_KEYS = {
    "total", "total_tokens", "total_input_tokens", "input_tokens", "output_tokens",
    "cache_creation_input_tokens", "cache_creation.input_tokens", "cache_read.input_tokens",
    "cache_write_input_tokens", "cacheReadInputTokens", "cacheWriteInputTokens",
}


def request_cost(body, usage):
    """Evaluate the public tier contract for one disjoint usage fixture."""
    tiers = body["pricingTiers"]
    selected = next(tier for tier in tiers if tier["isDefault"])
    for tier in sorted(tiers, key=lambda tier: tier["priority"]):
        if tier["isDefault"]:
            continue
        condition, = tier["conditions"]
        assert condition["operator"] == "gt" and condition["caseSensitive"]
        count = sum(value for key, value in usage.items() if re.search(condition["usageDetailPattern"], key))
        if count > condition["value"]:
            selected = tier
            break
    return sum(usage.get(key, 0) * price for key, price in selected["prices"].items()), selected


def model_response(body, model_id="model-1", **changes):
    """v3.153 model response, including generated tier IDs and legacy prices."""
    body = copy.deepcopy(body)
    prices = body["pricingTiers"][0]["prices"]
    return {
        **body, "id": model_id, "isLangfuseManaged": False,
        "createdAt": "2026-09-28T00:00:00Z",
        "inputPrice": prices.get("input"), "outputPrice": prices.get("output"), "totalPrice": None,
        "prices": {key: {"price": price} for key, price in prices.items()},
        "pricingTiers": [{**tier, "id": f"{model_id}-tier-{i}"} for i, tier in enumerate(body["pricingTiers"])],
        **changes,
    }


class ModelServer:
    def __init__(self, models=()):
        self.models = copy.deepcopy(list(models))
        self.calls = []
        self.fail_after_write = False
        self.reject_write = False
        self.corrupt_read = False

    def handle(self, request):
        self.calls.append((request.method, request.url.path))
        assert request.headers["Authorization"].startswith("Basic ")
        if request.method == "GET" and request.url.path == "/api/public/models":
            # Force pagination, even though the caller allows 100 results.
            page = int(request.url.params["page"])
            return httpx.Response(200, json={
                "data": self.models[page - 1:page],
                "meta": {"page": page, "limit": 1, "totalPages": len(self.models), "totalItems": len(self.models)},
            })
        if request.method == "POST" and request.url.path == "/api/public/models":
            if self.reject_write:
                return httpx.Response(400, json={"message": "invalid model", "error": "InvalidRequestError"})
            body = json.loads(request.content)
            assert not any(m["modelName"] == body["modelName"] for m in self.models if not m["isLangfuseManaged"])
            model = model_response(body, f"model-{len(self.models) + 1}")
            self.models.append(model)
            if self.fail_after_write:
                raise httpx.ReadTimeout("response lost after commit", request=request)
            return httpx.Response(200, json=model)
        if request.method == "GET" and request.url.path.startswith("/api/public/models/"):
            model = copy.deepcopy(next(m for m in self.models if m["id"] == request.url.path.rsplit("/", 1)[1]))
            if self.corrupt_read:
                model["pricingTiers"][0]["prices"]["input"] *= 2
            return httpx.Response(200, json=model)
        raise AssertionError(f"Unexpected mutation/path: {request.method} {request.url.path}")

    def run(self, definitions):
        with httpx.Client(transport=httpx.MockTransport(self.handle)) as client:
            api = LangfuseAPI(base_url=BASE, username="pk-fixture", password="sk-fixture", httpx_client=client)
            return setup.setup_models(api, definitions)

    @property
    def writes(self):
        return [call for call in self.calls if call[0] != "GET"]


def test_defaults_are_all_five_exact_global_profiles_with_all_registry_cache_prices():
    from workshop_utils.pricing import get_price

    definitions = setup.build_definitions()
    assert [body["modelName"] for body in definitions] == CLAUDE_IDS + GPT_IDS
    for body in definitions:
        model_id = body["modelName"]
        pattern = body["matchPattern"]
        assert re.fullmatch(pattern, model_id)
        for other in (model_id[7:], model_id.replace("global.", "us."), model_id + "-extra", "prefix" + model_id):
            assert re.search(pattern, other) is None
        assert body["unit"] == "TOKENS" and body["tokenizerId"] is None
        rate = get_price(model_id)
        read = "cache_read_input_tokens" if model_id in CLAUDE_IDS else "cache_read_30m_input_tokens"
        assert body["pricingTiers"][0]["prices"] == {
            "input": rate.input_per_million / 1_000_000,
            "output": rate.output_per_million / 1_000_000,
            read: rate.cache_read_per_million / 1_000_000,
            **{f"cache_write_{ttl}_input_tokens": price / 1_000_000
               for ttl, price in rate.cache_write_per_million.items()},
        }
        for tier in body["pricingTiers"]:
            assert not (set(tier["prices"]) & (UNPRICED_KEYS | LEGACY_KEYS))
        assert not ({"inputPrice", "outputPrice", "totalPrice", "prices"} & set(body))


@pytest.mark.parametrize("writes", [{"5m": 700}, {"1h": 900}, {"5m": 700, "1h": 900}])
def test_claude_5m_1h_and_mixed_writes_match_per_request_calculator(writes):
    from workshop_utils.bedrock import NormalizedUsage
    from workshop_utils.pricing import calculate_cost

    usage = NormalizedUsage(120, 40, 300, sum(writes.values()), writes)
    canonical = {
        "input": 120, "output": 40, "cache_read_input_tokens": 300,
        **{f"cache_write_{ttl}_input_tokens": count for ttl, count in writes.items()},
    }
    for body in setup.build_definitions()[:3]:
        cost, tier = request_cost(body, canonical)
        assert tier["isDefault"]
        assert cost == pytest.approx(calculate_cost(body["modelName"], usage).total_cost)


def test_exact_registry_records_and_sources_flow_into_payload_without_fallback(monkeypatch):
    from dataclasses import replace

    from workshop_utils import pricing

    original = pricing.get_price
    fetched = []

    def get_price(model_id):
        record = replace(
            original(model_id), input_per_million=123, output_per_million=456,
            cache_read_per_million=7,
            cache_write_per_million={ttl: 80 + index for index, ttl in enumerate(original(model_id).cache_write_per_million)},
            long_context_input_factor=3, long_context_output_factor=4,
        )
        fetched.append((model_id, record))
        return record

    monkeypatch.setattr(pricing, "get_price", get_price)
    definitions = setup.build_definitions()
    assert [model_id for model_id, _ in fetched] == CLAUDE_IDS + GPT_IDS
    for body, (model_id, record) in zip(definitions, fetched, strict=True):
        assert record.sources == original(model_id).sources
        assert record.sources and record.evidence in {"documented", "price-list"} and not record.inferred
        standard = body["pricingTiers"][0]["prices"]
        assert standard["input"] == 123 / 1_000_000
        assert standard["output"] == 456 / 1_000_000
        read = "cache_read_input_tokens" if model_id in CLAUDE_IDS else "cache_read_30m_input_tokens"
        assert standard[read] == 7 / 1_000_000
        for ttl, price in record.cache_write_per_million.items():
            assert standard[f"cache_write_{ttl}_input_tokens"] == price / 1_000_000
        if model_id in GPT_IDS:
            assert body["pricingTiers"][1]["prices"] == {
                key: price * (4 if key == "output" else 3) for key, price in standard.items()
            }


@pytest.mark.parametrize("error_name", ["UnknownPriceError", "InferredPriceError"])
def test_unknown_and_inferred_registry_rates_fail_without_a_fallback(monkeypatch, error_name):
    from workshop_utils import pricing

    def fail(model_id):
        raise getattr(pricing, error_name)("unavailable")

    monkeypatch.setattr(pricing, "get_price", fail)
    with pytest.raises(getattr(pricing, error_name)):
        setup.build_definitions()


def test_unpriced_cache_alias_and_duplicate_aliases_preserve_registry_rules(monkeypatch):
    from workshop_utils import models
    from workshop_utils.pricing import UnknownPriceError

    aliases = dict(models.DEFAULT_ALIASES)
    aliases["workhorse"] = "global.openai.gpt-6-sol"
    monkeypatch.setattr(models, "DEFAULT_ALIASES", aliases)
    # Its uncached rates are documented, but unverified cache rates must not
    # produce an apparently complete Langfuse model definition.
    with pytest.raises(UnknownPriceError, match="cache prices"):
        setup.build_definitions()
    aliases["workhorse"] = aliases["small"]
    names = [body["modelName"] for body in setup.build_definitions()]
    assert len(names) == len(set(names)) == 4


@pytest.mark.parametrize("change", [
    {"cache_read_per_million": None}, {"cache_write_per_million": {"5m": 1}},
    {"cache_write_per_million": {"5m": 1, "1h": 2, "2h": 3}},
])
def test_incomplete_or_unsupported_registry_cache_rates_fail(monkeypatch, change):
    from dataclasses import replace

    from workshop_utils import pricing

    original = pricing.get_price
    monkeypatch.setattr(pricing, "get_price", lambda model_id: replace(original(model_id), **change))
    with pytest.raises(pricing.UnknownPriceError):
        setup.build_definitions()


@pytest.mark.parametrize("count", [272000, 272001])
@pytest.mark.parametrize("read,write", [(0, 0), (100000, 90000)])
def test_gpt_30m_and_long_context_prices_match_per_request_calculator(count, read, write):
    from workshop_utils.bedrock import NormalizedUsage
    from workshop_utils.pricing import calculate_cost

    for body in setup.build_definitions()[3:]:
        standard, long = body["pricingTiers"]
        condition = long["conditions"][0]
        assert condition["operator"] == "gt" and condition["value"] == 272000
        assert condition["caseSensitive"] is True
        assert long["prices"] == {
            key: price * (1.5 if key == "output" else 2) for key, price in standard["prices"].items()
        }
        usage = NormalizedUsage(count - read - write, 10, read, write, {"30m": write} if write else {})
        canonical = {
            "input": usage.input_tokens, "output": 10,
            "cache_read_30m_input_tokens": read, "cache_write_30m_input_tokens": write,
        }
        cost, selected = request_cost(body, canonical)
        assert selected["isDefault"] == (count <= 272000)
        assert cost == pytest.approx(calculate_cost(body["modelName"], usage).total_cost)


@pytest.mark.parametrize("key", setup.INPUT_USAGE_KEYS)
def test_gpt_context_condition_counts_every_disjoint_input_including_unpriced(key):
    for body in setup.build_definitions()[3:]:
        usage = {"input": 272000, "output": 10}
        standard_cost, _ = request_cost(body, usage)
        usage[key] = usage.get(key, 0) + 1
        cost, tier = request_cost(body, usage)
        assert not tier["isDefault"]
        assert cost > standard_cost
        if key in UNPRICED_KEYS:
            assert key not in tier["prices"]
            assert cost == pytest.approx(272000 * tier["prices"]["input"] + 10 * tier["prices"]["output"])


def test_aggregate_aliases_output_and_unknown_meters_cannot_double_count_or_trigger_context_tier():
    for body in setup.build_definitions():
        usage = {"input": 200, "output": 50, "cache_read_unpriced_input_tokens": 20}
        expected, _ = request_cost(body, usage)
        cost, tier = request_cost(body, {**usage, **dict.fromkeys(LEGACY_KEYS, 1_000_000)})
        assert cost == expected and tier["isDefault"]
        if body["modelName"] in GPT_IDS:
            pattern = body["pricingTiers"][1]["conditions"][0]["usageDetailPattern"]
            assert all(re.search(pattern, key) is None for key in LEGACY_KEYS | {"output", "INPUT", "unrelated"})
            assert request_cost(body, {"input": 1, "output": 1_000_000})[1]["isDefault"]


@pytest.mark.parametrize("writes", [{"5m": 700}, {"1h": 900}, {"5m": 700, "1h": 900}, {"30m": 800}])
def test_shared_canonical_usage_matches_each_models_prices_and_preserves_unsupported_counts(writes):
    from workshop_utils.bedrock import NormalizedUsage
    from workshop_utils.observability import langfuse_usage_details
    from workshop_utils.pricing import calculate_cost

    usage = NormalizedUsage(272000, 40, 300, sum(writes.values()), writes)
    for body in setup.build_definitions():
        model_id = body["modelName"]
        api = "converse" if model_id in CLAUDE_IDS else "responses"
        canonical = langfuse_usage_details(usage, model_id=model_id, api=api)
        assert sum(value for key, value in canonical.items() if key != "output") == usage.total_input_tokens
        assert not set(canonical) & LEGACY_KEYS
        cost, tier = request_cost(body, canonical)
        assert set(canonical) <= set(tier["prices"]) | UNPRICED_KEYS
        if set(canonical) & UNPRICED_KEYS:
            assert canonical["cache_write_unpriced_input_tokens"] == usage.cache_write_tokens
            assert not (set(tier["prices"]) & UNPRICED_KEYS)
        else:
            assert cost == pytest.approx(calculate_cost(model_id, usage).total_cost)


def test_gpt_converse_cache_counts_remain_unpriced_but_select_long_context():
    from workshop_utils.bedrock import NormalizedUsage
    from workshop_utils.observability import langfuse_usage_details

    usage = NormalizedUsage(272000, 40, 300, 700, {"30m": 700})
    for body in setup.build_definitions()[3:]:
        canonical = langfuse_usage_details(usage, model_id=body["modelName"], api="converse")
        assert canonical == {
            "input": 272000, "output": 40,
            "cache_read_unpriced_input_tokens": 300, "cache_write_unpriced_input_tokens": 700,
        }
        cost, tier = request_cost(body, canonical)
        assert not tier["isDefault"] and not (set(tier["prices"]) & UNPRICED_KEYS)
        assert cost == pytest.approx(272000 * tier["prices"]["input"] + 40 * tier["prices"]["output"])


def test_public_sdk_ingestion_serializes_all_canonical_meters_without_cost_overrides():
    from langfuse.api import CreateGenerationBody, IngestionEvent_GenerationCreate

    usage = dict.fromkeys(set(setup.INPUT_USAGE_KEYS) | {"output"}, 7)
    calls = []

    def handle(request):
        assert request.url.path == "/api/public/ingestion"
        body = json.loads(request.content)["batch"][0]["body"]
        assert body["usageDetails"] == usage
        assert "costDetails" not in body and "usage" not in body
        calls.append(request)
        return httpx.Response(207, json={"successes": [{"id": "fixture-event", "status": 201}], "errors": []})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        api = LangfuseAPI(base_url=BASE, username="pk-fixture", password="sk-fixture", httpx_client=client)
        response = api.ingestion.batch(batch=[IngestionEvent_GenerationCreate(
            id="fixture-event", timestamp="2026-09-28T00:00:00Z",
            body=CreateGenerationBody(
                id="fixture-generation", trace_id="fixture-trace", model="synthetic-fixture",
                usage_details=usage,
            ),
        )], request_options=setup.REQUEST_OPTIONS)
    assert len(calls) == 1 and response.errors == []


def test_public_sdk_payload_validation_roundtrip_pagination_and_idempotent_rerun(capsys):
    from langfuse.api import CreateModelRequest, PricingTierInput, PricingTierUsageConditionInput

    definitions = setup.build_definitions()
    for body in definitions:
        request = setup._model_request(body)
        assert isinstance(request, CreateModelRequest)
        assert request.model_name == body["modelName"]
        assert all(isinstance(tier, PricingTierInput) for tier in request.pricing_tiers)
        assert all(isinstance(condition, PricingTierUsageConditionInput)
                   for tier in request.pricing_tiers for condition in tier.conditions)
    server = ModelServer()
    first = server.run(definitions)
    before = copy.deepcopy(server.models)
    second = server.run(definitions)
    assert len(server.writes) == 5 and server.models == before
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert {row["status"] for row in first} == {"created"}
    assert {row["status"] for row in second} == {"reused"}
    for body, model in zip(definitions, server.models, strict=True):
        assert [{key: value for key, value in tier.items() if key != "id"} for tier in model["pricingTiers"]] == body["pricingTiers"]
    assert capsys.readouterr().out == ""


def test_invalid_public_sdk_body_fails_before_any_api_call():
    from pydantic import ValidationError

    definitions = setup.build_definitions()
    definitions[-1]["pricingTiers"][1]["conditions"][0]["operator"] = "unsupported"
    server = ModelServer()
    with pytest.raises(ValidationError):
        server.run(definitions)
    assert server.calls == []


@pytest.mark.parametrize("difference", ["price", "pattern", "tokenizer", "date", "tier", "usage"])
def test_any_conflicting_behavior_stops_entire_plan_before_first_create(difference):
    definitions = setup.build_definitions()
    model = model_response(definitions[-1])
    if difference == "price":
        model["pricingTiers"][0]["prices"]["input"] *= 2
    elif difference == "pattern":
        model["matchPattern"] = ".*"
    elif difference == "tokenizer":
        model["tokenizerId"] = "claude"
    elif difference == "date":
        model["startDate"] = "2026-09-27T00:00:00Z"
    elif difference == "tier":
        model["pricingTiers"][0]["name"] = "Another tier"
    else:
        model["pricingTiers"][0]["prices"]["cache_read.input_tokens"] = model["pricingTiers"][0]["prices"].pop("cache_read_30m_input_tokens")
    server = ModelServer([model])
    before = copy.deepcopy(server.models)
    with pytest.raises(setup.ModelSetupError, match="Conflicting"):
        server.run(definitions)
    assert not server.writes and server.models == before


def test_old_aggregate_cache_definition_stops_plan_without_migration():
    definitions = setup.build_definitions()
    model = model_response(definitions[0], model_id="old-project-definition")
    prices = model["pricingTiers"][0]["prices"]
    prices["cache_creation.input_tokens"] = prices.pop("cache_write_5m_input_tokens")
    prices.pop("cache_write_1h_input_tokens")
    server = ModelServer([model])
    before = copy.deepcopy(server.models)
    with pytest.raises(setup.ModelSetupError, match="old-project-definition"):
        server.run(definitions)
    assert not server.writes and server.models == before


def test_foreign_broad_pattern_and_duplicate_custom_matches_are_not_overridden():
    definitions = setup.build_definitions()
    foreign = model_response(definitions[0], model_id="foreign", modelName="Other team", matchPattern=r"^global\.")
    server = ModelServer([model_response(definitions[0]), foreign])
    with pytest.raises(setup.ModelSetupError, match="foreign"):
        server.run(definitions)
    assert not server.writes


def test_uninterpretable_custom_postgres_regex_requires_review():
    definitions = setup.build_definitions()
    server = ModelServer([model_response(definitions[0], modelName="Other", matchPattern=r"\mword\M")])
    with pytest.raises(setup.ModelSetupError, match="Cannot check"):
        server.run(definitions)
    assert not server.writes


def test_managed_and_unrelated_records_are_preserved():
    definitions = setup.build_definitions()
    existing = [
        model_response(definitions[0], model_id="builtin", isLangfuseManaged=True),
        model_response(definitions[0], model_id="unrelated", modelName="other", matchPattern="^other$"),
    ]
    server = ModelServer(existing)
    server.run(definitions)
    assert len(server.writes) == 5 and server.models[:2] == existing


def test_timeout_after_committed_create_is_not_retried_and_rerun_reconciles():
    definitions = setup.build_definitions()[:1]
    server = ModelServer()
    server.fail_after_write = True
    with pytest.raises(setup.ModelSetupError, match="outcome uncertain"):
        server.run(definitions)
    assert len(server.writes) == 1
    assert server.run(definitions)[0]["status"] == "reused"
    assert len(server.writes) == 1


def test_rejected_create_is_not_retried():
    server = ModelServer()
    server.reject_write = True
    with pytest.raises(setup.ModelSetupError, match="No automatic retry"):
        server.run(setup.build_definitions())
    assert len(server.writes) == 1


def test_changed_server_readback_stops_without_overwrite():
    server = ModelServer()
    server.corrupt_read = True
    with pytest.raises(setup.ModelSetupError, match="Read-back differs"):
        server.run(setup.build_definitions())
    assert len(server.writes) == 1


def test_import_and_dry_run_work_from_another_directory_without_network_or_keys(tmp_path):
    result = subprocess.run([
        sys.executable, "-I", "-c",
        """
import runpy, socket, sys
from unittest.mock import patch
with patch.object(socket.socket, "connect", side_effect=AssertionError("offline")):
    module = runpy.run_path(sys.argv[1])
    assert module["main"](["--dry-run"]) == 0
""", str(SCRIPT),
    ], cwd=tmp_path, text=True, capture_output=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)) == 5
    assert result.stderr == ""


def test_cli_defaults_report_only_created_and_reused_counts(monkeypatch, capsys):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("LANGFUSE_BASE_URL", BASE)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-private-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-private-test")
    server = ModelServer()
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(server.handle), **kwargs))
    assert setup.main([]) == 0
    out = capsys.readouterr()
    assert out.out == "Models: 5 created, 0 reused.\n" and out.err == ""
    assert setup.main([]) == 0
    out = capsys.readouterr()
    assert out.out == "Models: 0 created, 5 reused.\n" and out.err == ""
    assert len(server.writes) == 5


def test_cli_help_has_no_participant_pricing_decisions(capsys):
    with pytest.raises(SystemExit) as exc:
        setup.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr()
    assert "--dry-run" in out.out
    assert all(flag not in out.out for flag in ("--include-gpt", "--usage-style", "--cache-write-ttl"))


def test_cli_sanitizes_server_errors_and_does_not_follow_redirects(monkeypatch, capsys):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("LANGFUSE_BASE_URL", BASE)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-private-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-private-test")
    requests = []
    original = httpx.Client

    def handle(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "https://other.test"}, json={"secret": "sk-private-test"})

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    assert setup.main([]) == 1
    out = capsys.readouterr()
    assert "sk-private-test" not in out.err + out.out
    assert len(requests) == 1 and requests[0].url.host == "langfuse.example.test"
