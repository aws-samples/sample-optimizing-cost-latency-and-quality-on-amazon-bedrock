"""Offline capability and model-resolution contracts."""

from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace

from workshop_utils.models import (
    DEFAULT_ALIASES,
    MODEL_CATALOG,
    ModelRegistry,
    ModelUnavailableError,
    ProbeFailedError,
    ProbeStatus,
    SelectionFrozenError,
    Support,
    UnknownCapabilityError,
    UnknownModelError,
    UnsupportedFeatureError,
    caps,
    get_model,
    probe,
    resolve_model,
)

SONNET = "global.anthropic.claude-sonnet-5"
HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
GPT = "global.openai.gpt-6-luna"
SOL = "global.openai.gpt-5.6-sol"
NOVA = "us.amazon.nova-2-lite-v1:0"


class ServiceError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": "Probe fixture"}}


class FakeClient:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.calls = []
        self.meta = SimpleNamespace(region_name="us-east-1", endpoint_url="https://runtime.example")

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else {}
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class ModelTests(unittest.TestCase):
    def test_sonnet_46_supports_documented_max_effort_without_xhigh(self):
        record = caps("global.anthropic.claude-sonnet-4-6")
        self.assertEqual(record.effort_values, ("low", "medium", "high", "max"))
        self.assertEqual(record.default_effort, "high")
        self.assertEqual(record.verified_on, "2026-09-30")
        self.assertIn(
            "https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-adaptive-thinking.html",
            record.sources,
        )

    def test_gpt_6_cards_scope_capabilities_to_converse(self):
        for variant in ("sol", "luna"):
            with self.subTest(variant=variant):
                model_id = f"global.openai.gpt-6-{variant}"
                record = caps(model_id)
                self.assertEqual((record.endpoint, record.api), ("bedrock-runtime", "converse"))
                self.assertEqual(record.verified_on, "2026-09-30")
                self.assertIn(
                    f"https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-{variant}.html",
                    record.sources,
                )
                record.require("invocation")
                record.require("guardrails")
                record.require("effort")
                self.assertEqual(
                    record.support("structured_output"),
                    Support.SUPPORTED if variant == "luna" else Support.UNSUPPORTED,
                )
                self.assertEqual(record.cache_mode, "none")
                self.assertIsNone(record.cache_min_tokens)
                self.assertEqual(record.cache_ttls, ())
                self.assertIsNone(record.max_cache_checkpoints)
                for feature in ("cache_points", "implicit_caching", "count_tokens"):
                    with self.subTest(feature=feature), self.assertRaises(UnsupportedFeatureError):
                        record.require(feature)
                self.assertEqual(record.service_tiers, ("standard",))
                self.assertEqual(record.output_quota_factor, 10)
                self.assertEqual(get_model(model_id).profiles, ("global", "us"))
                self.assertNotIn(model_id, DEFAULT_ALIASES.values())
                with self.assertRaises(UnknownCapabilityError):
                    caps(model_id, api="responses")
                with self.assertRaises(UnknownCapabilityError):
                    caps(model_id, endpoint="bedrock-mantle")

    def test_luna_converse_has_no_documented_cache_mode_minimum_or_ttl(self):
        record = caps("global.openai.gpt-5.6-luna")
        self.assertEqual(record.verified_on, "2026-09-25")
        self.assertEqual(record.cache_mode, "none")
        self.assertIsNone(record.cache_min_tokens)
        self.assertEqual(record.cache_ttls, ())
        for feature in ("cache_points", "implicit_caching"):
            with self.subTest(feature=feature), self.assertRaises(UnsupportedFeatureError):
                record.require(feature)
        record.require("effort")
        self.assertIn("none", record.effort_values)

    def test_nova_tier_example_exposes_only_the_verified_invocation_contract(self):
        selected = resolve_model(NOVA, region="us-east-1")
        self.assertEqual(selected.base_model_id, "amazon.nova-2-lite-v1:0")
        self.assertEqual(selected.provider, "amazon")
        self.assertEqual(selected.profile, "us")
        self.assertEqual(get_model(NOVA).profiles, ("us",))
        record = selected.capabilities
        self.assertEqual(record.verified_on, "2026-09-28")
        record.require("invocation")
        self.assertEqual(record.service_tiers, ("standard", "priority", "flex"))
        self.assertEqual(record.effort_values, ())
        self.assertIsNone(record.default_effort)
        self.assertEqual(record.cache_mode, "unknown")
        self.assertIsNone(record.output_quota_factor)
        for feature in ("effort", "temperature", "top_p", "top_k", "tools", "cache_points", "batch"):
            with self.subTest(feature=feature), self.assertRaises(UnknownCapabilityError):
                record.require(feature)
        with self.assertRaises(UnknownModelError):
            resolve_model("amazon.nova-2-lite-v1:0")

    def test_tier_example_does_not_change_default_model_selection(self):
        self.assertEqual(
            dict(DEFAULT_ALIASES),
            {
                "workhorse": SONNET,
                "small": HAIKU,
                "deep": "global.anthropic.claude-opus-5",
                "gpt-workhorse": SOL,
                "gpt-small": "global.openai.gpt-5.6-luna",
            },
        )

    def test_sol_uses_verified_converse_profile_and_conservative_capabilities(self):
        self.assertEqual(resolve_model(SOL).base_model_id, "openai.gpt-5.6-sol")
        self.assertEqual(get_model(SOL).profiles, ("global", "us"))
        record = caps(SOL)
        self.assertEqual(record.verified_on, "2026-09-25")
        record.require("invocation")
        record.require("guardrails")
        for feature in ("effort", "tools", "forced_tool_choice", "strict_tools", "temperature", "top_p", "top_k"):
            with self.subTest(feature=feature), self.assertRaises(UnknownCapabilityError):
                record.require(feature)
        for feature in ("cache_points", "implicit_caching", "structured_output", "count_tokens"):
            with self.subTest(feature=feature), self.assertRaises(UnsupportedFeatureError):
                record.require(feature)
        self.assertEqual(record.cache_mode, "none")
        self.assertIsNone(record.cache_min_tokens)
        self.assertEqual(record.cache_ttls, ())
        self.assertEqual(record.effort_values, ())
        for model_id in ("openai.gpt-5.6-sol", "in.openai.gpt-5.6-sol", "eu.openai.gpt-5.6-sol"):
            with self.subTest(model_id=model_id), self.assertRaises(UnknownModelError):
                resolve_model(model_id)

    def test_exact_cache_minimums_and_sources(self):
        expected = {
            SONNET: 1024,
            "global.anthropic.claude-opus-5": 512,
            "global.anthropic.claude-opus-5-5": 512,
            HAIKU: 4096,
            "global.anthropic.claude-opus-4-8": 1024,
            "global.anthropic.claude-sonnet-4-6": 1024,
        }
        for model_id, minimum in expected.items():
            with self.subTest(model_id=model_id):
                record = caps(model_id)
                self.assertEqual(record.cache_min_tokens, minimum)
                expected_date = "2026-09-30" if model_id.endswith("claude-sonnet-4-6") else "2026-09-24"
                self.assertEqual(record.verified_on, expected_date)
                self.assertTrue(all(url.startswith("https://") for url in record.sources))
        self.assertTrue(all(spec.capabilities.sources for spec in MODEL_CATALOG.values()))

    def test_unknown_is_distinct_from_unsupported(self):
        self.assertEqual(caps(GPT).support("strict_tools"), Support.UNKNOWN)
        with self.assertRaises(UnknownCapabilityError):
            caps(GPT).require("strict_tools")
        with self.assertRaises(UnsupportedFeatureError):
            caps(GPT).require("cache_points")
        self.assertEqual(caps(GPT).support("invented_feature"), Support.UNKNOWN)

    def test_new_api_is_unverified_not_inherited(self):
        with self.assertRaises(UnknownCapabilityError):
            caps(SONNET, api="responses")
        with self.assertRaises(UnknownCapabilityError):
            caps(SONNET, endpoint="bedrock-mantle")

    def test_exact_ids_no_substring_or_profile_guess(self):
        for value in (
            "sonnet-5",
            SONNET + "-preview",
            "global.openai.gpt-6-nonexistent",
            "apac.anthropic.claude-sonnet-5",
            "eu.openai.gpt-5.6-sol",
            "regional.anthropic.claude-sonnet-5",
        ):
            with self.subTest(value=value), self.assertRaises(UnknownModelError):
                get_model(value)
        with self.assertRaises(UnknownModelError):
            ModelRegistry().resolve("anthropic.claude-sonnet-5")

    def test_explicit_override_and_frozen_resolution(self):
        registry = ModelRegistry(overrides={"workhorse": HAIKU}, region="us-east-1")
        selected = registry.resolve("workhorse")
        self.assertEqual(selected.model_id, HAIKU)
        self.assertIs(selected, registry.resolve("workhorse"))
        self.assertEqual(selected.as_dict()["profile"], "global")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            selected.model_id = SONNET
        with self.assertRaises(TypeError):
            selected.capabilities.features["effort"] = Support.SUPPORTED
        with self.assertRaises(SelectionFrozenError):
            registry.resolve("workhorse", override=SONNET)
        with self.assertRaises(SelectionFrozenError):
            registry.resolve("workhorse", api="responses")
        self.assertEqual(ModelRegistry().resolve(override=SONNET).model_id, SONNET)

    def test_empty_explicit_override_is_not_replaced_by_default(self):
        with self.assertRaises(UnknownModelError):
            ModelRegistry().resolve(override="")


class ProbeTests(unittest.TestCase):
    def call_probe(self, client, model=SONNET, **overrides):
        params = {
            "account_id": "111111111111",
            "region": "us-east-1",
            "sdk_version": "boto3-1.43.101/botocore-1.43.101",
            "checked_on": "2026-09-24",
        }
        params.update(overrides)
        return probe(client, model, **params)

    def test_probe_uses_small_valid_budget_and_scopes_cache(self):
        client, cache = FakeClient(), {}
        first = self.call_probe(client, cache=cache)
        self.assertIs(first, self.call_probe(client, cache=cache))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["inferenceConfig"], {"maxTokens": 256})
        self.assertEqual(first.status, ProbeStatus.AVAILABLE)
        for overrides in (
            {"account_id": "222222222222"},
            {"sdk_version": "new-sdk"},
            {"checked_on": "2026-09-25"},
            {"model": HAIKU},
        ):
            self.call_probe(client, cache=cache, **overrides)
        client.meta.endpoint_url = "https://other-runtime.example"
        self.call_probe(client, cache=cache)
        client.meta.region_name = "us-west-2"
        self.call_probe(client, cache=cache, region="us-west-2")
        self.assertEqual(len(client.calls), 7)
        self.assertEqual({key.api for key in cache}, {"converse"})
        self.assertEqual(len(cache), 7)

    def test_sol_probe_does_not_inherit_unverified_effort_fields(self):
        client = FakeClient()
        result = self.call_probe(client, model=SOL)
        self.assertEqual(result.status, ProbeStatus.AVAILABLE)
        self.assertNotIn("additionalModelRequestFields", client.calls[0])
        self.assertIn("not proof of durable subscription", result.message)

    def test_transient_failure_is_never_negative_cached(self):
        for code in (
            "ThrottlingException",
            "InternalServerException",
            "ModelNotReadyException",
            "ServiceUnavailableException",
            "ReadTimeoutError",
        ):
            with self.subTest(code=code):
                client, cache = FakeClient([ServiceError(code), {}]), {}
                self.assertEqual(self.call_probe(client, cache=cache).status, ProbeStatus.TRANSIENT_ERROR)
                self.assertEqual(cache, {})
                self.assertEqual(self.call_probe(client, cache=cache).status, ProbeStatus.AVAILABLE)
                self.assertEqual(len(client.calls), 2)

    def test_access_denied_is_cached_as_unavailable_not_transient(self):
        client, cache = FakeClient([ServiceError("AccessDeniedException")]), {}
        result = self.call_probe(client, cache=cache)
        self.assertEqual(result.status, ProbeStatus.UNAVAILABLE)
        self.assertEqual(result.error_code, "AccessDeniedException")
        self.assertIs(self.call_probe(client, cache=cache), result)
        self.assertEqual(len(client.calls), 1)

    def test_validation_error_does_not_mean_model_unavailable(self):
        client, cache = FakeClient([ServiceError("ValidationException")]), {}
        result = self.call_probe(client, cache=cache)
        self.assertEqual(result.status, ProbeStatus.UNKNOWN_ERROR)
        self.assertEqual(cache, {})

    def test_region_mismatch_rejected_without_call(self):
        client = FakeClient()
        with self.assertRaises(ValueError):
            self.call_probe(client, region="us-west-2")
        self.assertEqual(client.calls, [])

    def test_transient_probe_stops_fallback_and_can_be_retried(self):
        client = FakeClient([ServiceError("ThrottlingException"), {}])
        registry = ModelRegistry(
            candidates={"workhorse": [SONNET, HAIKU]},
            probe_fn=lambda model: self.call_probe(client, model),
        )
        with self.assertRaises(ProbeFailedError):
            registry.resolve()
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(registry.resolve().model_id, SONNET)
        registry.resolve()
        self.assertEqual(len(client.calls), 2)

    def test_unavailable_can_advance_explicit_same_provider_candidates(self):
        client = FakeClient([ServiceError("AccessDeniedException"), {}])
        registry = ModelRegistry(
            candidates={"workhorse": [SONNET, HAIKU]},
            probe_fn=lambda model: self.call_probe(client, model),
        )
        self.assertEqual(registry.resolve().model_id, HAIKU)

    def test_cross_provider_fallback_requires_opt_in(self):
        for allowed in (False, True):
            client = FakeClient([ServiceError("AccessDeniedException"), {}])
            registry = ModelRegistry(
                candidates={"workhorse": [SONNET, GPT]},
                probe_fn=lambda model, client=client: self.call_probe(client, model),
                allow_cross_provider_fallback=allowed,
            )
            if allowed:
                self.assertEqual(registry.resolve().provider, "openai")
            else:
                with self.assertRaises(SelectionFrozenError):
                    registry.resolve()
                self.assertEqual(len(client.calls), 1)

    def test_single_unavailable_does_not_get_a_hidden_fallback(self):
        client = FakeClient([ServiceError("ResourceNotFoundException")])
        registry = ModelRegistry(probe_fn=lambda model: self.call_probe(client, model))
        with self.assertRaises(ModelUnavailableError):
            registry.resolve()
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
