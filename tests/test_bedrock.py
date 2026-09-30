"""Offline request-shape and usage regression tests."""

from __future__ import annotations

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from workshop_utils.bedrock import (
    NormalizedUsage,
    build_converse_request,
    converse,
    normalize_usage,
)
from workshop_utils.models import (
    ModelRegistry,
    UnknownCapabilityError,
    UnsupportedFeatureError,
)
from workshop_utils.pacing import InferencePacer

SONNET = "global.anthropic.claude-sonnet-5"
OPUS = "global.anthropic.claude-opus-5-5"
HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
GPT = "global.openai.gpt-6-luna"
LUNA = "global.openai.gpt-5.6-luna"
NOVA = "us.amazon.nova-2-lite-v1:0"
MESSAGES = [{"role": "user", "content": [{"text": "Classify this ticket."}]}]
TOOL = {
    "toolSpec": {
        "name": "classify",
        "description": "Classify the ticket",
        "inputSchema": {"json": {"type": "object", "properties": {}}},
    }
}
FORMAT = {
    "textFormat": {
        "type": "json_schema",
        "structure": {
            "jsonSchema": {"name": "ticket", "schema": '{"type":"object"}'},
        },
    }
}


class RequestTests(unittest.TestCase):
    def test_sonnet_46_max_effort_builds_an_adaptive_converse_request(self):
        request = build_converse_request(
            "global.anthropic.claude-sonnet-4-6",
            MESSAGES,
            effort="max",
            thinking={"type": "adaptive"},
        )
        self.assertEqual(
            request["additionalModelRequestFields"],
            {"output_config": {"effort": "max"}, "thinking": {"type": "adaptive"}},
        )

    def test_gpt_6_guardrails_and_structured_output_follow_exact_model_cards(self):
        guardrail = {"guardrailIdentifier": "test", "guardrailVersion": "1"}
        for model_id in ("global.openai.gpt-6-sol", GPT):
            with self.subTest(model_id=model_id):
                request = build_converse_request(model_id, MESSAGES, guardrail_config=guardrail)
                self.assertEqual(request["guardrailConfig"], guardrail)
                for controls in (
                    {"system": [{"text": "Prefix"}, {"cachePoint": {"type": "default"}}]},
                    {"additional_model_request_fields": {"prompt_cache_options": {"mode": "explicit", "ttl": "30m"}}},
                ):
                    with self.subTest(controls=controls), self.assertRaises(UnsupportedFeatureError):
                        build_converse_request(model_id, MESSAGES, **controls)
                for tier in ("priority", "flex", "reserved"):
                    with self.subTest(tier=tier), self.assertRaises(UnsupportedFeatureError):
                        build_converse_request(model_id, MESSAGES, service_tier=tier)
        self.assertEqual(build_converse_request(GPT, MESSAGES, output_config=FORMAT)["outputConfig"], FORMAT)
        with self.assertRaises(UnsupportedFeatureError):
            build_converse_request("global.openai.gpt-6-sol", MESSAGES, output_config=FORMAT)

    def test_requests_validate_against_installed_botocore_schema(self):
        try:
            import botocore.session
            from botocore.validate import validate_parameters
        except ImportError:
            self.skipTest("Optional SDK schema check; core tests need only the standard library")
        service = botocore.session.get_session().get_service_model("bedrock-runtime")
        shape = service.operation_model("Converse").input_shape
        for request in (
            build_converse_request(SONNET, MESSAGES, effort="low", service_tier="standard"),
            build_converse_request(GPT, MESSAGES, effort="none"),
            build_converse_request(LUNA, MESSAGES, effort="none"),
            *(
                build_converse_request(NOVA, MESSAGES, max_tokens=128, service_tier=tier)
                for tier in ("standard", "flex", "priority")
            ),
            build_converse_request(HAIKU, MESSAGES, output_config=FORMAT),
        ):
            with self.subTest(model_id=request["modelId"]):
                validate_parameters(request, shape)
        self.assertIn("default", shape.members["serviceTier"].members["type"].enum)

    def test_effort_uses_exact_provider_shape(self):
        for model, expected in (
            (SONNET, {"output_config": {"effort": "low"}}),
            (OPUS, {"output_config": {"effort": "low"}}),
            (GPT, {"reasoning": {"effort": "low"}}),
            (LUNA, {"reasoning": {"effort": "low"}}),
        ):
            with self.subTest(model=model):
                request = build_converse_request(model, MESSAGES, effort="low")
                self.assertEqual(request["additionalModelRequestFields"], expected)
                self.assertEqual(request["modelId"], model)

    def test_provider_effort_fields_require_the_exact_nested_shape(self):
        for model, fields in (
            (SONNET, {"reasoning": {"effort": "low"}}),
            (SONNET, {"output_config": "low"}),
            (LUNA, {"output_config": {"effort": "low"}}),
            (LUNA, {"reasoning_effort": "low"}),
            (LUNA, {"reasoning": "low"}),
        ):
            with self.subTest(model=model, fields=fields), self.assertRaises(UnsupportedFeatureError):
                build_converse_request(model, MESSAGES, additional_model_request_fields=fields)
        for model, fields in (
            (SONNET, {"output_config": {"effort": "high"}}),
            (LUNA, {"reasoning": {"effort": "high"}}),
        ):
            with self.subTest(model=model), self.assertRaises(ValueError):
                build_converse_request(model, MESSAGES, effort="low", additional_model_request_fields=fields)

    def test_nova_tiers_use_minimal_converse_controls(self):
        for tier, wire in (("standard", "default"), ("flex", "flex"), ("priority", "priority")):
            with self.subTest(tier=tier):
                self.assertEqual(
                    build_converse_request(NOVA, MESSAGES, max_tokens=128, service_tier=tier),
                    {
                        "modelId": NOVA,
                        "messages": MESSAGES,
                        "inferenceConfig": {"maxTokens": 128},
                        "serviceTier": {"type": wire},
                    },
                )
        self.assertNotIn("additionalModelRequestFields", build_converse_request(NOVA, MESSAGES))
        for controls in (
            {"effort": "low"},
            {"temperature": 0},
            {"top_p": 0.9},
            {"tool_config": {"tools": [TOOL]}},
            {"system": [{"cachePoint": {"type": "default"}}]},
        ):
            with self.subTest(controls=controls), self.assertRaises(UnknownCapabilityError):
                build_converse_request(NOVA, MESSAGES, **controls)
        for fields in (
            {"reasoning_effort": "low"},
            {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}},
        ):
            with self.subTest(fields=fields), self.assertRaises(UnsupportedFeatureError):
                build_converse_request(NOVA, MESSAGES, additional_model_request_fields=fields)
        for tier in ("reserved", "batch", "default"):
            with self.subTest(tier=tier), self.assertRaises(UnsupportedFeatureError):
                build_converse_request(NOVA, MESSAGES, service_tier=tier)

    def test_luna_does_not_accept_responses_cache_controls_or_converse_points(self):
        for controls in (
            {"system": [{"text": "Prefix"}, {"cachePoint": {"type": "default", "ttl": "30m"}}]},
            {"additional_model_request_fields": {"prompt_cache_options": {"mode": "explicit"}}},
        ):
            with self.subTest(controls=controls), self.assertRaises(UnsupportedFeatureError):
                build_converse_request(LUNA, MESSAGES, **controls)

    def test_invalid_effort_and_haiku_effort_rejected(self):
        for model, effort in (
            (SONNET, "none"),
            (HAIKU, "low"),
            (GPT, "bogus"),
            ("global.anthropic.claude-sonnet-4-6", "xhigh"),
        ):
            with self.subTest(model=model), self.assertRaises(UnsupportedFeatureError):
                build_converse_request(model, MESSAGES, effort=effort)

    def test_sampling_is_rejected_even_when_gpt_reasoning_is_none(self):
        for model, controls in (
            (SONNET, {"temperature": 0.1}),
            (OPUS, {"top_p": 0.9}),
            (GPT, {"temperature": 0.1, "effort": "none"}),
            (GPT, {"top_k": 20}),
        ):
            with self.subTest(model=model), self.assertRaises(UnsupportedFeatureError):
                build_converse_request(model, MESSAGES, **controls)
        self.assertEqual(build_converse_request(HAIKU, MESSAGES, temperature=0)["inferenceConfig"]["temperature"], 0)

    def test_provider_fields_cannot_bypass_validation(self):
        for model, fields in (
            (SONNET, {"reasoning": {"effort": "low"}}),
            (GPT, {"output_config": {"effort": "low"}}),
            (GPT, {"reasoning_effort": "low"}),
            (SONNET, {"temperature": 0.2}),
            (SONNET, {"output_config": {"effort": "low", "task_budget": {"total": 20000}}}),
            (GPT, {"prompt_cache_options": {"mode": "explicit"}}),
        ):
            with self.subTest(fields=fields), self.assertRaises(UnsupportedFeatureError):
                build_converse_request(model, MESSAGES, additional_model_request_fields=fields)

    def test_duplicate_effort_is_not_silently_overwritten(self):
        with self.assertRaises(ValueError):
            build_converse_request(
                SONNET, MESSAGES, effort="low", additional_model_request_fields={"output_config": {"effort": "high"}}
            )

    def test_thinking_constraints(self):
        for model, controls in (
            (OPUS, {"thinking": {"type": "disabled"}}),
            (OPUS, {"thinking": {"type": "enabled", "budget_tokens": 2048}, "max_tokens": 4096}),
            (SONNET, {"thinking": {"type": "disabled"}, "effort": "max"}),
            (GPT, {"thinking": {"type": "adaptive"}}),
            (HAIKU, {"thinking": {"type": "enabled", "budget_tokens": 2048}, "max_tokens": 4096, "temperature": 0.5}),
        ):
            with self.subTest(controls=controls), self.assertRaises(UnsupportedFeatureError):
                build_converse_request(model, MESSAGES, **controls)
        valid = build_converse_request(
            OPUS,
            MESSAGES,
            thinking={"type": "adaptive", "display": "summarized"},
            effort="low",
        )
        self.assertEqual(valid["additionalModelRequestFields"]["thinking"]["display"], "summarized")

    def test_haiku_thinking_budget_must_fit_output_limit(self):
        with self.assertRaises(ValueError):
            build_converse_request(
                HAIKU, MESSAGES, max_tokens=1024, thinking={"type": "enabled", "budget_tokens": 1024}
            )

    def test_stop_sequences_require_verified_thinking_combination(self):
        with self.assertRaises(UnknownCapabilityError):
            build_converse_request(SONNET, MESSAGES, stop_sequences=["END"])
        request = build_converse_request(SONNET, MESSAGES, stop_sequences=["END"], thinking={"type": "disabled"})
        self.assertEqual(request["inferenceConfig"]["stopSequences"], ["END"])

    def test_structure_and_tools_are_not_substituted(self):
        for model in (SONNET, OPUS):
            with self.subTest(model=model), self.assertRaises(UnsupportedFeatureError):
                build_converse_request(model, MESSAGES, output_config=FORMAT)
        self.assertEqual(build_converse_request(HAIKU, MESSAGES, output_config=FORMAT)["outputConfig"], FORMAT)
        forced = {"tools": [TOOL], "toolChoice": {"tool": {"name": "classify"}}}
        self.assertEqual(build_converse_request(SONNET, MESSAGES, tool_config=forced)["toolConfig"], forced)
        with self.assertRaises(UnsupportedFeatureError):
            build_converse_request(OPUS, MESSAGES, tool_config=forced)
        strict = deepcopy(TOOL)
        strict["toolSpec"]["strict"] = True
        with self.assertRaises(UnsupportedFeatureError):
            build_converse_request(SONNET, MESSAGES, tool_config={"tools": [strict]})

    def test_guardrail_unknown_differs_from_disabled_feature(self):
        with self.assertRaises(UnknownCapabilityError):
            build_converse_request(NOVA, MESSAGES, guardrail_config={"guardrailIdentifier": "test"})
        with self.assertRaises(UnsupportedFeatureError):
            build_converse_request(SONNET, MESSAGES, service_tier="flex")

    def test_standard_tier_uses_the_actual_boto_wire_value(self):
        request = build_converse_request(SONNET, MESSAGES, service_tier="standard")
        self.assertEqual(request["serviceTier"], {"type": "default"})

    def test_cache_points_are_explicit_and_inputs_are_not_mutated(self):
        messages = deepcopy(MESSAGES)
        system = [{"text": "Static prefix"}, {"cachePoint": {"type": "default", "ttl": "1h"}}]
        original = deepcopy((messages, system))
        request = build_converse_request(SONNET, messages, system=system)
        request["messages"][0]["content"][0]["text"] = "changed"
        request["system"][0]["text"] = "changed"
        self.assertEqual((messages, system), original)
        with self.assertRaises(UnsupportedFeatureError):
            build_converse_request(GPT, messages, system=system)

    def test_invalid_cache_blocks_and_limits(self):
        for system in (
            [{"text": "Invalid combined block", "cachePoint": {"type": "default"}}],
            [{"cachePoint": {"type": "default"}}] * 5,
        ):
            with self.subTest(system=system), self.assertRaises(ValueError):
                build_converse_request(SONNET, MESSAGES, system=system)
        with self.assertRaises(UnsupportedFeatureError):
            build_converse_request(SONNET, MESSAGES, system=[{"cachePoint": {"type": "default", "ttl": "30m"}}])

    def test_schema_named_cache_point_is_not_a_request_cache_control(self):
        tool = deepcopy(TOOL)
        tool["toolSpec"]["inputSchema"]["json"]["properties"]["cachePoint"] = {"type": "string"}
        build_converse_request(GPT, MESSAGES, tool_config={"tools": [tool]})

    def test_mixed_cache_ttl_order_follows_tools_system_messages(self):
        build_converse_request(
            SONNET,
            MESSAGES,
            tool_config={"tools": [TOOL, {"cachePoint": {"type": "default", "ttl": "1h"}}]},
            system=[{"text": "prefix"}, {"cachePoint": {"type": "default"}}],
        )
        with self.assertRaises(ValueError):
            build_converse_request(
                SONNET,
                MESSAGES,
                tool_config={"tools": [TOOL, {"cachePoint": {"type": "default"}}]},
                system=[{"text": "prefix"}, {"cachePoint": {"type": "default", "ttl": "1h"}}],
            )

    def test_invalid_token_and_sampling_values(self):
        for maximum in (True, 0, -1, 1.5):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                build_converse_request(HAIKU, MESSAGES, max_tokens=maximum)
        for temperature in (True, float("nan"), float("inf"), -1, 2):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                build_converse_request(HAIKU, MESSAGES, temperature=temperature)

    def test_unknown_control_is_not_dropped(self):
        with self.assertRaises(TypeError):
            build_converse_request(SONNET, MESSAGES, fictional_control=True)


class UsageTests(unittest.TestCase):
    def test_converse_input_excludes_cache(self):
        result = normalize_usage(
            {
                "inputTokens": 2,
                "outputTokens": 20,
                "cacheReadInputTokens": 5501,
                "cacheWriteInputTokens": 59,
                "totalTokens": 22,
            },
            source="converse",
        )
        self.assertEqual(result.input_tokens, 2)
        self.assertEqual(result.total_input_tokens, 5562)
        self.assertEqual(result.total_tokens, 5582)

    def test_strands_inclusive_input_normalizes_to_same_disjoint_meters(self):
        attrs = {
            "gen_ai.usage.input_tokens": 5562,
            "gen_ai.usage.output_tokens": 20,
            "gen_ai.usage.cache_read.input_tokens": 5501,
            "gen_ai.usage.cache_creation.input_tokens": 59,
        }
        result = normalize_usage(attrs, source="strands", cache_ttl="30m")
        self.assertEqual(result, NormalizedUsage(2, 20, 5501, 59, {"30m": 59}))

    def test_raw_usage_cannot_be_mistaken_for_strands_telemetry(self):
        with self.assertRaises(ValueError):
            normalize_usage({"inputTokens": 5562, "outputTokens": 20}, source="strands")
        with self.assertRaises(ValueError):
            normalize_usage({}, source="unknown")

    def test_inconsistent_cache_counts_do_not_get_clamped(self):
        with self.assertRaises(ValueError):
            normalize_usage(
                {
                    "gen_ai.usage.input_tokens": 10,
                    "gen_ai.usage.output_tokens": 1,
                    "gen_ai.usage.cache_read.input_tokens": 11,
                },
                source="strands",
            )
        with self.assertRaises(ValueError):
            normalize_usage(
                {
                    "gen_ai.usage.input_tokens": 100,
                    "gen_ai.usage.output_tokens": 1,
                    "gen_ai.usage.cache_read.input_tokens": 10,
                    "gen_ai.usage.cache_read_input_tokens": 11,
                },
                source="strands",
            )

    def test_ttl_details_are_a_partition_not_an_additional_meter(self):
        result = normalize_usage(
            {
                "inputTokens": 2,
                "outputTokens": 1,
                "cacheWriteInputTokens": 1500,
                "cacheDetails": [{"ttl": "5m", "inputTokens": 500}, {"ttl": "1h", "inputTokens": 1000}],
            },
            source="converse",
        )
        self.assertEqual(result.total_input_tokens, 1502)
        self.assertEqual(result.cache_write_by_ttl, {"5m": 500, "1h": 1000})
        with self.assertRaises(TypeError):
            result.cache_write_by_ttl["5m"] = 0

    def test_missing_bad_or_inconsistent_meters_are_errors(self):
        for usage in (
            {},
            {"inputTokens": -1, "outputTokens": 1},
            {"inputTokens": 1.5, "outputTokens": 1},
            {"inputTokens": 1, "outputTokens": True},
            {
                "inputTokens": 1,
                "outputTokens": 1,
                "cacheWriteInputTokens": 2,
                "cacheDetails": [{"ttl": "5m", "inputTokens": 1}],
            },
        ):
            with self.subTest(usage=usage), self.assertRaises(ValueError):
                normalize_usage(usage, source="converse")


class ConverseTests(unittest.TestCase):
    def setUp(self):
        self.clock = 100.0
        self.sleeps = []

        def sleep(delay):
            self.sleeps.append(delay)
            self.clock += delay

        self.pacer = InferencePacer(clock=lambda: self.clock, sleeper=sleep)
        factory_patch = patch("workshop_utils.pacing.get_workshop_pacer", return_value=self.pacer)
        self.factory = factory_patch.start()
        self.addCleanup(factory_patch.stop)

    def test_default_pacing_preserves_request_and_injected_pacer_is_not_a_model_control(self):
        sent = []
        response = {"usage": {"inputTokens": 2, "outputTokens": 4}, "metrics": {"latencyMs": 25}}
        client = SimpleNamespace(converse=lambda **kwargs: sent.append((self.clock, kwargs)) or response)
        first = converse(client, SONNET, MESSAGES, effort="low")
        second = converse(client, SONNET, MESSAGES, effort="low", pacer=self.pacer)
        self.factory.assert_called_once_with()
        self.assertAlmostEqual(sent[1][0] - sent[0][0], 1.1)
        self.assertEqual(sent[0][1], sent[1][1])
        self.assertNotIn("pacer", second.request)
        self.assertEqual(first.latency_ms, 25)
        self.assertEqual(second.latency_ms, 25)

    def test_undocumented_cache_meters_survive_without_invented_ttl(self):
        response = {
            "usage": {"inputTokens": 2, "outputTokens": 20, "cacheReadInputTokens": 5501, "cacheWriteInputTokens": 59}
        }
        original = deepcopy(response)
        for model in (LUNA, GPT, "global.openai.gpt-5.6-sol", "global.openai.gpt-6-sol"):
            with self.subTest(model=model):
                result = converse(SimpleNamespace(converse=lambda **kwargs: response), model, MESSAGES)
                self.assertIs(result.response, response)
                self.assertEqual(result.usage, NormalizedUsage(2, 20, 5501, 59))
                self.assertEqual(result.usage.total_input_tokens, 5562)
                self.assertEqual(result.usage.cache_write_by_ttl, {})
        self.assertEqual(response, original)

    def test_observed_cache_details_survive_even_without_documented_api_caching(self):
        for model in (LUNA, GPT):
            for ttl in ("5m", "30m"):
                response = {
                    "usage": {
                        "inputTokens": 2,
                        "outputTokens": 20,
                        "cacheReadInputTokens": 5501,
                        "cacheWriteInputTokens": 59,
                        "cacheDetails": [{"ttl": ttl, "inputTokens": 59}],
                    }
                }
                with self.subTest(model=model, ttl=ttl):
                    result = converse(
                        SimpleNamespace(converse=lambda response=response, **kwargs: response), model, MESSAGES
                    )
                    self.assertEqual(result.usage.cache_write_by_ttl, {ttl: 59})
                    self.assertEqual(result.usage.total_input_tokens, 5562)

    def test_injected_client_receives_one_exact_request(self):
        calls = []
        response = {
            "output": {"message": {"content": [{"reasoningContent": {}}, {"text": "OK"}]}},
            "usage": {"inputTokens": 2, "outputTokens": 4, "cacheWriteInputTokens": 2048},
            "metrics": {"latencyMs": 200},
        }
        client = SimpleNamespace(converse=lambda **kwargs: calls.append(kwargs) or response)
        result = converse(
            client,
            SONNET,
            MESSAGES,
            effort="low",
            system=[{"text": "Prefix"}, {"cachePoint": {"type": "default", "ttl": "1h"}}],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], result.request)
        self.assertIs(result.response, response)
        self.assertEqual(result.text, "OK")
        self.assertEqual(result.latency_ms, 200)
        self.assertEqual(result.usage.cache_write_by_ttl, {"1h": 2048})

    def test_unsupported_parameter_never_reaches_client(self):
        calls = []
        client = SimpleNamespace(converse=lambda **kwargs: calls.append(kwargs))
        with self.assertRaises(UnsupportedFeatureError):
            converse(client, SONNET, MESSAGES, temperature=0.5)
        self.assertEqual(calls, [])

    def test_region_mismatch_cannot_change_an_experiment(self):
        model = ModelRegistry(region="us-east-1").resolve()
        client = SimpleNamespace(meta=SimpleNamespace(region_name="us-west-2"))
        with self.assertRaises(ValueError):
            converse(client, model, MESSAGES)

    def test_service_failures_propagate_no_api_or_model_retry(self):
        calls = []

        def failure(**kwargs):
            calls.append(kwargs)
            raise TimeoutError("service fixture")

        with self.assertRaises(TimeoutError):
            converse(SimpleNamespace(converse=failure), SONNET, MESSAGES)
        self.assertEqual(len(calls), 1)



if __name__ == "__main__":
    unittest.main()
