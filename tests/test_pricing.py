"""Cost tests exercise billing invariants, not just table lookup."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from workshop_utils.bedrock import NormalizedUsage, converse, normalize_usage
from workshop_utils.models import resolve_model
from workshop_utils.pricing import (
    AmbiguousCacheUsageError,
    InferredPriceError,
    UnknownPriceError,
    calculate_cost,
    get_price,
)

SONNET = "global.anthropic.claude-sonnet-5"
OPUS = "global.anthropic.claude-opus-5-5"
HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
GPT = "global.openai.gpt-5.6-luna"
SOL = "global.openai.gpt-5.6-sol"
NOVA = "us.amazon.nova-2-lite-v1:0"


class PricingTests(unittest.TestCase):
    def test_observed_luna_converse_writes_are_not_automatically_priced_as_30m(self):
        response = {
            "usage": {"inputTokens": 2, "outputTokens": 20, "cacheReadInputTokens": 5501, "cacheWriteInputTokens": 59}
        }
        result = converse(
            SimpleNamespace(converse=lambda **kwargs: response),
            GPT,
            [{"role": "user", "content": [{"text": "Reply OK."}]}],
        )
        with self.assertRaises(AmbiguousCacheUsageError):
            calculate_cost(result.model, result.usage)
        # A caller can explicitly request an estimate, with the billing caveat
        # carried in serialized price metadata; this does not verify the TTL.
        estimate = calculate_cost(result.model, result.usage, cache_ttl="30m")
        self.assertAlmostEqual(estimate.cache_write_cost, 59 * 0.25 / 1_000_000)
        self.assertTrue(
            any("Converse cache billing remains unverified" in note for note in estimate.as_dict()["price_notes"])
        )
        self.assertEqual(result.usage.cache_write_by_ttl, {})

    def test_sol_documented_standard_rates_and_geo_premium(self):
        for model_id, factor in ((SOL, 1), ("us.openai.gpt-5.6-sol", 1.1)):
            with self.subTest(model_id=model_id):
                record = get_price(model_id)
                self.assertFalse(record.inferred)
                self.assertEqual(record.verified_on, "2026-09-25")
                self.assertEqual(record.evidence, "documented")
                self.assertAlmostEqual(record.input_per_million, 4 * factor)
                self.assertAlmostEqual(record.output_per_million, 20 * factor)
                self.assertAlmostEqual(record.cache_read_per_million, 0.4 * factor)
                self.assertAlmostEqual(record.cache_write_per_million["30m"], 5 * factor)
                self.assertIn(
                    "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html",
                    record.sources,
                )
        for profile in ("in", "eu", "regional"):
            with self.subTest(profile=profile), self.assertRaises(UnknownPriceError):
                get_price("openai.gpt-5.6-sol", profile=profile)
        for tier in ("priority", "flex", "reserved"):
            with self.subTest(tier=tier), self.assertRaises(UnknownPriceError):
                get_price(SOL, service_tier=tier)

    def test_sol_long_context_prices_all_disjoint_meters_once(self):
        short = calculate_cost(SOL, NormalizedUsage(1000, 100, 270000, 1000, {"30m": 1000}))
        self.assertFalse(short.long_context)
        self.assertAlmostEqual(short.total_cost, (1000 * 4 + 100 * 20 + 270000 * 0.4 + 1000 * 5) / 1_000_000)
        long = calculate_cost(SOL, NormalizedUsage(1001, 100, 270000, 1000, {"30m": 1000}))
        self.assertTrue(long.long_context)
        self.assertAlmostEqual(long.input_cost, 1001 * 8 / 1_000_000)
        self.assertAlmostEqual(long.output_cost, 100 * 30 / 1_000_000)
        self.assertAlmostEqual(long.cache_read_cost, 270000 * 0.8 / 1_000_000)
        self.assertAlmostEqual(long.cache_write_cost, 1000 * 10 / 1_000_000)
        with self.assertRaises(UnknownPriceError):
            calculate_cost(SOL, NormalizedUsage(0, 0, 0, 10, {"5m": 10}))

    def test_deep_default_uses_opus_5_rates_and_history_retains_opus_55(self):
        from workshop_utils.models import ModelRegistry

        selected = ModelRegistry().resolve("deep")
        self.assertEqual(selected.model_id, "global.anthropic.claude-opus-5")
        self.assertAlmostEqual(calculate_cost(selected, NormalizedUsage(1000, 1000)).total_cost, 0.03)
        self.assertAlmostEqual(calculate_cost(OPUS, NormalizedUsage(1000, 1000)).total_cost, 0.024)

    def test_no_cache_counts_are_double_charged(self):
        usage = NormalizedUsage(100, 20, 4000, 1000, {"5m": 1000})
        cost = calculate_cost(SONNET, usage)
        expected = (100 * 2 + 20 * 10 + 4000 * 0.2 + 1000 * 2.5) / 1_000_000
        self.assertAlmostEqual(cost.total_cost, expected)
        self.assertEqual(cost.input_cost, 0.0002)
        self.assertEqual(cost.as_dict()["pricing_date"], "2026-09-24")

    def test_strands_and_converse_usage_have_identical_cost(self):
        converse_usage = normalize_usage(
            {
                "inputTokens": 2,
                "outputTokens": 20,
                "cacheReadInputTokens": 5501,
                "cacheWriteInputTokens": 59,
            },
            source="converse",
            cache_ttl="30m",
        )
        strands_usage = normalize_usage(
            {
                "gen_ai.usage.input_tokens": 5562,
                "gen_ai.usage.output_tokens": 20,
                "gen_ai.usage.cache_read.input_tokens": 5501,
                "gen_ai.usage.cache_creation.input_tokens": 59,
            },
            source="strands",
            cache_ttl="30m",
        )
        self.assertEqual(calculate_cost(GPT, converse_usage).total_cost, calculate_cost(GPT, strands_usage).total_cost)

    def test_mixed_write_ttls_are_priced_separately(self):
        usage = NormalizedUsage(0, 0, cache_write_tokens=2000, cache_write_by_ttl={"5m": 1000, "1h": 1000})
        self.assertAlmostEqual(calculate_cost(SONNET, usage).total_cost, 0.0065)
        self.assertAlmostEqual(calculate_cost(OPUS, usage).total_cost, 0.013)
        with self.assertRaises(AmbiguousCacheUsageError):
            calculate_cost(SONNET, usage, cache_ttl="5m")

    def test_unknown_write_ttl_is_not_defaulted_to_cheapest_rate(self):
        usage = NormalizedUsage(0, 0, cache_write_tokens=1000)
        with self.assertRaises(AmbiguousCacheUsageError):
            calculate_cost(SONNET, usage)
        self.assertAlmostEqual(calculate_cost(SONNET, usage, cache_ttl="5m").total_cost, 0.0025)
        self.assertAlmostEqual(calculate_cost(SONNET, usage, cache_ttl="1h").total_cost, 0.004)
        with self.assertRaises(UnknownPriceError):
            calculate_cost(SONNET, usage, cache_ttl="30m")

    def test_opus_55_cache_reads_are_five_percent_not_ten_percent(self):
        cost = calculate_cost(OPUS, NormalizedUsage(0, 0, cache_read_tokens=1_000_000))
        self.assertAlmostEqual(cost.total_cost, 0.2)

    def test_haiku_45_does_not_use_haiku_3_prices(self):
        self.assertAlmostEqual(calculate_cost(HAIKU, NormalizedUsage(1_000_000, 1_000_000)).total_cost, 6.0)

    def test_exact_model_and_profile_required(self):
        for model_id in ("unknown", SONNET + "-preview", "global.openai.gpt-5.5"):
            with self.subTest(model_id=model_id), self.assertRaises(UnknownPriceError):
                calculate_cost(model_id, NormalizedUsage(100, 10))
        with self.assertRaises(UnknownPriceError):
            get_price("anthropic.claude-sonnet-5")
        with self.assertRaises(UnknownPriceError):
            get_price(SONNET, profile="us")
        self.assertEqual(get_price("anthropic.claude-sonnet-5", profile="global"), get_price(SONNET))

    def test_geo_profile_premium_and_inferred_mapping_are_explicit(self):
        global_cost = calculate_cost(HAIKU, NormalizedUsage(1000, 100))
        geo_cost = calculate_cost(HAIKU.replace("global.", "us."), NormalizedUsage(1000, 100))
        self.assertAlmostEqual(geo_cost.total_cost, global_cost.total_cost * 1.1)
        self.assertEqual(geo_cost.price.profile_factor, 1.1)
        with self.assertRaises(InferredPriceError):
            get_price(SONNET.replace("global.", "us."))
        self.assertTrue(get_price(SONNET.replace("global.", "us."), allow_inferred=True).inferred)

    def test_gpt_6_documented_prices_need_no_opt_in_and_preserve_profile_rates(self):
        for variant, rates in (("sol", (2, 10)), ("luna", (0.1, 0.5))):
            for profile, factor in (("global", 1), ("us", 1.1)):
                model_id = f"{profile}.openai.gpt-6-{variant}"
                with self.subTest(model_id=model_id):
                    record = get_price(model_id)
                    self.assertEqual(record.evidence, "documented")
                    self.assertEqual(record.verified_on, "2026-09-30")
                    self.assertEqual(record.service_tier, "standard")
                    self.assertEqual(record.profile_factor, factor)
                    actual = (
                        record.input_per_million,
                        record.output_per_million,
                    )
                    for actual_rate, base_rate in zip(actual, rates, strict=True):
                        self.assertAlmostEqual(actual_rate, base_rate * factor)
                    self.assertIsNone(record.cache_read_per_million)
                    self.assertEqual(record.cache_write_per_million, {})
                    result = calculate_cost(model_id, NormalizedUsage(100, 10))
                    self.assertFalse(result.as_dict()["inferred"])
                    self.assertIn(
                        f"https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-{variant}.html",
                        result.as_dict()["price_sources"],
                    )
                    self.assertTrue(any("2026-09-24" in note for note in record.notes))
                    self.assertTrue(any("TTL" in note and "30m" in note for note in record.notes))
                    for tier in ("priority", "flex", "reserved"):
                        with self.subTest(tier=tier), self.assertRaises(UnknownPriceError):
                            get_price(model_id, service_tier=tier)
            for profile in ("regional", "eu", "in"):
                with self.subTest(variant=variant, profile=profile), self.assertRaises(UnknownPriceError):
                    get_price(f"openai.gpt-6-{variant}", profile=profile)

    def test_gpt_6_documented_long_context_rates_cover_uncached_input_and_output(self):
        for variant, rates in (("sol", (4, 15)), ("luna", (0.2, 0.75))):
            for profile, factor in (("global", 1), ("us", 1.1)):
                model_id = f"{profile}.openai.gpt-6-{variant}"
                with self.subTest(model_id=model_id):
                    short = calculate_cost(model_id, NormalizedUsage(272000, 100))
                    long = calculate_cost(model_id, NormalizedUsage(272001, 100))
                    self.assertFalse(short.long_context)
                    self.assertTrue(long.long_context)
                    self.assertAlmostEqual(short.input_cost, 272000 * rates[0] / 2 * factor / 1_000_000)
                    self.assertAlmostEqual(short.output_cost, 100 * rates[1] / 1.5 * factor / 1_000_000)
                    self.assertAlmostEqual(long.input_cost, 272001 * rates[0] * factor / 1_000_000)
                    self.assertAlmostEqual(long.output_cost, 100 * rates[1] * factor / 1_000_000)

    def test_gpt_6_cache_write_prices_remain_unknown_even_with_an_explicit_ttl(self):
        for variant in ("sol", "luna"):
            for profile in ("global", "us"):
                model_id = f"{profile}.openai.gpt-6-{variant}"
                for model in (model_id, resolve_model(model_id)):
                    for allow_inferred in (False, True):
                        with self.subTest(model=model, allow_inferred=allow_inferred):
                            usage = NormalizedUsage(100, 10, cache_write_tokens=1000)
                            with self.assertRaises(AmbiguousCacheUsageError):
                                calculate_cost(model, usage, allow_inferred=allow_inferred)
                            for ttl in ("30m", "5m", "1h"):
                                with self.subTest(ttl=ttl), self.assertRaises(UnknownPriceError):
                                    calculate_cost(model, usage, cache_ttl=ttl, allow_inferred=allow_inferred)
                                with self.subTest(ttl=ttl), self.assertRaises(UnknownPriceError):
                                    calculate_cost(
                                        model,
                                        NormalizedUsage(100, 10, 0, 300000, {ttl: 300000}),
                                        allow_inferred=allow_inferred,
                                    )

    def test_gpt_6_unexpected_converse_cache_reads_cannot_use_responses_prices(self):
        raw = {"inputTokens": 2, "outputTokens": 20, "cacheReadInputTokens": 5501}
        usage = normalize_usage(raw, source="converse")
        for variant in ("sol", "luna"):
            for profile in ("global", "us"):
                model_id = f"{profile}.openai.gpt-6-{variant}"
                for model in (model_id, resolve_model(model_id)):
                    for allow_inferred in (False, True):
                        with (
                            self.subTest(model=model, allow_inferred=allow_inferred),
                            self.assertRaises(UnknownPriceError),
                        ):
                            calculate_cost(model, usage, allow_inferred=allow_inferred)
        self.assertEqual(usage.cache_read_tokens, 5501)
        self.assertEqual(usage.total_input_tokens, 5503)

    def test_long_context_threshold_uses_total_input_and_whole_request(self):
        short = calculate_cost(GPT, NormalizedUsage(2000, 100, 270000))
        long = calculate_cost(GPT, NormalizedUsage(2001, 100, 270000))
        self.assertFalse(short.long_context)
        self.assertTrue(long.long_context)
        self.assertAlmostEqual(long.input_cost, 2001 * 0.2 * 2 / 1_000_000)
        self.assertAlmostEqual(long.cache_read_cost, 270000 * 0.02 * 2 / 1_000_000)
        self.assertAlmostEqual(long.output_cost, 100 * 1.2 * 1.5 / 1_000_000)

    def test_long_context_write_factor_is_applied_once(self):
        cost = calculate_cost(
            GPT, NormalizedUsage(2, 10, cache_write_tokens=300000, cache_write_by_ttl={"30m": 300000})
        )
        self.assertAlmostEqual(cost.cache_write_cost, 300000 * 0.25 * 2 / 1_000_000)

    def test_claude_long_context_has_no_gpt_surcharge_or_quota_multiplier(self):
        cost = calculate_cost(SONNET, NormalizedUsage(900000, 100))
        self.assertFalse(cost.long_context)
        self.assertAlmostEqual(cost.total_cost, 1.801)
        opus = calculate_cost(OPUS, NormalizedUsage(0, 1000))
        self.assertAlmostEqual(opus.total_cost, 0.02)  # NOT .20 with 10x quota burndown

    def test_reasoning_output_is_already_in_the_output_meter(self):
        usage = normalize_usage(
            {
                "inputTokens": 10,
                "outputTokens": 100,
                "output_tokens_details": {"reasoning_tokens": 90},
            },
            source="converse",
        )
        self.assertAlmostEqual(calculate_cost(GPT, usage).output_cost, 0.00012)

    def test_nova_exact_us_tier_prices_are_dated_and_sourced(self):
        for tier, input_rate, output_rate, factor in (
            ("standard", 0.33, 2.75, 1),
            ("flex", 0.165, 1.375, 0.5),
            ("priority", 0.5775, 4.8125, 1.75),
        ):
            with self.subTest(tier=tier):
                cost = calculate_cost(NOVA, NormalizedUsage(1000, 1000), region="us-east-1", service_tier=tier)
                record = cost.price
                self.assertEqual(record.input_per_million, input_rate)
                self.assertEqual(record.output_per_million, output_rate)
                self.assertEqual(record.service_tier, tier)
                self.assertEqual(record.tier_factor, factor)
                self.assertEqual(record.profile, "us")
                self.assertEqual(record.region, "us-east-1")
                self.assertEqual(record.verified_on, "2026-09-28")
                self.assertEqual(record.evidence, "price-list")
                self.assertFalse(record.inferred)
                self.assertIn("https://aws.amazon.com/nova/pricing/", record.sources)
                self.assertAlmostEqual(cost.total_cost, (input_rate + output_rate) / 1000)
        self.assertEqual(
            get_price(resolve_model(NOVA, region="us-east-1")),
            get_price("amazon.nova-2-lite-v1:0", profile="us", region="us-east-1"),
        )

    def test_nova_region_profile_and_tier_are_not_assumed(self):
        for region in (None, "us-west-2"):
            with self.subTest(region=region), self.assertRaises(UnknownPriceError):
                get_price(NOVA, region=region)
        for profile in (None, "global", "eu", "jp", "regional"):
            with self.subTest(profile=profile), self.assertRaises(UnknownPriceError):
                get_price("amazon.nova-2-lite-v1:0", profile=profile, region="us-east-1")
        for tier in ("reserved", "batch", "default", "unknown"):
            with self.subTest(tier=tier), self.assertRaises(UnknownPriceError):
                get_price(NOVA, region="us-east-1", service_tier=tier)
        with self.assertRaises(UnknownPriceError):
            get_price(resolve_model(NOVA, region="us-east-1"), region="us-west-2")
        with self.assertRaises(UnknownPriceError):
            get_price(NOVA, profile="global", region="us-east-1")

    def test_nova_cache_usage_is_unpriced_without_a_verified_cache_rate(self):
        for tier in ("standard", "flex", "priority"):
            for usage in (
                NormalizedUsage(0, 0, cache_read_tokens=100),
                NormalizedUsage(0, 0, cache_write_tokens=100, cache_write_by_ttl={"5m": 100}),
            ):
                with self.subTest(tier=tier, usage=usage), self.assertRaises(UnknownPriceError):
                    calculate_cost(NOVA, usage, region="us-east-1", service_tier=tier)
        with self.assertRaises(AmbiguousCacheUsageError):
            calculate_cost(NOVA, NormalizedUsage(0, 0, cache_write_tokens=100), region="us-east-1")

    def test_service_tier_prices_are_not_inherited_by_other_models(self):
        with self.assertRaises(UnknownPriceError):
            calculate_cost(SONNET, NormalizedUsage(1, 1), service_tier="priority")
        with self.assertRaises(UnknownPriceError):
            calculate_cost(GPT, NormalizedUsage(1, 1), service_tier="flex")

    def test_raw_usage_and_negative_counts_are_rejected(self):
        with self.assertRaises(TypeError):
            calculate_cost(SONNET, {"inputTokens": 100, "outputTokens": 10})
        with self.assertRaises(ValueError):
            NormalizedUsage(-1, 0)

    def test_zero_usage_costs_zero(self):
        self.assertEqual(calculate_cost(SONNET, NormalizedUsage(0, 0)).total_cost, 0)


if __name__ == "__main__":
    unittest.main()
