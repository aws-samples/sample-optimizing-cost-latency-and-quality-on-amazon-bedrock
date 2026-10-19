"""Principal-scoped preflight with injected clients; no live AWS calls."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from workshop_utils.models import ProbeFailedError
from workshop_utils.preflight import preflight_models
from workshop_utils.pricing import InferredPriceError, get_price


class Error(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code, "Message": "offline fixture"}}


class Client:
    def __init__(self, outcomes=()):
        self.outcomes, self.calls = list(outcomes), []
        self.meta = SimpleNamespace(
            region_name="us-east-1", endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
        )

    def converse(self, **request):
        self.calls.append(request)
        outcome = self.outcomes.pop(0) if self.outcomes else {}
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class PreflightTests(unittest.TestCase):
    def preflight(
        self, client, path, *, principal="arn:aws:sts::111111111111:assumed-role/notebook/session", **kwargs
    ):
        sts = Mock()
        sts.get_caller_identity.return_value = {"Account": "111111111111", "Arn": principal}
        return preflight_models(
            client,
            sts_client=sts,
            region="us-east-1",
            cache_path=path,
            sdk_version="fixture-sdk",
            checked_on="2026-09-24",
            **kwargs,
        )

    def test_unavailable_primary_falls_back_before_freezing(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Client([Error("AccessDeniedException"), {}, {}])
            result = self.preflight(client, Path(directory) / ".models.json")
            self.assertEqual(result.models["workhorse"].model_id, "global.anthropic.claude-sonnet-4-6")
            self.assertEqual(len(client.calls), 3)
            self.assertEqual(result.pricing["workhorse"]["input_per_million"], 3)
            self.assertEqual(result.probes[0]["status"], "unavailable")

    def test_transient_failure_never_falls_back_or_gets_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".models.json"
            client = Client([Error("ThrottlingException")])
            with self.assertRaises(ProbeFailedError):
                self.preflight(client, path)
            self.assertEqual(len(client.calls), 1)
            self.assertFalse(path.exists())

    def test_cache_is_principal_scoped_with_full_probe_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".models.json"
            client = Client()
            self.preflight(client, path)
            cached = self.preflight(client, path)
            self.assertEqual(len(client.calls), 2)
            self.assertTrue(all(row["cache_hit"] for row in cached.probes))
            self.preflight(client, path, principal="arn:aws:sts::111111111111:assumed-role/runtime/session")
            self.assertEqual(len(client.calls), 4)
            entries = json.loads(path.read_text())["entries"]
            self.assertEqual(len(entries), 4)
            self.assertTrue(
                {"principal_arn", "account_id", "region", "model_id", "api", "endpoint", "sdk_version", "checked_on"}
                <= entries[0]["key"].keys()
            )

    def test_gpt_and_inferred_prices_need_separate_explicit_opt_ins(self):
        # Exercise the evidence gate without assuming a real model's price will
        # remain inferred after its provider publishes a model card.
        def experimental_price(model, *, allow_inferred=False, **kwargs):
            price = get_price(model, allow_inferred=allow_inferred, **kwargs)
            if price.model_id == "openai.gpt-5.6-luna":
                if not allow_inferred:
                    raise InferredPriceError("Synthetic inferred-price fixture")
                return replace(price, evidence="inferred")
            return price

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("workshop_utils.preflight.get_price", side_effect=experimental_price),
        ):
            client = Client()
            override = {"workhorse": "global.openai.gpt-5.6-luna"}
            with self.assertRaises(ValueError):
                self.preflight(client, Path(directory) / "cache", overrides=override)
            with self.assertRaises(InferredPriceError):
                self.preflight(client, Path(directory) / "cache", overrides=override, allow_gpt=True)
            self.assertEqual(client.calls, [])
            result = self.preflight(
                client, Path(directory) / "cache", overrides=override, allow_gpt=True, allow_inferred=True
            )
            self.assertTrue(result.pricing["workhorse"]["inferred"])

    def test_documented_gpt_price_needs_only_the_provider_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.preflight(
                Client(), Path(directory) / "cache",
                overrides={"workhorse": "global.openai.gpt-6-sol"}, allow_gpt=True,
            )
            self.assertFalse(result.pricing["workhorse"]["inferred"])

    def test_explicit_refresh_rechecks_changed_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            path = Path(directory) / "cache"
            self.preflight(client, path)
            self.preflight(client, path, refresh=True)
            self.assertEqual(len(client.calls), 4)


if __name__ == "__main__":
    unittest.main()
