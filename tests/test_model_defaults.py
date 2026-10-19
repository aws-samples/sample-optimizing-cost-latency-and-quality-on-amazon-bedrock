"""Workshop model selection and public experiment metadata."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from workshop_utils.bedrock import build_converse_request
from workshop_utils.models import (
    DEFAULT_ALIASES,
    ModelRegistry,
    ProbeStatus,
    UnknownCapabilityError,
    UnknownModelError,
    UnsupportedFeatureError,
    probe,
)
from workshop_utils.pricing import get_price

MESSAGES = [{"role": "user", "content": [{"text": "Reply OK."}]}]


@pytest.mark.parametrize("alias", list(DEFAULT_ALIASES))
def test_defaults_resolve_build_requests_and_have_prices(alias):
    selected = ModelRegistry().resolve(alias)
    assert selected.model_id == DEFAULT_ALIASES[alias]
    assert build_converse_request(selected, MESSAGES)["modelId"] == selected.model_id
    assert not get_price(selected).inferred


def test_availability_probe_records_only_experiment_metadata():
    model_id = DEFAULT_ALIASES["workhorse"]
    result = probe(
        SimpleNamespace(converse=lambda **kwargs: {}),
        model_id,
        account_id="111111111111",
        region="us-east-1",
        sdk_version="offline-fixture",
        checked_on="2026-09-28",
    )
    assert result.status is ProbeStatus.AVAILABLE
    selected = ModelRegistry(probe_fn=lambda model: result, region="us-east-1").resolve()
    metadata = selected.as_dict()
    assert metadata["model_id"] == model_id
    assert metadata["probe_status"] == "available"
    assert metadata["api"] == "converse"
    assert metadata["region"] == "us-east-1"
    assert set(metadata) == {
        "model_id", "base_model_id", "provider", "profile", "api", "endpoint",
        "region", "alias", "capabilities_verified_on", "probe_status",
    }


@pytest.mark.parametrize("model_id", [
    "anthropic.claude-sonnet-5",
    "apac.anthropic.claude-sonnet-5",
    "global.anthropic.claude-sonnet-5-preview",
])
def test_invalid_invocation_profile_is_rejected_before_probe(model_id):
    calls = []
    registry = ModelRegistry(overrides={"workhorse": model_id}, probe_fn=calls.append)
    with pytest.raises(UnknownModelError):
        registry.resolve()
    assert calls == []


def test_sol_does_not_inherit_luna_converse_controls():
    sol = ModelRegistry().resolve("gpt-workhorse")
    for controls in ({"effort": "low"}, {"temperature": 0.5}, {"tool_config": {"tools": []}}):
        with pytest.raises(UnknownCapabilityError):
            build_converse_request(sol, MESSAGES, **controls)
    for controls in (
        {"system": [{"cachePoint": {"type": "default"}}]},
        {"output_config": {"textFormat": {}}},
    ):
        with pytest.raises(UnsupportedFeatureError):
            build_converse_request(sol, MESSAGES, **controls)
