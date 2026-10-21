"""Keep SDK/server compatibility restricted to this text-only dataset API."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

p = Path(__file__).resolve().parents[1] / "04-deep-dive-topics/utils/langfuse_text_transport.py"
spec = importlib.util.spec_from_file_location("stable_text_transport", p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def item():
    return {
        "id": "item-1",
        "status": "ACTIVE",
        "input": {"query": "A question"},
        "expectedOutput": {"should_contain": ["answer"]},
        "metadata": {},
        "datasetId": "dataset-1",
        "datasetName": "test",
        "createdAt": "2026-10-20T00:00:00Z",
        "updatedAt": "2026-10-20T00:00:00Z",
    }


def test_adds_only_missing_field_without_mutating_source():
    source = item()
    result = module.adapt_text_dataset_response(source, collection=False)
    assert result == {**source, "mediaReferences": []}
    assert "mediaReferences" not in source


@pytest.mark.parametrize(
    "content",
    [{"image_url": "https://example.com/image"}, "@@@langfuseMedia:x", {"text": "data:image/png;base64,AA=="}],
)
def test_media_rejected_before_write(content):
    calls = []
    with httpx.Client(
        transport=module.TextDatasetTransport("https://example.com", httpx.MockTransport(lambda r: calls.append(r)))
    ) as client, pytest.raises(module.TextDatasetCompatibilityError):
        client.post("https://example.com/api/public/dataset-items", json={"input": content})
    assert calls == []


def test_unrelated_response_left_unchanged():
    with httpx.Client(
        transport=module.TextDatasetTransport(
            "https://example.com", httpx.MockTransport(lambda r: httpx.Response(200, json={"unchanged": True}))
        )
    ) as client:
        assert client.get("https://example.com/api/public/traces").json() == {"unchanged": True}


def test_sdk_validation_still_rejects_missing_identity():
    source = item()
    del source["id"]
    with pytest.raises(ValidationError):
        module.adapt_text_dataset_response(source, collection=False)
