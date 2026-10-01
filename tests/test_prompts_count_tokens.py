"""Run the token-count cell with a fake clock/client; no generation or network."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import anthropic
import httpx2
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "01-fundamentals/01-prompts-101.ipynb"
CELL = next(
    "".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text())["cells"]
    if cell["id"] == "cell-008"
)
ACTIVATION_MESSAGE = (
    "Your subscription to the model is being set up. "
    "Check your Marketplace subscriptions page to confirm."
)
MODEL = "anthropic.claude-sonnet-5"


def denied(body, status=403):
    response = httpx2.Response(
        status, request=httpx2.Request("POST", "https://offline.invalid/count_tokens"),
    )
    return anthropic.PermissionDeniedError("Synthetic permission error", response=response, body=body)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0
        self.sleeps.append(seconds)
        self.now += seconds


class CountClient:
    def __init__(self):
        self.closed = False
        self.messages = SimpleNamespace(
            count_tokens=Mock(),
            create=Mock(side_effect=AssertionError("Counting must not generate")),
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


@pytest.fixture
def lab(monkeypatch):
    clock, client = Clock(), CountClient()
    factory = Mock(return_value=client)
    monkeypatch.setattr(anthropic, "AnthropicBedrockMantle", factory)
    # Any accidental SDK/network path is forbidden, even if client patching regresses.
    monkeypatch.setattr(
        httpx2.Client, "send", Mock(side_effect=AssertionError("No network in this test")),
    )
    supported = object()
    runtime = SimpleNamespace(
        count_tokens=Mock(),
        converse=Mock(side_effect=AssertionError("Counting must not generate")),
    )
    namespace = {
        "SELECTED": SimpleNamespace(
            base_model_id=MODEL, capabilities=SimpleNamespace(support=lambda _: None),
        ),
        "Support": SimpleNamespace(SUPPORTED=supported),
        "time": clock, "REGION": "us-east-1", "SYSTEM": "Use the supplied policy.",
        "QUESTION": "Can I return these headphones?", "runtime": runtime, "print": Mock(),
        "request": {"messages": [{"role": "user", "content": [{"text": "same input"}]}],
                    "system": [{"text": "same system"}]},
    }
    return SimpleNamespace(clock=clock, client=client, factory=factory, namespace=namespace)


def execute(lab):
    exec(compile(CELL, f"{NOTEBOOK}:cell-008", "exec"), lab.namespace)


@pytest.mark.parametrize("body", [
    {"message": ACTIVATION_MESSAGE},
    {"error": {"type": "permission_error", "message": ACTIVATION_MESSAGE}},
    ACTIVATION_MESSAGE,
])
def test_activation_403_retries_only_counting_then_preserves_real_count(lab, body):
    lab.client.messages.count_tokens.side_effect = [
        denied(body), SimpleNamespace(input_tokens=87),
    ]
    execute(lab)
    calls = lab.client.messages.count_tokens.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs == calls[1].kwargs
    assert calls[0].kwargs["model"] == MODEL
    assert calls[0].kwargs["system"] == lab.namespace["SYSTEM"]
    assert calls[0].kwargs["messages"] == [
        {"role": "user", "content": lab.namespace["QUESTION"]},
    ]
    assert lab.namespace["counted_input_tokens"] == 87
    assert lab.clock.sleeps == [10]
    assert lab.client.closed
    assert lab.factory.call_args.kwargs["max_retries"] == 0
    lab.client.messages.create.assert_not_called()
    lab.namespace["runtime"].converse.assert_not_called()
    output = " ".join(str(call.args) for call in lab.namespace["print"].call_args_list)
    assert "activation" in output.lower() and "ready" in output.lower()
    assert "87" in output
    assert "Marketplace" not in output


def test_activation_deadline_is_bounded_and_suggests_count_cell_retry_only(lab):
    lab.client.messages.count_tokens.side_effect = denied({"message": ACTIVATION_MESSAGE})
    with pytest.raises(TimeoutError, match="rerun only this token-count cell") as caught:
        execute(lab)
    assert lab.clock.now == 180
    assert sum(lab.clock.sleeps) == 180
    assert lab.client.messages.count_tokens.call_count == 18
    assert lab.client.closed
    assert lab.namespace["counted_input_tokens"] is None
    assert caught.value.__suppress_context__
    assert "Marketplace" not in str(caught.value)
    lab.namespace["runtime"].converse.assert_not_called()


def test_request_time_and_sleep_share_deadline_and_final_timeout_is_capped(lab):
    request_timeouts = []

    def slow_activation(**request):
        remaining = 180 - lab.clock.now
        assert 0 < request["timeout"] <= min(30, remaining)
        request_timeouts.append(request["timeout"])
        lab.clock.now += min(7, request["timeout"])
        raise denied({"message": ACTIVATION_MESSAGE})

    lab.client.messages.count_tokens.side_effect = slow_activation
    with pytest.raises(TimeoutError):
        execute(lab)
    assert lab.clock.now == 180
    assert min(request_timeouts) < 30
    assert lab.client.closed


@pytest.mark.parametrize("body", [
    {"message": "Access denied for this model."},
    {"message": "You are not authorized to perform this action."},
    {"message": "Your subscription to the model is not being set up."},
    {"message": "Access denied. " + ACTIVATION_MESSAGE},
    {"error": {"message": "Subscription approval is required."}},
    None,
])
def test_unrelated_403_propagates_immediately_without_sleep_or_substitution(lab, body):
    error = denied(body)
    lab.client.messages.count_tokens.side_effect = error
    with pytest.raises(anthropic.PermissionDeniedError) as caught:
        execute(lab)
    assert caught.value is error
    assert lab.client.messages.count_tokens.call_count == 1
    assert not lab.clock.sleeps
    assert lab.client.closed
    assert lab.namespace["counted_input_tokens"] is None


def test_matching_text_with_different_status_is_not_retried(lab):
    error = denied({"message": ACTIVATION_MESSAGE}, status=400)
    lab.client.messages.count_tokens.side_effect = error
    with pytest.raises(anthropic.PermissionDeniedError) as caught:
        execute(lab)
    assert caught.value is error
    assert not lab.clock.sleeps
    assert lab.client.messages.count_tokens.call_count == 1
    assert lab.client.closed


def test_other_request_failure_propagates_and_closes_client(lab):
    error = RuntimeError("Synthetic connection failure")
    lab.client.messages.count_tokens.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        execute(lab)
    assert caught.value is error
    assert not lab.clock.sleeps
    assert lab.client.closed


def test_immediately_ready_count_needs_no_wait(lab):
    lab.client.messages.count_tokens.return_value = SimpleNamespace(input_tokens=61)
    execute(lab)
    assert lab.namespace["counted_input_tokens"] == 61
    assert lab.client.messages.count_tokens.call_count == 1
    assert not lab.clock.sleeps
    assert lab.client.closed


def test_native_count_tokens_path_is_unchanged(lab):
    lab.namespace["SELECTED"].capabilities.support = lambda _: lab.namespace["Support"].SUPPORTED
    lab.namespace["runtime"].count_tokens.return_value = {"inputTokens": 42}
    execute(lab)
    lab.factory.assert_not_called()
    lab.namespace["runtime"].count_tokens.assert_called_once_with(
        modelId=MODEL, input={"converse": lab.namespace["request"]},
    )
    assert lab.namespace["counted_input_tokens"] == 42
    assert not lab.clock.sleeps
