from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

from botocore.exceptions import ClientError

from workshop_utils.readiness import ActivationPending, wait_for_readiness, workshop_checks


def test_activation_retries_without_repeating_success_or_denial():
    now = [0]
    ready = Mock()
    pending = Mock(side_effect=[ActivationPending(), None])
    denied = Mock(side_effect=ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "not authorized"}}, "Converse",
    ))
    states = wait_for_readiness(
        {"ready": ready, "pending": pending, "denied": denied},
        clock=lambda: now[0], sleep=lambda delay: now.__setitem__(0, now[0] + delay),
    )
    assert [states[n]["status"] for n in states] == ["ready", "ready", "failed"]
    assert [ready.call_count, pending.call_count, denied.call_count] == [1, 2, 1]
    assert now[0] == 30


def test_activation_403_times_out_without_becoming_ready():
    now = [0]
    pending = Mock(side_effect=RuntimeError("403: Your subscription to the model is being set up."))
    states = wait_for_readiness(
        {"mantle": pending}, timeout_s=60, interval_s=30,
        clock=lambda: now[0], sleep=lambda delay: now.__setitem__(0, now[0] + delay),
    )
    assert states["mantle"] == {"status": "timed_out", "attempts": 2, "error_code": "RuntimeError"}


def test_checks_cover_actual_apis_and_poll_cohere_without_reinvoking():
    runtime, bedrock, mantle = Mock(), Mock(), Mock()
    runtime.invoke_model.return_value = {"body": io.BytesIO(json.dumps({"results": [{"index": 0}]}).encode())}
    mantle.messages.count_tokens.return_value = SimpleNamespace(input_tokens=8)
    ready = {"authorizationStatus": "AUTHORIZED", "regionAvailability": "AVAILABLE",
             "entitlementAvailability": "AVAILABLE", "agreementAvailability": {"status": "AVAILABLE"}}
    bedrock.get_foundation_model_availability.side_effect = [
        {**ready, "agreementAvailability": {"status": "NOT_AVAILABLE"}}, ready,
    ]
    runtime.converse.return_value = {"usage": {"inputTokens": 8}, "output": {"message": {"content": [{"text": "OK"}]}}}
    now = [0]
    states = wait_for_readiness(
        workshop_checks(runtime, bedrock, mantle, region="us-east-1"),
        clock=lambda: now[0], sleep=lambda delay: now.__setitem__(0, now[0] + delay),
    )
    assert all(item["status"] == "ready" for item in states.values())
    assert runtime.converse.call_count == 3
    assert runtime.invoke_model.call_count == 1
    assert bedrock.get_foundation_model_availability.call_count == 2
    assert mantle.messages.create.call_count == mantle.messages.count_tokens.call_count == 1
    requests = [call.kwargs for call in runtime.converse.call_args_list]
    assert {request["modelId"] for request in requests} == {
        "global.anthropic.claude-sonnet-5", "global.anthropic.claude-opus-5",
        "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    }
    assert all("temperature" not in request["inferenceConfig"] for request in requests)


def test_cli_uses_closeable_boto_clients_and_saves_no_credentials(monkeypatch, tmp_path):
    import anthropic
    import boto3

    import workshop_utils.readiness as module

    clients = {name: Mock() for name in ("sts", "bedrock-runtime", "bedrock")}
    clients["sts"].get_caller_identity.return_value = {
        "Account": "123456789012", "Arn": "arn:aws:sts::123456789012:assumed-role/editor/instance",
    }
    session = Mock()
    session.client.side_effect = lambda name, **kwargs: clients[name]
    monkeypatch.setattr(boto3, "Session", lambda **kwargs: session)
    mantle = Mock()
    mantle.__enter__ = Mock(return_value=mantle)
    mantle.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(anthropic, "AnthropicBedrockMantle", lambda **kwargs: mantle)
    monkeypatch.setattr(module, "workshop_checks", lambda *args, **kwargs: {"probe": lambda: None})
    report = tmp_path / "readiness.json"
    monkeypatch.setattr("sys.argv", ["readiness", "--report", str(report)])
    assert module.main() == 0
    for client in clients.values():
        client.close.assert_called_once()
    result = json.loads(report.read_text())
    assert set(result) == {"region", "account", "principal_arn", "checked_at", "checks"}
    assert result["checks"]["probe"]["status"] == "ready"
