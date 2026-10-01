"""Memory recall supplies retrieved text, not SDK record metadata, to the model."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

NOTEBOOK = Path(__file__).parents[1] / "02-optimization-playbook/02-medium-effort.ipynb"
CELLS = {cell["id"]: "".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text())["cells"]}
WRITE_CELL = "medium-15-83e801ba"
RECALL_CELL = "medium-16-2cf6a06a"
CHECKS_CELL = "medium-memory-scope-corrections"


def execute(cell_id, namespace):
    exec(compile(CELLS[cell_id], str(NOTEBOOK), "exec"), namespace)


class Clock:
    def __init__(self):
        self.now = 0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert 0 < seconds <= 5
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def memory():
    clock = Clock()
    agentcore = SimpleNamespace(
        retrieve_memory_records=Mock(return_value={"memoryRecordSummaries": []}),
        create_event=Mock(return_value={"event": {"eventId": "fixture-event"}}),
        list_events=Mock(return_value={"events": []}),
    )
    control = SimpleNamespace(get_memory=Mock(return_value={"memory": {"status": "ACTIVE"}}), close=Mock())
    namespace = {
        "MEMORY_ID": "fixture-memory", "MEMORY_NAMESPACE": "/users/fixture-actor/facts",
        "MEMORY_NAMESPACE_TEMPLATE": "/users/{actorId}/facts",
        "ACTOR_ID": "fixture-actor", "MEMORY_SESSION": "fixture-session", "event": {"eventId": "existing"},
        "AGENTCORE": agentcore, "os": SimpleNamespace(environ={
            "RUN_MEMORY": "1", "MEMORY_NAMESPACE_TEMPLATE": "/users/{actorId}/facts",
        }),
        "time": clock, "json": json, "SMALL": "fixture-model", "run_case": Mock(), "show": Mock(),
        "print": Mock(), "uuid": uuid, "datetime": datetime, "UTC": UTC,
        "boto3": SimpleNamespace(client=Mock(return_value=control)),
        "REGION": "us-east-1", "SDK_CONFIG": object(),
        "parameter_or_env": Mock(return_value="fixture-memory"),
    }
    return SimpleNamespace(ns=namespace, clock=clock, api=agentcore, control=control)


def test_default_poll_waits_for_late_extraction_and_only_retrieves_existing_actor(memory):
    facts = [{"content": {"text": "Order A-12345; preferred contact is email."}}]
    memory.api.retrieve_memory_records.side_effect = lambda **_: {
        "memoryRecordSummaries": facts if memory.clock.now >= 64 else [],
    }
    identity = {key: memory.ns[key] for key in ("ACTOR_ID", "MEMORY_SESSION", "event")}
    execute(RECALL_CELL, memory.ns)
    assert memory.clock.now == 65
    assert memory.ns["facts"] == facts
    memory.ns["run_case"].assert_called_once()
    assert identity == {key: memory.ns[key] for key in identity}
    assert all(call.kwargs["namespace"] == "/users/fixture-actor/facts"
               for call in memory.api.retrieve_memory_records.call_args_list)
    progress = [call.args[0] for call in memory.ns["print"].call_args_list
                if isinstance(call.args[0], dict) and "remaining_seconds" in call.args[0]]
    assert progress[0]["remaining_seconds"] == 180 and progress[-1]["records_found"] == 1
    assert memory.ns["MEMORY_READY"] is True
    memory.api.create_event.assert_not_called()


def test_poll_timeout_is_bounded_clears_stale_context_and_retries_without_writing(memory):
    memory.ns["memory_context"] = ["Stale recalled facts."]
    execute(RECALL_CELL, memory.ns)
    assert memory.clock.now == 180
    assert memory.api.retrieve_memory_records.call_count == 36
    assert memory.ns["facts"] == memory.ns["memory_context"] == []
    assert memory.ns["MEMORY_READY"] is False
    memory.ns["run_case"].assert_not_called()
    assert any("Rerun this retrieval cell" in str(call) for call in memory.ns["print"].call_args_list)
    memory.api.retrieve_memory_records.return_value = {
        "memoryRecordSummaries": [{"content": {"text": "Order A-12345; email updates."}}],
    }
    execute(RECALL_CELL, memory.ns)
    assert memory.clock.now == 180  # Immediate readiness on the second read; no real sleeps.
    assert memory.ns["MEMORY_READY"] is True
    memory.ns["run_case"].assert_called_once()
    memory.api.create_event.assert_not_called()
    assert memory.ns["ACTOR_ID"] == "fixture-actor" and memory.ns["event"] == {"eventId": "existing"}


def test_partial_records_keep_polling_until_order_and_email_are_both_retrievable(memory):
    partial = [
        {"content": {"text": "The monitor order number is A-12345."}},
        {"content": {"text": "The user uses the Pro plan."}},
        {"content": {"text": "The monitor has a dead pixel."}},
    ]
    complete = [*partial[:2], {"content": {"text": "The user prefers email updates."}}, partial[2]]
    identity = {key: memory.ns[key] for key in ("ACTOR_ID", "MEMORY_SESSION", "event")}

    def retrieve(**request):
        assert request["namespace"] == "/users/fixture-actor/facts"
        assert request["searchCriteria"] == {
            "searchQuery": "What is my monitor order number and preferred contact method?", "topK": 5,
        }
        if memory.clock.now < 105:
            return {"memoryRecordSummaries": []}
        return {"memoryRecordSummaries": partial if memory.clock.now < 135 else complete}

    def answer(*args, **kwargs):
        assert memory.clock.now == 135  # No call on first nonempty retrieval.
        context = json.loads(kwargs["system"][0]["text"].split("\n", 1)[1])
        assert context == [record["content"]["text"] for record in complete]
        return {"text": "Offline response"}

    memory.api.retrieve_memory_records.side_effect = retrieve
    memory.ns["run_case"].side_effect = answer
    execute(RECALL_CELL, memory.ns)
    assert memory.clock.now == 135 and memory.ns["MEMORY_READY"] is True
    assert memory.ns["facts"] == complete
    memory.ns["run_case"].assert_called_once()
    progress = [call.args[0] for call in memory.ns["print"].call_args_list
                if isinstance(call.args[0], dict) and "expected_facts_ready" in call.args[0]]
    partial_progress = [row for row in progress if row["records_found"] == 3]
    assert partial_progress and all(not row["expected_facts_ready"] for row in partial_progress)
    assert all(row["fixture_checks"] == {"order": True, "email_preference": False} for row in partial_progress)
    assert progress[-1]["expected_facts_ready"]
    assert identity == {key: memory.ns[key] for key in identity}
    memory.api.create_event.assert_not_called()


@pytest.mark.parametrize("partial_text,ready_checks", [
    ("The monitor order is A-12345.", {"order": True, "email_preference": False}),
    ("The user prefers email updates.", {"order": False, "email_preference": True}),
    ("Order A-12345. An email service is unavailable.", {"order": True, "email_preference": False}),
])
def test_partial_timeout_keeps_actual_records_but_skips_recall_and_read_model_checks(memory, partial_text, ready_checks):
    partial = [{"content": {"text": partial_text}}]
    memory.api.retrieve_memory_records.return_value = {"memoryRecordSummaries": partial}
    # A previous successful recall must not leave readiness or its context behind.
    memory.ns.update(MEMORY_READY=True, memory_context=["Order A-12345; prefers email."])
    memory.ns["os"].environ["RUN_MEMORY_CHECKS"] = "1"
    execute(RECALL_CELL, memory.ns)
    assert memory.clock.now == 180
    assert memory.ns["MEMORY_READY"] is False and memory.ns["memory_fixture_checks"] == ready_checks
    assert memory.ns["facts"] == partial and memory.ns["memory_context"] == [partial_text]
    displayed = [call.args[0] for call in memory.ns["print"].call_args_list
                 if isinstance(call.args[0], dict) and "records" in call.args[0]]
    assert displayed == [{"records": partial, "fixture_checks": ready_checks, "ready": False}]
    retrieval_count = memory.api.retrieve_memory_records.call_count
    execute(CHECKS_CELL, memory.ns)
    assert memory.ns["MEMORY_READY"] is False
    assert memory.api.retrieve_memory_records.call_count == retrieval_count  # No other-actor read either.
    memory.ns["run_case"].assert_not_called()
    memory.api.create_event.assert_not_called()
    assert any("same actor, namespace, and event" in str(call) for call in memory.ns["print"].call_args_list)


def test_ready_fixture_allows_existing_read_model_checks_without_writes(memory):
    memory.api.retrieve_memory_records.return_value = {
        "memoryRecordSummaries": [{"content": {"text": "Order A-12345; prefers email updates."}}],
    }
    memory.ns["os"].environ["RUN_MEMORY_CHECKS"] = "1"
    execute(RECALL_CELL, memory.ns)
    memory.api.retrieve_memory_records.return_value = {"memoryRecordSummaries": []}
    execute(CHECKS_CELL, memory.ns)
    assert memory.ns["MEMORY_READY"] is True
    assert memory.ns["run_case"].call_count == 3  # Recall plus the correction and unknown-fact checks.
    memory.api.create_event.assert_not_called()


def test_poll_window_can_be_extended_to_300_seconds(memory):
    memory.ns["os"].environ["MEMORY_MAX_WAIT_SECONDS"] = "300"
    execute(RECALL_CELL, memory.ns)
    assert memory.clock.now == 300 and memory.ns["MEMORY_READY"] is False
    assert memory.api.retrieve_memory_records.call_count == 60
    memory.ns["run_case"].assert_not_called()


@pytest.mark.parametrize("wait", ["0", "-1", "301", "inf", "nan"])
def test_invalid_poll_window_fails_before_any_retrieval(memory, wait):
    memory.ns["os"].environ["MEMORY_MAX_WAIT_SECONDS"] = wait
    with pytest.raises(ValueError, match="wait limit"):
        execute(RECALL_CELL, memory.ns)
    memory.api.retrieve_memory_records.assert_not_called()


def test_short_poll_window_does_not_start_a_request_after_deadline(memory):
    memory.ns["os"].environ["MEMORY_MAX_WAIT_SECONDS"] = "7"
    execute(RECALL_CELL, memory.ns)
    assert memory.clock.now == 7 and memory.clock.sleeps == [5, 2]
    assert memory.api.retrieve_memory_records.call_count == 2


def test_inflight_retrieval_may_finish_after_deadline_but_does_not_start_another(memory):
    def slow_retrieval(**_):
        memory.clock.now += 181
        return {"memoryRecordSummaries": []}

    memory.api.retrieve_memory_records.side_effect = slow_retrieval
    execute(RECALL_CELL, memory.ns)
    memory.api.retrieve_memory_records.assert_called_once()
    assert not memory.clock.sleeps
    memory.ns["run_case"].assert_not_called()


def test_same_kernel_write_retry_reuses_event_and_identity(memory):
    execute(WRITE_CELL, memory.ns)
    state = dict(memory.ns["MEMORY_WRITE_STATE"])
    execute(WRITE_CELL, memory.ns)
    memory.api.create_event.assert_called_once()
    assert memory.ns["MEMORY_WRITE_STATE"] == state
    assert memory.api.list_events.call_args.kwargs["actorId"] == state["actor"]
    assert memory.ns["event"] is state["event"]
    assert memory.control.close.call_count == 2


def test_write_timeout_reuses_the_exact_request_and_idempotency_token(memory):
    response = {"event": {"eventId": "fixture-event"}}
    memory.api.create_event.side_effect = [TimeoutError("Response lost"), response]
    with pytest.raises(TimeoutError):
        execute(WRITE_CELL, memory.ns)
    first_request = memory.api.create_event.call_args.kwargs
    assert memory.ns["MEMORY_WRITE_STATE"]["event"] is None
    execute(WRITE_CELL, memory.ns)
    assert memory.api.create_event.call_count == 2
    assert memory.api.create_event.call_args.kwargs == first_request
    assert memory.ns["event"] is response


def test_list_failure_after_write_still_keeps_event_for_retry(memory):
    memory.api.list_events.side_effect = [RuntimeError("List failed"), {"events": []}]
    with pytest.raises(RuntimeError, match="List failed"):
        execute(WRITE_CELL, memory.ns)
    execute(WRITE_CELL, memory.ns)
    memory.api.create_event.assert_called_once()


@pytest.mark.parametrize("change", ["resource", "namespace"])
def test_changed_memory_configuration_cannot_silently_create_another_event(memory, change):
    execute(WRITE_CELL, memory.ns)
    before = dict(memory.ns["MEMORY_WRITE_STATE"])
    if change == "resource":
        memory.ns["parameter_or_env"].return_value = "different-memory"
    else:
        memory.ns["os"].environ["MEMORY_NAMESPACE_TEMPLATE"] = "/other/{actorId}/facts"
    with pytest.raises(ValueError, match="configuration changed"):
        execute(WRITE_CELL, memory.ns)
    assert memory.ns["MEMORY_WRITE_STATE"] == before
    memory.api.create_event.assert_called_once()


def test_disabled_memory_cells_make_no_service_or_model_calls(memory):
    memory.ns["os"].environ["RUN_MEMORY"] = "0"
    execute(WRITE_CELL, memory.ns)
    execute(RECALL_CELL, memory.ns)
    memory.ns["boto3"].client.assert_not_called()
    memory.api.create_event.assert_not_called()
    memory.api.retrieve_memory_records.assert_not_called()
    memory.ns["run_case"].assert_not_called()


def test_memory_context_accepts_sdk_datetime_fields_without_leaking_metadata():
    notebook = json.loads((Path(__file__).parents[1] / '02-optimization-playbook/02-medium-effort.ipynb').read_text())
    cell = next(cell for cell in notebook['cells'] if cell['cell_type'] == 'code'
                and 'retrieve_memory_records' in ''.join(cell['source']))
    facts = [{'memoryRecordId': 'service-id', 'createdAt': datetime(2026, 9, 25, tzinfo=UTC),
              'content': {'text': 'Order A-12345; preferred contact is email.'},
              'metadata': {'updatedAt': {'dateTimeValue': datetime(2026, 9, 25, tzinfo=UTC)}}}]
    run_case = Mock(return_value={'text': 'A-12345, email'})
    namespace = {'MEMORY_ID': 'owned-memory', 'MEMORY_NAMESPACE': '/users/fixture/facts',
                 'AGENTCORE': SimpleNamespace(retrieve_memory_records=Mock(return_value={'memoryRecordSummaries': facts})),
                 'os': SimpleNamespace(environ={}), 'time': SimpleNamespace(monotonic=lambda: 0, sleep=Mock()),
                 'json': json, 'SMALL': 'model-fixture', 'run_case': run_case, 'show': Mock()}
    exec(compile(''.join(cell['source']), '<memory-recall-cell>', 'exec'), namespace)
    context = run_case.call_args.kwargs['system'][0]['text']
    assert 'Order A-12345; preferred contact is email.' in context
    assert 'service-id' not in context
    assert 'createdAt' not in context and 'updatedAt' not in context
    assert 'transcript' not in context
