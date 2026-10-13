"""Memory recall supplies retrieved text, not SDK record metadata, to the model."""
from __future__ import annotations

import copy
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
ANSWER_CELL = "medium-memory-answer"
CORRECTION_CELL = "medium-memory-correction"
CORRECTION_READ_CELL = "medium-memory-correction-retrieve"
CORRECTION_ANSWER_CELL = "medium-memory-correction-answer"
UNKNOWN_CELL = "medium-memory-unknown"
OTHER_CELL = "medium-memory-other-customer"


def execute(cell_id, namespace):
    namespace.setdefault("pretty_json", namespace.get("print", print))
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
    control = SimpleNamespace(get_memory=Mock(return_value={"memory": {
        "status": "ACTIVE", "strategies": [{
            "type": "SEMANTIC", "namespaces": ["/users/{actorId}/facts"],
        }],
    }}), close=Mock())
    namespace = {
        "MEMORY_ID": "fixture-memory", "MEMORY_NAMESPACE": "/users/fixture-actor/facts/",
        "MEMORY_NAMESPACE_TEMPLATE": "/users/{actorId}/facts",
        "ACTOR_ID": "fixture-actor", "MEMORY_SESSION": "fixture-session", "event": {"eventId": "existing"},
        "AGENTCORE": agentcore, "os": SimpleNamespace(environ={
            "MEMORY_NAMESPACE_TEMPLATE": "/users/{actorId}/facts",
        }),
        "time": clock, "json": json, "SMALL": "fixture-model", "run_case": Mock(), "show": Mock(),
        "print": Mock(), "uuid": uuid, "datetime": datetime, "UTC": UTC,
        "boto3": SimpleNamespace(client=Mock(return_value=control)),
        "REGION": "us-east-1", "SDK_CONFIG": object(),
        "parameter_or_env": Mock(return_value="fixture-memory"),
    }
    return SimpleNamespace(ns=namespace, clock=clock, api=agentcore, control=control)


@pytest.fixture
def correction_backend(memory):
    old = {"memoryRecordId": "mem-" + "a" * 40,
           "content": {"text": "Alex's monitor order is A-12345; prefers email updates."}}
    corrected = {"memoryRecordId": "mem-" + "b" * 40,
                 "content": {"text": "Alex corrected the monitor order to A-12346. A-12345 was a mistake."}}
    events = {}

    def create_event(**request):
        events.setdefault(request["clientToken"], copy.deepcopy(request))
        return {"event": {"eventId": "correction-event", "actorId": request["actorId"],
                          "sessionId": request["sessionId"]}}

    memory.api.create_event.side_effect = create_event
    memory.api.batch_update_memory_records = Mock(side_effect=AssertionError("Do not edit extracted records"))
    memory.api.get_memory_record = Mock(side_effect=AssertionError("Observe asynchronous extraction through retrieval"))
    memory.api.retrieve_memory_records.side_effect = lambda **_: {
        "memoryRecordSummaries": copy.deepcopy([old, corrected] if events and memory.clock.now >= 15 else [old]),
    }
    memory.ns.update(MEMORY_READY=True, facts=[copy.deepcopy(old)], memory_context=[old["content"]["text"]])
    return SimpleNamespace(memory=memory, old=old, corrected=corrected, events=events, create=create_event)


def test_correction_saves_a_followup_for_same_actor_and_waits_for_extraction(correction_backend):
    from botocore.session import Session
    from botocore.validate import validate_parameters

    backend = correction_backend
    memory = backend.memory
    execute(CORRECTION_CELL, memory.ns)
    request = memory.api.create_event.call_args.kwargs
    shape = Session().get_service_model("bedrock-agentcore").operation_model("CreateEvent").input_shape
    validate_parameters(request, shape)
    assert request["memoryId"] == memory.ns["MEMORY_ID"]
    assert request["actorId"] == memory.ns["ACTOR_ID"]
    assert request["sessionId"] != memory.ns["MEMORY_SESSION"]
    messages = [item["conversational"] for item in request["payload"]]
    assert [item["role"] for item in messages] == ["USER", "ASSISTANT"]
    assert "A-12346" in messages[0]["content"]["text"] and "was a mistake" in messages[0]["content"]["text"]
    assert "A-12346" in messages[1]["content"]["text"]
    assert memory.ns["CORRECTION_SAVED"] and not memory.ns["CORRECTION_READY"]
    memory.ns["run_case"].assert_not_called()  # The two saved messages are clearly labelled sample input.
    execute(CORRECTION_READ_CELL, memory.ns)
    assert memory.clock.now == 15 and memory.ns["CORRECTION_READY"]
    assert memory.ns["corrected_records"] == [backend.old, backend.corrected]
    assert memory.ns["correction_progress"][0]["new_number_found"] is False
    assert memory.ns["correction_progress"][-1] == {"new_number_found": True, "old_number_also_found": True}
    execute(CORRECTION_ANSWER_CELL, memory.ns)
    args, kwargs = memory.ns["run_case"].call_args
    assert args[1] == "What is my monitor order number?"
    context = json.loads(kwargs["system"][0]["text"].split("\n", 1)[1])
    assert context == [backend.old["content"]["text"], backend.corrected["content"]["text"]]
    assert "explicit correction" in kwargs["system"][0]["text"]
    memory.api.batch_update_memory_records.assert_not_called()
    memory.api.get_memory_record.assert_not_called()
    memory.api.create_event.assert_called_once()


def test_followup_retry_reuses_event_without_saving_another_conversation(correction_backend):
    memory = correction_backend.memory
    execute(CORRECTION_CELL, memory.ns)
    state = copy.deepcopy(memory.ns["MEMORY_CORRECTION_STATE"])
    execute(CORRECTION_CELL, memory.ns)
    assert memory.ns["MEMORY_CORRECTION_STATE"] == state
    memory.api.create_event.assert_called_once()


def test_followup_timeout_retries_with_identical_idempotency_token_and_payload(correction_backend):
    backend = correction_backend
    memory = backend.memory

    def save_then_timeout(**request):
        backend.create(**request)
        raise TimeoutError("Event saved; response lost")

    memory.api.create_event.side_effect = save_then_timeout
    with pytest.raises(TimeoutError):
        execute(CORRECTION_CELL, memory.ns)
    assert not memory.ns["CORRECTION_SAVED"]
    first = copy.deepcopy(memory.api.create_event.call_args.kwargs)
    memory.api.create_event.side_effect = backend.create
    execute(CORRECTION_CELL, memory.ns)
    assert memory.api.create_event.call_args.kwargs == first
    assert len(backend.events) == 1 and memory.ns["CORRECTION_SAVED"]


@pytest.mark.parametrize("changed_key", ["MEMORY_ID", "ACTOR_ID", "MEMORY_NAMESPACE"])
def test_followup_state_cannot_be_reused_for_another_customer_or_resource(correction_backend, changed_key):
    memory = correction_backend.memory
    execute(CORRECTION_CELL, memory.ns)
    memory.ns[changed_key] = "another-customer-or-resource"
    with pytest.raises(ValueError, match="different customer or Memory resource"):
        execute(CORRECTION_CELL, memory.ns)
    memory.api.create_event.assert_called_once()


@pytest.mark.parametrize("raw_namespace", ["/users/fixture-actor/facts", "///users/fixture-actor/facts///"])
def test_followup_retrieval_uses_normalized_customer_namespace(correction_backend, raw_namespace):
    memory = correction_backend.memory
    memory.ns["MEMORY_NAMESPACE"] = raw_namespace
    execute(CORRECTION_CELL, memory.ns)
    execute(CORRECTION_READ_CELL, memory.ns)
    assert all(call.kwargs["namespace"] == "/users/fixture-actor/facts/"
               for call in memory.api.retrieve_memory_records.call_args_list)


def test_extraction_timeout_does_not_inject_correction_or_generate_answer(correction_backend):
    backend = correction_backend
    memory = backend.memory
    execute(CORRECTION_CELL, memory.ns)
    memory.api.retrieve_memory_records.side_effect = None
    memory.api.retrieve_memory_records.return_value = {"memoryRecordSummaries": [backend.old]}
    execute(CORRECTION_READ_CELL, memory.ns)
    execute(CORRECTION_ANSWER_CELL, memory.ns)
    assert memory.clock.now == 180 and not memory.ns["CORRECTION_READY"]
    memory.ns["run_case"].assert_not_called()
    assert memory.ns["memory_context"] == [backend.old["content"]["text"]]
    memory.api.retrieve_memory_records.return_value = {"memoryRecordSummaries": [backend.corrected]}
    execute(CORRECTION_READ_CELL, memory.ns)
    assert memory.ns["CORRECTION_READY"]
    memory.api.create_event.assert_called_once()


def test_correction_retrieval_error_does_not_resubmit_event_or_generate_answer(correction_backend):
    memory = correction_backend.memory
    execute(CORRECTION_CELL, memory.ns)
    memory.api.retrieve_memory_records.side_effect = RuntimeError("Read failed")
    with pytest.raises(RuntimeError, match="Read failed"):
        execute(CORRECTION_READ_CELL, memory.ns)
    assert not memory.ns["CORRECTION_READY"]
    memory.ns["run_case"].assert_not_called()
    memory.api.create_event.assert_called_once()


def test_initial_retrieval_normalizes_an_existing_namespace_before_requests(memory):
    memory.ns["MEMORY_NAMESPACE"] = "/users/fixture-actor/facts"
    memory.api.retrieve_memory_records.return_value = {
        "memoryRecordSummaries": [{"content": {"text": "Order A-12345; prefers email updates."}}],
    }
    execute(RECALL_CELL, memory.ns)
    assert memory.api.retrieve_memory_records.call_args.kwargs["namespace"] == "/users/fixture-actor/facts/"
    memory.api.create_event.assert_not_called()


def test_default_poll_waits_for_late_extraction_and_only_retrieves_existing_actor(memory):
    facts = [{"content": {"text": "Order A-12345; preferred contact is email."}}]
    memory.api.retrieve_memory_records.side_effect = lambda **_: {
        "memoryRecordSummaries": facts if memory.clock.now >= 64 else [],
    }
    identity = {key: memory.ns[key] for key in ("ACTOR_ID", "MEMORY_SESSION", "event")}
    execute(RECALL_CELL, memory.ns)
    memory.ns["run_case"].assert_not_called()
    execute(ANSWER_CELL, memory.ns)
    assert memory.clock.now == 65
    assert memory.ns["facts"] == facts
    memory.ns["run_case"].assert_called_once()
    assert identity == {key: memory.ns[key] for key in identity}
    assert all(call.kwargs["namespace"] == "/users/fixture-actor/facts/"
               for call in memory.api.retrieve_memory_records.call_args_list)
    progress = memory.ns["memory_progress"]
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
    execute(ANSWER_CELL, memory.ns)
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
        assert request["namespace"] == "/users/fixture-actor/facts/"
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
    memory.ns["run_case"].assert_not_called()
    execute(ANSWER_CELL, memory.ns)
    assert memory.clock.now == 135 and memory.ns["MEMORY_READY"] is True
    assert memory.ns["facts"] == complete
    memory.ns["run_case"].assert_called_once()
    progress = memory.ns["memory_progress"]
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
    execute(RECALL_CELL, memory.ns)
    assert memory.clock.now == 180
    assert memory.ns["MEMORY_READY"] is False and memory.ns["memory_fixture_checks"] == ready_checks
    assert memory.ns["facts"] == partial and memory.ns["memory_context"] == [partial_text]
    displayed = [call.args[0] for call in memory.ns["print"].call_args_list
                 if isinstance(call.args[0], dict) and "records" in call.args[0]]
    assert displayed == [{"records": partial, "fixture_checks": ready_checks, "ready": False}]
    retrieval_count = memory.api.retrieve_memory_records.call_count
    for cell_id in (CORRECTION_CELL, UNKNOWN_CELL, OTHER_CELL):
        execute(cell_id, memory.ns)
    assert memory.ns["MEMORY_READY"] is False
    assert memory.api.retrieve_memory_records.call_count == retrieval_count  # No other-actor read either.
    memory.ns["run_case"].assert_not_called()
    memory.api.create_event.assert_not_called()
    assert any("Retrieve Alex" in str(call) for call in memory.ns["print"].call_args_list)


def test_ready_fixture_allows_existing_read_model_checks_without_writes(memory):
    memory.api.retrieve_memory_records.return_value = {
        "memoryRecordSummaries": [{"content": {"text": "Order A-12345; prefers email updates."}}],
    }
    execute(RECALL_CELL, memory.ns)
    memory.ns["run_case"].assert_not_called()
    execute(ANSWER_CELL, memory.ns)
    memory.api.retrieve_memory_records.return_value = {"memoryRecordSummaries": []}
    memory.ns.update(CORRECTION_READY=True, corrected_context=["Order A-12346; prefers email updates."])
    for cell_id in (CORRECTION_ANSWER_CELL, UNKNOWN_CELL, OTHER_CELL):
        execute(cell_id, memory.ns)
    assert memory.ns["MEMORY_READY"] is True
    assert memory.ns["run_case"].call_count == 3  # Recall plus the correction and unknown-fact checks.
    memory.api.create_event.assert_not_called()


@pytest.mark.parametrize("cell_id,label,question", [
    (ANSWER_CELL, "L10-new-session-memory", "What is my monitor order number and preferred contact method?"),
    (CORRECTION_ANSWER_CELL, "L10-corrected-memory", "What is my monitor order number?"),
    (UNKNOWN_CELL, "L10-unrelated", "What is my passport number?"),
])
def test_each_question_cell_makes_one_fresh_request_with_retrieved_text_only(memory, cell_id, label, question):
    context = ["Order A-12346; prefers email updates."]
    memory.ns.update(MEMORY_READY=True, CORRECTION_READY=True,
                     corrected_context=context.copy(), memory_context=context.copy())
    execute(cell_id, memory.ns)
    memory.ns["run_case"].assert_called_once()
    args, kwargs = memory.ns["run_case"].call_args
    assert args == ("fixture-model", question) and kwargs["label"] == label
    assert "messages" not in kwargs  # No earlier conversation is replayed.
    assert json.loads(kwargs["system"][0]["text"].split("\n", 1)[1]) == context
    assert memory.ns["memory_context"] == context
    memory.ns["show"].assert_called_once_with(memory.ns["run_case"].return_value)
    memory.api.create_event.assert_not_called()
    memory.api.retrieve_memory_records.assert_not_called()


@pytest.mark.parametrize("cell_id", [ANSWER_CELL, CORRECTION_CELL, CORRECTION_READ_CELL,
                                    CORRECTION_ANSWER_CELL, UNKNOWN_CELL, OTHER_CELL])
def test_separate_followup_steps_wait_for_successful_retrieval(memory, cell_id):
    memory.ns.update(MEMORY_READY=False, CORRECTION_SAVED=False, CORRECTION_READY=False)
    execute(cell_id, memory.ns)
    memory.ns["run_case"].assert_not_called()
    memory.api.create_event.assert_not_called()
    memory.api.retrieve_memory_records.assert_not_called()


def test_other_customer_check_surfaces_unexpected_records_without_a_model_call(memory):
    memory.ns["MEMORY_READY"] = True
    unexpected = [{"content": {"text": "Unexpected cross-customer record."}}]
    memory.api.retrieve_memory_records.return_value = {"memoryRecordSummaries": unexpected}
    with pytest.raises(AssertionError, match="Unexpected memory records"):
        execute(OTHER_CELL, memory.ns)
    assert memory.ns["other_records"] == unexpected
    assert memory.api.retrieve_memory_records.call_args.kwargs["namespace"] != memory.ns["MEMORY_NAMESPACE"]
    memory.ns["run_case"].assert_not_called()
    memory.api.create_event.assert_not_called()


def test_other_customer_namespace_is_normalized(memory):
    memory.ns["MEMORY_READY"] = True
    memory.ns["MEMORY_NAMESPACE_TEMPLATE"] = "users/{actorId}/facts"
    execute(OTHER_CELL, memory.ns)
    namespace = memory.api.retrieve_memory_records.call_args.kwargs["namespace"]
    assert namespace.startswith("/users/playbook-") and namespace.endswith("/facts/")
    assert namespace != memory.ns["MEMORY_NAMESPACE"]


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
        memory.control.get_memory.return_value["memory"]["strategies"][0]["namespaces"] = [
            "/other/{actorId}/facts",
        ]
    with pytest.raises(ValueError, match="configuration changed"):
        execute(WRITE_CELL, memory.ns)
    assert memory.ns["MEMORY_WRITE_STATE"] == before
    memory.api.create_event.assert_called_once()


def test_required_memory_cells_store_and_recall_without_a_run_flag(memory):
    memory.api.retrieve_memory_records.return_value = {
        "memoryRecordSummaries": [{"content": {"text": "Order A-12345; prefers email updates."}}],
    }
    execute(WRITE_CELL, memory.ns)
    execute(RECALL_CELL, memory.ns)
    memory.ns["run_case"].assert_not_called()
    execute(ANSWER_CELL, memory.ns)
    memory.api.create_event.assert_called_once()
    memory.api.retrieve_memory_records.assert_called_once()
    memory.ns["run_case"].assert_called_once()
    assert "RUN_MEMORY" not in memory.ns


def test_default_settings_discover_namespace_from_workshop_resource(memory):
    import yaml

    template = yaml.load(
        (NOTEBOOK.parents[1] / "03-developer-journey/prerequisite/infrastructure.yaml").read_text(),
        Loader=yaml.BaseLoader,
    )
    configured = template["Resources"]["PlaybookMemory"]["Properties"]["MemoryStrategies"][0][
        "SemanticMemoryStrategy"
    ]["Namespaces"]
    memory.control.get_memory.return_value["memory"]["strategies"][0]["namespaces"] = configured
    memory.ns["os"].environ = {}
    memory.ns.pop("MEMORY_NAMESPACE_TEMPLATE")
    assert "medium-memory-options" not in CELLS
    execute(WRITE_CELL, memory.ns)
    memory.ns["parameter_or_env"].assert_called_once_with(
        "/app/customersupport/agentcore/memory_id", "MEMORY_ID",
    )
    memory.control.get_memory.assert_called_once_with(memoryId="fixture-memory")
    assert memory.ns["MEMORY_NAMESPACE_TEMPLATE"] == "/users/{actorId}/facts"
    assert memory.ns["MEMORY_NAMESPACE"] == f"/users/{memory.ns['ACTOR_ID']}/facts/"
    memory.api.create_event.assert_called_once()
    memory.control.close.assert_called_once()


def test_namespace_templates_response_field_is_supported(memory):
    memory.ns["os"].environ.pop("MEMORY_NAMESPACE_TEMPLATE", None)
    memory.control.get_memory.return_value["memory"]["strategies"] = [
        {"type": "SEMANTIC", "namespaceTemplates": ["/users/{actorId}/facts"]},
        {"type": "USER_PREFERENCE", "namespaces": ["/users/{actorId}/preferences"]},
    ]
    execute(WRITE_CELL, memory.ns)
    assert memory.ns["MEMORY_NAMESPACE_TEMPLATE"] == "/users/{actorId}/facts"
    memory.api.create_event.assert_called_once()


@pytest.mark.parametrize("namespaces", [[], ["/users/{actorId}/one", "/users/{actorId}/two"]])
def test_missing_or_ambiguous_namespace_is_not_guessed_before_writing(memory, namespaces):
    memory.ns["os"].environ.pop("MEMORY_NAMESPACE_TEMPLATE", None)
    memory.control.get_memory.return_value["memory"]["strategies"][0]["namespaces"] = namespaces
    with pytest.raises(ValueError, match="one per-customer semantic namespace"):
        execute(WRITE_CELL, memory.ns)
    memory.api.create_event.assert_not_called()
    memory.control.close.assert_called_once()


def test_namespace_environment_value_does_not_override_provisioned_configuration(memory):
    memory.ns["os"].environ["MEMORY_NAMESPACE_TEMPLATE"] = "/users/{actorId}/wrong"
    execute(WRITE_CELL, memory.ns)
    assert memory.ns["MEMORY_NAMESPACE_TEMPLATE"] == "/users/{actorId}/facts"
    memory.api.create_event.assert_called_once()


def test_memory_configuration_read_failure_propagates_without_writing(memory):
    memory.control.get_memory.side_effect = RuntimeError("Access denied")
    with pytest.raises(RuntimeError, match="Access denied"):
        execute(WRITE_CELL, memory.ns)
    memory.api.create_event.assert_not_called()
    memory.control.close.assert_called_once()


@pytest.mark.parametrize("missing", ["MEMORY_ID", "MEMORY_NAMESPACE"])
def test_required_recall_rejects_missing_setup_before_any_requests(memory, missing):
    memory.ns[missing] = ""
    with pytest.raises(ValueError, match="setup and store cell"):
        execute(RECALL_CELL, memory.ns)
    memory.api.retrieve_memory_records.assert_not_called()
    memory.ns["run_case"].assert_not_called()


@pytest.mark.parametrize("missing", ["resource", "namespace"])
def test_required_write_preserves_configuration_checks_before_writing(memory, missing):
    if missing == "resource":
        memory.ns["parameter_or_env"].return_value = ""
    else:
        memory.control.get_memory.return_value["memory"]["strategies"][0]["namespaces"] = ["/shared/facts"]
    with pytest.raises(ValueError):
        execute(WRITE_CELL, memory.ns)
    memory.api.create_event.assert_not_called()


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
                 'json': json, 'SMALL': 'model-fixture', 'run_case': run_case, 'show': Mock(),
                 'pretty_json': Mock()}
    exec(compile(''.join(cell['source']), '<memory-recall-cell>', 'exec'), namespace)
    run_case.assert_not_called()
    execute(ANSWER_CELL, namespace)
    context = run_case.call_args.kwargs['system'][0]['text']
    assert 'Order A-12345; preferred contact is email.' in context
    assert 'service-id' not in context
    assert 'createdAt' not in context and 'updatedAt' not in context
    assert 'transcript' not in context
