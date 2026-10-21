"""Offline regression checks for the context exercise; no AWS clients."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from IPython.display import HTML

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "context_management", ROOT / "02-optimization-playbook/utils/context_management.py"
)
cm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cm
SPEC.loader.exec_module(cm)


def exchange(i, text):
    return cm.Exchange(f"E{i}", text + " " + "Background detail. " * 40, "Acknowledged.")


@pytest.fixture
def state():
    result = cm.RollingContext(
        keep_exchanges=1,
        required_evidence={
            "E1": ["I cannot install software.", "Do not factory-reset the headset."],
            "E2": ["Correction: my headset is H-220, not H-200."],
        },
    )
    result.append(exchange(1, "I cannot install software. Do not factory-reset the headset."))
    result.append(exchange(2, "Correction: my headset is H-220, not H-200."))
    return result


def summary(*ids):
    return json.dumps({"evidence_ids": list(ids)})


FIRST = ["E1:S1", "E1:S2"]


def test_no_summary_path_and_no_eligible_history(state):
    before = state.messages("What next?")
    assert not state.should_compact("full", "What next?", "System", 1)
    assert not state.should_compact("budget", "What next?", "System", 100_000)
    assert state.messages("What next?") == before
    assert state.summary == []
    lone = cm.RollingContext(keep_exchanges=1)
    lone.append(exchange(1, "Keep this."))
    assert not lone.should_compact("frequent", "Next?", "System", 1)


def test_lost_constraint_rejected_without_discarding_history(state):
    before = state.messages("What next?")
    errors = state.accept_summary(summary(FIRST[0]), stop_reason="end_turn")
    assert any("factory-reset" in error for error in errors)
    assert state.messages("What next?") == before
    assert state.summary == []


def test_summary_request_exposes_sentence_ids_without_assistant_or_oracle():
    state = cm.RollingContext(keep_exchanges=1, required_evidence={"E1": ["ORACLE_ONLY"]})
    state.append(cm.Exchange(
        "E1", "Do not reset.\nUse written steps!\tApproval pending?  Wait.", "ASSISTANT_ONLY",
    ))
    state.append(cm.Exchange("E2", "Recent exchange.", "Acknowledged."))
    request = json.loads(state.summary_request())
    assert request == {
        "retained_evidence": [],
        "candidate_sentences": [
            {"id": f"E1:S{i}", "text": sentence} for i, sentence in enumerate(
                ["Do not reset.", "Use written steps!", "Approval pending?", "Wait."], start=1,
            )
        ],
    }
    assert "ORACLE_ONLY" not in state.summary_request()
    assert "ASSISTANT_ONLY" not in state.summary_request()
    state.required_evidence = {}
    assert not state.accept_summary(summary("E1:S1", "E1:S2", "E1:S3", "E1:S4"), stop_reason="end_turn")
    assert state.summary[0] == {"exchange_id": "E1", "quote": "Do not reset."}


@pytest.mark.parametrize("text,stop", [
    ("not json", "end_turn"),
    ("```json\n" + summary(*FIRST) + "\n```", "end_turn"),
    ('{"evidence_ids":[]}', "end_turn"),
    ('{"evidence_ids":"bad"}', "end_turn"),
    ('{"evidence_ids":[], "invented_field":true}', "end_turn"),
    ('{"evidence_ids":[null]}', "end_turn"),
    ('{"evidence":[{"exchange_id":"E1","quote":"I can install software."}]}', "end_turn"),
    (summary("E99:S1"), "end_turn"),
    (summary("E2:S1"), "end_turn"),  # Still in the recent window, not eligible.
    (summary(*FIRST, FIRST[0]), "end_turn"),
    (summary(*FIRST), "max_tokens"),
    (summary(*FIRST), "guardrail_intervened"),
])
def test_invalid_or_incomplete_selection_keeps_role_pairs(state, text, stop):
    before = state.messages("Next?")
    assert state.accept_summary(text, stop_reason=stop)
    assert state.messages("Next?") == before


def test_repeated_selection_carries_old_evidence_and_newly_evicted_pairs(state):
    assert not state.accept_summary(summary(*FIRST), stop_reason="end_turn")
    state.append(exchange(3, "The diagnostic is pending."))
    request = json.loads(state.summary_request())
    assert [item["id"] for item in request["retained_evidence"]] == FIRST
    assert next(s["text"] for s in request["candidate_sentences"] if s["id"] == "E2:S1") == (
        "Correction: my headset is H-220, not H-200."
    )
    assert "E1:S3" not in {s["id"] for s in request["candidate_sentences"]}
    assert not state.accept_summary(summary("E2:S1"), stop_reason="end_turn")
    assert all(quote in state.messages("What next?")[0]["content"][0]["text"]
               for quote in ("I cannot install software.", "Do not factory-reset the headset."))
    messages = state.messages("What next?")
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert "H-220" in messages[0]["content"][0]["text"]
    assert len(state.archive) == 3


def test_empty_increment_keeps_prior_evidence_when_new_history_is_incidental():
    state = cm.RollingContext(keep_exchanges=1, required_evidence={
        "E1": ["I cannot install software."],
    })
    state.append(cm.Exchange("E1", "I cannot install software.", "Understood."))
    state.append(cm.Exchange("E2", "Thanks for waiting.", "No problem."))
    assert not state.accept_summary(summary("E1:S1"), stop_reason="end_turn")
    state.append(cm.Exchange("E3", "I'm back.", "Welcome back."))
    assert not state.accept_summary(summary(), stop_reason="end_turn")
    assert state.summary == [{"exchange_id": "E1", "quote": "I cannot install software."}]
    assert [item.exchange_id for item in state.recent] == ["E3"]


def test_empty_increment_still_rejects_missing_new_correction(state):
    assert not state.accept_summary(summary(*FIRST), stop_reason="end_turn")
    state.append(exchange(3, "Thanks."))
    before = state.messages("What next?")
    errors = state.accept_summary(summary(), stop_reason="end_turn")
    assert any("Missing required evidence: E2" in error for error in errors)
    assert state.messages("What next?") == before


def test_discarded_archive_sentence_cannot_be_recovered_through_an_unoffered_id(state):
    assert not state.accept_summary(summary(*FIRST), stop_reason="end_turn")
    state.append(exchange(3, "Still waiting."))
    before = state.messages("Next?")
    errors = state.accept_summary(summary(*FIRST, "E2:S1", "E1:S3"), stop_reason="end_turn")
    assert any("unavailable sentence ID: E1:S3" in error for error in errors)
    assert state.messages("Next?") == before


def test_size_trigger_counts_system_summary_and_question_as_characters(state):
    size = state.context_chars("Next?", "System")
    assert state.should_compact("budget", "Next?", "System", size)
    assert not state.should_compact("budget", "Next?", "System", size + 1)
    assert state.context_chars("Next?" * 100, "System") > size
    assert state.context_chars("Next?", "System" * 100) > size
    assert not state.accept_summary(summary(*FIRST), stop_reason="end_turn")
    assert "earlier_evidence" in state.messages("Next?")[0]["content"][0]["text"]


def test_selection_keeps_source_chronology_for_corrections(state):
    state.append(exchange(3, "Still waiting."))
    assert not state.accept_summary(summary("E2:S1", *FIRST), stop_reason="end_turn")
    assert [entry["exchange_id"] for entry in state.summary] == ["E1", "E1", "E2"]


def test_empty_history_is_a_valid_single_question_and_cannot_be_summarized():
    state = cm.RollingContext()
    assert state.messages("Hello") == [{"role": "user", "content": [{"text": "Hello"}]}]
    assert state.accept_summary(summary(*FIRST), stop_reason="end_turn")


def test_rejects_rich_history_and_bad_settings():
    with pytest.raises(ValueError):
        cm.RollingContext(keep_exchanges=0)
    state = cm.RollingContext()
    with pytest.raises(ValueError):
        state.append(cm.Exchange("E1", [{"toolUse": {}}], "result"))
    state.append(exchange(1, "A fact."))
    with pytest.raises(ValueError):
        state.append(exchange(1, "Duplicate."))
    with pytest.raises(ValueError):
        state.should_compact("typo", "Next?", "System", 100)
    with pytest.raises(ValueError):
        state.should_compact("budget", "Next?", "System", 0)


def test_summary_and_answer_costs_accumulate_and_unknown_is_not_zero():
    rows = [
        {"context_kind": kind, "input_tokens": 100, "output_tokens": 20,
         "cache_read_tokens": 30, "cache_write_tokens": 10,
         "cost_usd": cost, "latency_ms": 50, "sdk_retries": 0}
        for kind, cost in [("summary", 0.02), ("summary", 0.03), ("answer", 0.01)]
    ]
    result = cm.call_totals(rows, full_task_latency_ms=190)
    assert result["summary_calls"] == 2 and result["answer_calls"] == 1
    assert result["total_cost_usd"] == pytest.approx(0.06)
    assert result["summary_cost_usd"] == pytest.approx(0.05)
    assert result["handoff_cost_usd"] == pytest.approx(0.01)
    assert result["handoff_input_tokens"] == 140
    assert result["summary_input_tokens"] == 280
    assert result["total_input_tokens"] == 420
    assert result["output_tokens"] == 60 and result["sdk_retries"] == 0
    assert result["full_task_latency_ms"] == 190
    rows[0]["cost_usd"] = None
    assert cm.call_totals(rows, full_task_latency_ms=190)["total_cost_usd"] is None
    assert cm.call_totals(rows, full_task_latency_ms=190)["summary_cost_usd"] is None
    assert cm.call_totals(rows, full_task_latency_ms=190)["handoff_cost_usd"] == pytest.approx(0.01)


def test_quality_gate_detects_lost_constraint_stale_fact_and_false_completion():
    correct = {
        "headset_model": "H-220",
        "constraints": ["no_software_install", "no_factory_reset"],
        "failed_checks": ["cable_swap", "other_device"],
        "next_action": "hardware_diagnostic",
        "diagnostic_status": "pending",
        "replacement_status": "not_approved",
    }
    assert all(cm.answer_checks(json.dumps(correct), "end_turn").values())
    for key, value in [
        ("headset_model", "H-200"),
        ("constraints", ["no_factory_reset"]),
        ("failed_checks", []),
        ("diagnostic_status", "completed"),
        ("replacement_status", "approved"),
        ("replacement_status", "unknown"),
    ]:
        assert not all(cm.answer_checks(json.dumps({**correct, key: value}), "end_turn").values())
    assert not all(cm.answer_checks(json.dumps(correct), "max_tokens").values())
    assert not all(cm.answer_checks("invalid", "end_turn").values())
    assert not all(cm.answer_checks("```json\n" + json.dumps(correct) + "\n```", "end_turn").values())


def test_native_schemas_do_not_encode_the_fixture_answers():
    from jsonschema import validate

    # Schema-valid output must still be allowed to fail the evidence/quality gates.
    validate({"evidence_ids": []}, cm.SUMMARY_SCHEMA)
    validate({"evidence_ids": ["E99:S1"]}, cm.SUMMARY_SCHEMA)
    wrong = {
        "headset_model": "H-200", "constraints": [], "failed_checks": [],
        "next_action": "cable_swap", "diagnostic_status": "completed", "replacement_status": "approved",
    }
    validate(wrong, cm.ANSWER_SCHEMA)
    assert not all(cm.answer_checks(json.dumps(wrong), "end_turn").values())
    assert "H-220" not in json.dumps(cm.ANSWER_SCHEMA)


def notebook_cells():
    notebook = json.loads((ROOT / "02-optimization-playbook/02-medium-effort.ipynb").read_text())
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


def run_notebook_context(*, enabled=True, budget=4200, rejected=False, unpriced=False,
                         retries=0, adapter="medium", model_id=None, fenced=False):
    """Execute the real cells and run_case adapter, replacing only the wire call."""
    from workshop_utils.bedrock import build_converse_request, normalize_usage
    from workshop_utils.models import resolve_model
    from workshop_utils.pricing import (
        AmbiguousCacheUsageError,
        InferredPriceError,
        UnknownPriceError,
        calculate_cost,
    )

    cells = notebook_cells()
    requests = []
    namespace = {
        "os": SimpleNamespace(environ={}),
        "json": json, "time": time, "uuid": uuid, "datetime": datetime, "UTC": UTC,
        "hashlib": hashlib, "boto3": SimpleNamespace(__version__="offline"),
        "Exchange": cm.Exchange, "RollingContext": cm.RollingContext,
        "answer_checks": cm.answer_checks, "call_totals": cm.call_totals,
        "SUMMARY_SCHEMA": cm.SUMMARY_SCHEMA, "ANSWER_SCHEMA": cm.ANSWER_SCHEMA,
        "SMALL": model_id or "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "REGION": "us-east-1", "RUNTIME": object(), "RUN_ID": "offline", "RUN_ROWS": [],
        "resolve_model": resolve_model, "build_converse_request": build_converse_request,
        "normalize_usage": normalize_usage, "calculate_cost": calculate_cost,
        "UnknownPriceError": UnknownPriceError, "InferredPriceError": InferredPriceError,
        "AmbiguousCacheUsageError": AmbiguousCacheUsageError,
        "response_span": lambda *args, **kwargs: None,
        "print": lambda *args: None,
        "pretty_json": lambda value: None,
        "SHOW_CONTEXT_DETAILS": False, "replay_context": cm.replay_context,
        "pd": pytest.importorskip("pandas"), "display": Mock(), "HTML": HTML, "escape": escape,
    }

    def wire_call(client, **request):
        from jsonschema import validate

        assert client is namespace["RUNTIME"]
        requests.append(request)
        system = request["system"][0]["text"]
        if system == namespace["SUMMARY_SYSTEM"]:
            source = json.loads(request["messages"][0]["content"][0]["text"])
            required_quotes = {quote for values in namespace["REQUIRED_EVIDENCE"].values() for quote in values}
            selected = [sentence["id"] for sentence in source["candidate_sentences"]
                        if sentence["text"] in required_quotes
                        and not (rejected and "factory reset" in sentence["text"])]
            text = json.dumps({"evidence_ids": selected})
        else:
            # Answer from the ACTUAL supplied context; dropped facts become failures.
            context = json.dumps(request["messages"])
            text = json.dumps({
                "headset_model": "H-220" if "H-220, not H-200" in context else "unknown",
                "constraints": [
                    value for marker, value in [
                        ("can\'t install software", "no_software_install"),
                        ("or do a factory reset", "no_factory_reset"),
                    ] if marker in context
                ],
                "failed_checks": [
                    value for marker, value in [
                        ("I tried a cable from a working headset, but the left side is still silent.", "cable_swap"),
                        ("Still no sound on the left when I use the tablet.", "other_device"),
                    ] if marker in context
                ],
                "next_action": "hardware_diagnostic" if "haven\'t run the hardware diagnostic yet" in context else "unknown",
                "diagnostic_status": "pending" if "haven\'t run the hardware diagnostic yet" in context else "unknown",
                "replacement_status": "not_approved" if "IT hasn\'t approved a replacement." in context else "unknown",
            })
        schema = request["outputConfig"]["textFormat"]["structure"]["jsonSchema"]
        assert json.loads(schema["schema"]) == (
            cm.SUMMARY_SCHEMA if system == namespace["SUMMARY_SYSTEM"] else cm.ANSWER_SCHEMA
        )
        validate(json.loads(text), json.loads(schema["schema"]))
        if fenced:
            text = "```json\n" + text + "\n```"
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "usage": {"inputTokens": 100, "outputTokens": 20},
            "stopReason": "end_turn",
            "ResponseMetadata": {"RetryAttempts": retries},
        }

    namespace["traced_converse"] = wire_call  # HIGH retains its existing convenience wrapper.
    namespace["RUNTIME"] = SimpleNamespace(converse=lambda **request: wire_call(namespace["RUNTIME"], **request))

    @contextmanager
    def observation(request, *, session_id):
        yield lambda response: response

    namespace["converse_observation"] = observation
    if unpriced:
        def unknown_price(*args, **kwargs):
            raise UnknownPriceError("Offline unknown price")
        namespace["calculate_cost"] = unknown_price
    adapter_source = cells["medium-setup-a4d9ea22"]
    if adapter == "high":
        high = json.loads((ROOT / "02-optimization-playbook/03-high-effort.ipynb").read_text())
        adapter_source = "".join(next(c["source"] for c in high["cells"] if c["id"] == "high-setup-87841c4c"))
    tree = ast.parse(adapter_source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_case"]
    exec(compile(tree, "<real-run-case>", "exec"), namespace)
    settings = ast.parse(cells["medium-context-options"])
    if model_id is not None:
        # Model the participant editing the comparison's explicit model setting.
        for node in settings.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "CONTEXT_MODEL" for target in node.targets
            ):
                node.value = ast.copy_location(ast.Constant(model_id), node.value)
    exec(compile(settings, "<context-settings>", "exec"), namespace)
    fixture = ast.parse((ROOT / "02-optimization-playbook/utils/context_fixture.py").read_text())
    fixture.body = [node for node in fixture.body if not isinstance(node, ast.ImportFrom)]
    exec(compile(fixture, "<context-data>", "exec"), namespace)
    for cell_id in ("medium-context-fixture", "medium-context-prepare"):
        tree = ast.parse(cells[cell_id])
        tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)]
        exec(compile(tree, f"<{cell_id}>", "exec"), namespace)
    if enabled:
        for cell_id in ("medium-context-full", "medium-context-frequent", "medium-context-budget"):
            tree = ast.parse(cells[cell_id])
            if cell_id == "medium-context-budget":
                for node in ast.walk(tree):
                    if isinstance(node, ast.keyword) and node.arg == "char_budget":
                        node.value = ast.copy_location(ast.Constant(budget), node.value)
            exec(compile(tree, f"<{cell_id}>", "exec"), namespace)
    # Combined views for cross-policy assertions; the notebook retains each run separately.
    namespace["context_results"] = [run["totals"] for run in namespace["context_runs"].values()]
    for key in ("calls", "events", "answers"):
        namespace[f"context_{key}"] = [row for run in namespace["context_runs"].values() for row in run[key]]
    namespace["context_states"] = {policy: run["state"] for policy, run in namespace["context_runs"].items()}
    return namespace, requests


@pytest.mark.parametrize("prior_turn,target_turn,prior_ids,returned_ids", [
    (
        5, 6,
        ["E1:S2", "E1:S3", "E2:S1", "E2:S2", "E2:S3", "E3:S1"],
        ["E1:S2", "E1:S3", "E2:S1", "E2:S3", "E3:S1", "E4:S1", "E4:S2"],
    ),
    (
        8, 11,
        ["E1:S2", "E1:S3", "E2:S1", "E2:S2", "E2:S3", "E3:S1", "E4:S1", "E4:S2", "E6:S2"],
        ["E3:S1", "E9:S2", "E1:S2", "E1:S3", "E2:S1", "E2:S3", "E4:S1", "E4:S2", "E6:S2", "E9:S1"],
    ),
])
def test_participant_reported_selections_keep_previously_validated_pending_diagnostic(
    prior_turn, target_turn, prior_ids, returned_ids,
):
    namespace, requests = run_notebook_context(enabled=False)
    state = cm.RollingContext(keep_exchanges=2, required_evidence=namespace["REQUIRED_EVIDENCE"])
    exchanges = namespace["SUPPORT_EXCHANGES"]
    for item in exchanges[:prior_turn]:
        state.append(item)
    assert not state.accept_summary(summary(*prior_ids), stop_reason="end_turn")
    pending = {"exchange_id": "E2", "quote": "I haven't run the hardware diagnostic yet."}
    assert pending in state.summary
    for item in exchanges[prior_turn:target_turn]:
        state.append(item)
    payload = json.loads(state.summary_request())
    assert "E2:S2" in {item["id"] for item in payload["retained_evidence"]}
    assert "E2:S2" not in {item["id"] for item in payload["candidate_sentences"]}
    assert "E2:S2" not in returned_ids  # Exact omission from the participant's responses.
    assert not state.accept_summary(summary(*returned_ids), stop_reason="end_turn")
    assert pending in state.summary
    assert len(state.recent) == 2

    # Without the earlier accepted evidence, the same omission must still fail.
    unvalidated = cm.RollingContext(keep_exchanges=2, required_evidence=namespace["REQUIRED_EVIDENCE"])
    for item in exchanges[:target_turn]:
        unvalidated.append(item)
    before = unvalidated.messages("Handoff?")
    errors = unvalidated.accept_summary(summary(*returned_ids), stop_reason="end_turn")
    assert any("I haven't run the hardware diagnostic yet." in error for error in errors)
    assert unvalidated.messages("Handoff?") == before
    assert not requests


def test_actual_notebook_comparison_preserves_evidence_and_accounts_for_every_call():
    namespace, requests = run_notebook_context()
    results = {r["policy"]: r for r in namespace["context_results"]}
    assert results["full"]["summary_calls"] == 0
    assert results["frequent"]["summary_calls"] == 12
    assert 0 < results["budget"]["summary_calls"] < 12
    assert all(r["checkpoints_passed"] == r["checkpoints"] == 6 for r in results.values())
    assert len(requests) == len(namespace["RUN_ROWS"]) == len(namespace["context_calls"]) <= 42
    for policy, result in results.items():
        calls = [r for r in namespace["context_calls"] if r["label"].startswith(f"L09-{policy}-")]
        assert result["total_cost_usd"] == pytest.approx(sum(r["cost_usd"] for r in calls))
        assert result["total_cost_usd"] == pytest.approx(
            result["handoff_cost_usd"] + result["summary_cost_usd"],
        )
        assert result["total_input_tokens"] == 100 * len(calls)
        assert result["sdk_retries"] == 0
        assert result["full_task_latency_ms"] >= result["summed_call_latency_ms"]
    for request in requests:
        assert request["modelId"] == namespace["CONTEXT_MODEL"]
        assert request["outputConfig"]["textFormat"]["type"] == "json_schema"
        roles = [m["role"] for m in request["messages"]]
        assert roles == ["user", "assistant"] * (len(roles) // 2) + ["user"]
        assert "required_evidence" not in json.dumps(request).lower()
    assert all(event["accepted"] for event in namespace["context_events"])
    assert any(event["after_chars"] < event["before_chars"] for event in namespace["context_events"])


def test_longer_case_reuses_reduced_context_after_successful_selection():
    # Structural check using a valid selection, not a claim about live token billing.
    namespace, _ = run_notebook_context()

    def input_characters(rows):
        return sum(len(json.dumps({
            "system": row["request"]["system"], "messages": row["request"]["messages"],
        })) for row in rows)

    full = namespace["context_runs"]["full"]
    budget = namespace["context_runs"]["budget"]
    assert len(full["answers"]) == len(budget["answers"]) == 6
    assert input_characters(budget["calls"]) < input_characters(full["calls"])
    first_compaction = next(event["turn"] for event in budget["events"] if event["accepted"])
    assert sum(answer["turn"] >= first_compaction for answer in budget["answers"]) >= 3
    assert budget["steps"][-1]["after_chars"] < full["steps"][-1]["after_chars"]


def test_comparison_displays_separate_costs_and_negative_savings_without_more_calls():
    namespace, requests = run_notebook_context()
    before = len(requests)
    namespace["display"] = Mock()
    exec(notebook_cells()["medium-context-results"], namespace)
    assert len(requests) == before
    costs = namespace["display"].call_args_list[0].args[0].data
    quality = namespace["display"].call_args_list[1].args[0].data
    for index, policy in enumerate(("full", "frequent", "budget")):
        totals = namespace["context_runs"][policy]["totals"]
        assert costs.iloc[index]["Handoff cost"] == totals["handoff_cost_usd"]
        assert costs.iloc[index]["Compaction cost"] == totals["summary_cost_usd"]
        assert costs.iloc[index]["Total"] == totals["total_cost_usd"]
    # The test's fixed per-call usage makes extra calls more expensive. Keep that loss visible.
    assert costs.iloc[1]["Net saving"] < 0
    assert quality.iloc[1]["Compactions accepted / rejected"] == "12 / 0"


@pytest.mark.parametrize("policy", ["full", "frequent", "budget"])
def test_each_policy_cell_runs_only_its_own_experiment(policy):
    namespace, requests = run_notebook_context(enabled=False)
    assert not requests
    exec(notebook_cells()[f"medium-context-{policy}"], namespace)
    assert set(namespace["context_runs"]) == {policy}
    run = namespace["context_runs"][policy]
    assert len(requests) == len(run["calls"])
    assert len(run["answers"]) == 6
    assert all(row["label"].startswith(f"L09-{policy}-") for row in run["calls"])


def test_timeline_snapshots_show_retained_history_without_later_mutation():
    namespace, _ = run_notebook_context()
    full = namespace["context_runs"]["full"]
    frequent = namespace["context_runs"]["frequent"]
    assert all(step["before_chars"] == step["after_chars"] for step in full["steps"])
    assert full["steps"][-1]["recent_ids"] == [f"E{i}" for i in range(1, 15)]
    first_summary = next(step for step in frequent["steps"] if step["summary"])
    assert first_summary["turn"] == 3
    assert first_summary["recent_ids"] == ["E2", "E3"]
    assert {item["exchange_id"] for item in first_summary["summary"]} == {"E1"}
    final_summary = frequent["steps"][-1]["summary"]
    assert any("H-220" in item["quote"] for item in final_summary)
    assert final_summary is not first_summary["summary"]
    assert frequent["steps"][-1]["recent_ids"] == ["E13", "E14"]


def test_local_details_display_shows_actual_messages_without_reinvoking_model():
    namespace, requests = run_notebook_context()
    before = len(requests)
    namespace.update(SHOW_CONTEXT_DETAILS=True, pretty_json=Mock(), print=Mock())
    run = namespace["context_runs"]["budget"]
    namespace["show_context_run"](run)
    assert len(requests) == before
    for row in run["calls"]:
        namespace["pretty_json"].assert_any_call(row["request"]["messages"])


def test_replay_rejects_insufficient_call_limit_before_any_model_call():
    namespace, requests = run_notebook_context(enabled=False)
    config = {**namespace["CONTEXT_EXPERIMENT"], "call_limit": 2}
    with pytest.raises(ValueError, match="up to 6 calls"):
        cm.replay_context("full", **config)
    assert not requests


def test_actual_notebook_rejects_unsupported_override_without_model_fallback():
    from workshop_utils.models import UnsupportedFeatureError

    with pytest.raises(UnsupportedFeatureError, match="structured_output"):
        run_notebook_context(model_id="global.openai.gpt-5.6-sol")


def test_actual_notebook_keeps_fenced_failures_and_their_costs():
    namespace, _ = run_notebook_context(fenced=True)
    for result in namespace["context_results"]:
        assert result["checkpoints_passed"] == 0
        assert result["summaries_accepted"] == 0
        assert result["summaries_rejected"] == result["summary_calls"]
        assert result["total_cost_usd"] > 0
    assert all(len(state.recent) == 14 for state in namespace["context_states"].values())
    assert all(row["text"].startswith("```json") for row in namespace["context_calls"])


@pytest.mark.parametrize("failed", [False, True])
def test_results_show_final_handoffs_and_untruncated_failures_without_more_model_calls(failed):
    namespace, requests = run_notebook_context(fenced=failed)
    before = len(requests)
    saved_answers = copy.deepcopy(namespace["context_answers"])
    saved_events = copy.deepcopy(namespace["context_events"])
    namespace.update(print=Mock(), display=Mock(), pretty_json=Mock(), pd=pytest.importorskip("pandas"))
    for run in namespace["context_runs"].values():
        namespace["show_context_run"](run)
    exec(notebook_cells()["medium-context-results"], namespace)
    assert len(requests) == before
    assert namespace["display"].call_count == 5  # Three timelines, then cost and quality tables.
    assert namespace["context_answers"] == saved_answers
    assert namespace["context_events"] == saved_events
    text_output = [call.args[0] for call in namespace["print"].call_args_list]
    json_output = [call.args[0] for call in namespace["pretty_json"].call_args_list]
    for policy in ("full", "frequent", "budget"):
        final = [a for a in saved_answers if a["policy"] == policy][-1]
        if failed:
            assert final["answer"] in text_output
        else:
            assert json.loads(final["answer"]) in json_output
        first_failure = next((e for e in saved_events if e["policy"] == policy and not e["accepted"]), None)
        if first_failure:
            assert first_failure["summary_text"] in text_output
            assert {"reasons": first_failure["errors"]} in json_output
    if failed:
        assert all(a["answer"] in text_output for a in saved_answers if not a["passed"])
        timeline = namespace["display"].call_args_list[1].args[0].data
        assert any("Summary is not JSON" in value for value in timeline["Validation"])


def test_context_prompts_clarify_sentence_and_approval_contract_without_repeated_schema_dumps():
    namespace, _ = run_notebook_context(enabled=False)
    assert "Select IDs from the supplied candidates" in namespace["SUMMARY_SYSTEM"]
    assert "pending diagnostic and replacement approval are separate facts" in namespace["SUMMARY_SYSTEM"]
    assert "no approval has been granted yet (this does not mean denial)" in namespace["ANSWER_SYSTEM"]
    assert '"unknown" only when no reliable approval-status fact is available' in namespace["ANSWER_SYSTEM"]
    assert "H-220" not in namespace["ANSWER_SYSTEM"]  # No fixture answer in the instructions.
    # Native schema configurations remain visible in preparation, not repeated at every call.
    assert namespace["context_request_preview"]["outputConfig"] == namespace["ANSWER_OUTPUT_CONFIG"]
    assert namespace["summary_request_preview"]["outputConfig"] == namespace["SUMMARY_OUTPUT_CONFIG"]
    assert "RUN_CONTEXT" not in "\n".join(notebook_cells().values())
    assert "SUPPORT_EXCHANGES =" not in notebook_cells()["medium-context-fixture"]


@pytest.mark.parametrize("adapter", ["medium", "high"])
def test_retries_keep_returned_cost_but_do_not_claim_complete_request_spend(adapter):
    namespace, _ = run_notebook_context(retries=2, adapter=adapter)
    for row in namespace["context_calls"]:
        assert row["sdk_retries"] == 2
        assert not row["usage_complete"]
        assert row["cost_usd"] is None
        assert row["returned_response_cost_usd"] > 0
        assert "earlier attempts is unknown" in row["price_note"]
    for result in namespace["context_results"]:
        assert result["total_cost_usd"] is None
        assert not result["usage_complete"]


def test_preparation_makes_no_calls_and_high_threshold_can_avoid_summaries():
    disabled, requests = run_notebook_context(enabled=False)
    assert not requests and disabled["context_results"] == []
    namespace, _ = run_notebook_context(budget=1_000_000)
    budget = next(r for r in namespace["context_results"] if r["policy"] == "budget")
    assert budget["summary_calls"] == 0 and budget["checkpoints_passed"] == 6
    assert budget["summary_cost_usd"] == 0


def test_actual_notebook_rejections_still_cost_and_keep_original_history():
    namespace, _ = run_notebook_context(rejected=True, unpriced=True)
    for result in namespace["context_results"]:
        assert result["checkpoints_passed"] == 6
        assert result["total_cost_usd"] is None
        if result["policy"] != "full":
            assert result["summaries_rejected"] == result["summary_calls"] > 0
            if result["policy"] == "frequent":
                assert result["summary_calls"] == 12
            assert result["summaries_accepted"] == 0
            assert result["summary_cost_usd"] is None
    assert all(len(s.recent) == 14 for s in namespace["context_states"].values())


def test_memory_scope_and_corrected_answer_use_retrieved_facts_without_writes():
    client = SimpleNamespace(retrieve_memory_records=Mock(return_value={"memoryRecordSummaries": []}))
    call = Mock(return_value={"text": "offline"})
    context = ["Order A-12345; prefers email."]
    namespace = {
        "os": SimpleNamespace(environ={}),
        "MEMORY_READY": True, "CORRECTION_READY": True, "pretty_json": lambda value: None,
        "corrected_context": ["Order A-12346; prefers email."],
        "MEMORY_ID": "existing", "facts": context, "memory_context": context,
        "MEMORY_NAMESPACE_TEMPLATE": "/users/{actorId}/facts/",
        "AGENTCORE": client, "uuid": uuid, "json": json, "SMALL": "offline",
        "run_case": call, "show": Mock(),
    }
    for cell_id in ("medium-memory-correction-answer", "medium-memory-unknown", "medium-memory-other-customer"):
        exec(notebook_cells()[cell_id], namespace)
    assert client.retrieve_memory_records.call_args.kwargs["namespace"].startswith("/users/playbook-")
    assert call.call_count == 2
    correction, unrelated = call.call_args_list
    assert "A-12346" not in correction.args[1] and "passport" in unrelated.args[1]
    assert "A-12346" in correction.kwargs["system"][0]["text"]
    assert "A-12345" in unrelated.kwargs["system"][0]["text"]
    assert all("messages" not in c.kwargs for c in call.call_args_list)
