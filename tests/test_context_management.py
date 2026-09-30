"""Offline regression checks for the context exercise; no AWS clients."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

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


def summary(*entries):
    return json.dumps({"evidence": [{"exchange_id": i, "quote": q} for i, q in entries]})


FIRST = [
    ("E1", "I cannot install software."),
    ("E1", "Do not factory-reset the headset."),
]


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


@pytest.mark.parametrize("text,stop", [
    ("not json", "end_turn"),
    ('{"evidence":[]}', "end_turn"),
    ('{"evidence":"bad"}', "end_turn"),
    ('{"evidence":[], "invented_field":true}', "end_turn"),
    ('{"evidence":[null]}', "end_turn"),
    (summary(("E99", "Invented.")), "end_turn"),
    (summary(("E1", "I can install software.")), "end_turn"),
    (summary(("E1", "install software.")), "end_turn"),
    (summary(*FIRST, FIRST[0]), "end_turn"),
    (summary(*FIRST), "max_tokens"),
    (summary(*FIRST), "guardrail_intervened"),
])
def test_invalid_or_incomplete_summary_keeps_role_pairs(state, text, stop):
    before = state.messages("Next?")
    assert state.accept_summary(text, stop_reason=stop)
    assert state.messages("Next?") == before


def test_repeated_summary_carries_old_evidence_and_only_newly_evicted_pairs(state):
    assert not state.accept_summary(summary(*FIRST), stop_reason="end_turn")
    state.append(exchange(3, "The diagnostic is pending."))
    request = json.loads(state.summary_request())
    assert request["previous_evidence"] == state.summary
    assert [e["exchange_id"] for e in request["newly_evicted_exchanges"]] == ["E2"]
    # A correction does not justify forgetting an unrelated constraint.
    assert state.accept_summary(
        summary(("E2", "Correction: my headset is H-220, not H-200.")),
        stop_reason="end_turn",
    )
    assert not state.accept_summary(
        summary(*FIRST, ("E2", "Correction: my headset is H-220, not H-200.")),
        stop_reason="end_turn",
    )
    messages = state.messages("What next?")
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert "H-220" in messages[0]["content"][0]["text"]
    assert len(state.archive) == 3  # Original evidence remains available for inspection.


def test_size_trigger_counts_system_summary_and_question_as_characters(state):
    size = state.context_chars("Next?", "System")
    assert state.should_compact("budget", "Next?", "System", size)
    assert not state.should_compact("budget", "Next?", "System", size + 1)
    assert state.context_chars("Next?" * 100, "System") > size
    assert state.context_chars("Next?", "System" * 100) > size
    assert not state.accept_summary(summary(*FIRST), stop_reason="end_turn")
    assert "earlier_evidence" in state.messages("Next?")[0]["content"][0]["text"]


def test_summary_keeps_source_chronology_for_corrections(state):
    state.append(exchange(3, "Still waiting."))
    correction = ("E2", "Correction: my headset is H-220, not H-200.")
    assert not state.accept_summary(summary(correction, *FIRST), stop_reason="end_turn")
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
    assert result["total_input_tokens"] == 420
    assert result["output_tokens"] == 60 and result["sdk_retries"] == 0
    assert result["full_task_latency_ms"] == 190
    rows[0]["cost_usd"] = None
    assert cm.call_totals(rows, full_task_latency_ms=190)["total_cost_usd"] is None
    assert cm.call_totals(rows, full_task_latency_ms=190)["summary_cost_usd"] is None


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
    ]:
        assert not all(cm.answer_checks(json.dumps({**correct, key: value}), "end_turn").values())
    assert not all(cm.answer_checks(json.dumps(correct), "max_tokens").values())
    assert not all(cm.answer_checks("invalid", "end_turn").values())


def notebook_cells():
    notebook = json.loads((ROOT / "02-optimization-playbook/02-medium-effort.ipynb").read_text())
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


def run_notebook_context(*, enabled=True, budget=4200, rejected=False, unpriced=False,
                         retries=0, adapter="medium"):
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
        "os": SimpleNamespace(environ={"RUN_CONTEXT_COMPARISON": "1" if enabled else "0"}),
        "json": json, "time": time, "uuid": uuid, "datetime": datetime, "UTC": UTC,
        "hashlib": hashlib, "boto3": SimpleNamespace(__version__="offline"),
        "Exchange": cm.Exchange, "RollingContext": cm.RollingContext,
        "answer_checks": cm.answer_checks, "call_totals": cm.call_totals,
        "SMALL": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "REGION": "us-east-1", "RUNTIME": object(), "RUN_ID": "offline", "RUN_ROWS": [],
        "resolve_model": resolve_model, "build_converse_request": build_converse_request,
        "normalize_usage": normalize_usage, "calculate_cost": calculate_cost,
        "UnknownPriceError": UnknownPriceError, "InferredPriceError": InferredPriceError,
        "AmbiguousCacheUsageError": AmbiguousCacheUsageError,
        "response_span": lambda *args, **kwargs: None,
        "print": lambda *args: None,
    }

    def wire_call(client, **request):
        assert client is namespace["RUNTIME"]
        requests.append(request)
        system = request["system"][0]["text"]
        if system == namespace["SUMMARY_SYSTEM"]:
            source = json.loads(request["messages"][0]["content"][0]["text"])
            evidence = list(source["previous_evidence"])
            for exchange in source["newly_evicted_exchanges"]:
                eid = exchange["exchange_id"]
                evidence.extend(
                    {"exchange_id": eid, "quote": quote}
                    for quote in namespace["REQUIRED_EVIDENCE"].get(eid, [])
                )
            if rejected:
                evidence = [e for e in evidence if "factory-reset" not in e["quote"]]
            text = json.dumps({"evidence": evidence})
        else:
            # Answer from the ACTUAL supplied context; dropped facts become failures.
            context = json.dumps(request["messages"])
            text = json.dumps({
                "headset_model": "H-220" if "Correction: my headset is H-220" in context else "unknown",
                "constraints": [
                    value for marker, value in [
                        ("I cannot install software.", "no_software_install"),
                        ("Do not factory-reset the headset.", "no_factory_reset"),
                    ] if marker in context
                ],
                "failed_checks": [
                    value for marker, value in [
                        ("A cable swap did not fix", "cable_swap"),
                        ("The left channel also fails on another device.", "other_device"),
                    ] if marker in context
                ],
                "next_action": "hardware_diagnostic" if "hardware diagnostic is still pending" in context else "unknown",
                "diagnostic_status": "pending" if "hardware diagnostic is still pending" in context else "unknown",
                "replacement_status": "not_approved" if "No replacement has been approved." in context else "unknown",
            })
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
    tree = ast.parse(cells["medium-context-fixture"])
    # Bind the same local helper through importlib to avoid other sections' utils packages.
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)]
    exec(compile(tree, "<context-fixture>", "exec"), namespace)
    namespace["CONTEXT_CHAR_BUDGET"] = budget
    exec(compile(cells["medium-context-comparison"], "<context-comparison>", "exec"), namespace)
    return namespace, requests


def test_actual_notebook_comparison_preserves_evidence_and_accounts_for_every_call():
    namespace, requests = run_notebook_context()
    results = {r["policy"]: r for r in namespace["context_results"]}
    assert results["full"]["summary_calls"] == 0
    assert results["frequent"]["summary_calls"] == 6
    assert 0 < results["budget"]["summary_calls"] < 6
    assert all(r["checkpoints_passed"] == r["checkpoints"] == 3 for r in results.values())
    assert len(requests) == len(namespace["RUN_ROWS"]) == len(namespace["context_calls"]) <= 21
    for policy, result in results.items():
        calls = [r for r in namespace["context_calls"] if r["label"].startswith(f"L09-{policy}-")]
        assert result["total_cost_usd"] == pytest.approx(sum(r["cost_usd"] for r in calls))
        assert result["total_input_tokens"] == 100 * len(calls)
        assert result["sdk_retries"] == 0
        assert result["full_task_latency_ms"] >= result["summed_call_latency_ms"]
    for request in requests:
        roles = [m["role"] for m in request["messages"]]
        assert roles == ["user", "assistant"] * (len(roles) // 2) + ["user"]
        assert "required_evidence" not in json.dumps(request).lower()
    assert all(event["accepted"] for event in namespace["context_events"])
    assert any(event["after_chars"] < event["before_chars"] for event in namespace["context_events"])


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


def test_actual_notebook_can_run_without_any_summaries_and_remains_opt_in():
    disabled, requests = run_notebook_context(enabled=False)
    assert not requests and disabled["context_results"] == []
    namespace, _ = run_notebook_context(budget=1_000_000)
    budget = next(r for r in namespace["context_results"] if r["policy"] == "budget")
    assert budget["summary_calls"] == 0 and budget["checkpoints_passed"] == 3
    assert budget["summary_cost_usd"] == 0


def test_actual_notebook_rejections_still_cost_and_keep_original_history():
    namespace, _ = run_notebook_context(rejected=True, unpriced=True)
    for result in namespace["context_results"]:
        assert result["checkpoints_passed"] == 3
        assert result["total_cost_usd"] is None
        if result["policy"] != "full":
            assert result["summaries_rejected"] == result["summary_calls"] > 0
            if result["policy"] == "frequent":
                assert result["summary_calls"] == 6
            assert result["summaries_accepted"] == 0
            assert result["summary_cost_usd"] is None
    assert all(len(s.recent) == 8 for s in namespace["context_states"].values())


def test_memory_scope_and_correction_cell_uses_fresh_questions_without_writes():
    client = SimpleNamespace(retrieve_memory_records=Mock(return_value={"memoryRecordSummaries": []}))
    call = Mock(return_value={"text": "offline"})
    context = ["Order A-12345; prefers email."]
    namespace = {
        "os": SimpleNamespace(environ={"RUN_MEMORY_CHECKS": "1"}),
        "MEMORY_ID": "existing", "facts": context, "memory_context": context,
        "MEMORY_NAMESPACE_TEMPLATE": "/users/{actorId}/facts/",
        "AGENTCORE": client, "uuid": uuid, "json": json, "SMALL": "offline",
        "run_case": call, "show": Mock(),
    }
    exec(notebook_cells()["medium-memory-scope-corrections"], namespace)
    assert client.retrieve_memory_records.call_args.kwargs["namespace"].startswith("/users/playbook-")
    assert call.call_count == 2
    correction, unrelated = call.call_args_list
    assert "A-12346" in correction.args[1] and "passport" in unrelated.args[1]
    assert all("A-12345" in c.kwargs["system"][0]["text"] for c in call.call_args_list)
    assert all("messages" not in c.kwargs for c in call.call_args_list)
