"""The harness uses real notebook control flow with scripted model responses."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
import time
from html import escape
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pandas as pd
import pytest

from workshop_utils.bedrock import build_converse_request
from workshop_utils.models import ModelRegistry

ROOT = Path(__file__).resolve().parents[1]
SECTION = ROOT / "02-optimization-playbook"
NOTEBOOK = SECTION / "03-high-effort.ipynb"


def load_modules():
    package = ModuleType("utils")
    package.__path__ = [str(SECTION / "utils")]
    modules = {"utils": package}
    with patch.dict(sys.modules, modules):
        for name in ("harness_audit", "harness_experiment"):
            key = "utils." + name
            spec = importlib.util.spec_from_file_location(key, SECTION / "utils" / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            modules[key] = module
            sys.modules[key] = module
            spec.loader.exec_module(module)
    return modules


MODULES = load_modules()
FIXTURE = MODULES["utils.harness_experiment"]
AUDIT = MODULES["utils.harness_audit"]


def execute_cell(cell_id, namespace):
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    source = "".join(next(cell["source"] for cell in cells if cell["id"] == cell_id))
    with patch.dict(sys.modules, MODULES):
        exec(compile(source, cell_id, "exec"), namespace)


def experiment(mode="normal"):
    calls, displayed = [], []

    def scripted_model(model, *, label, **controls):
        calls.append(copy.deepcopy({"model": model, "label": label, **controls}))
        case = next(case for case in AUDIT.AUDIT_CASES
                    if controls["messages"][0]["content"][0]["text"] == FIXTURE.task_message(case))
        if mode == "request_error":
            raise RuntimeError("Scripted transport failure")
        first = len(controls["messages"]) == 1
        content = [{"text": json.dumps(case["expected"])}]
        stop = "end_turn"
        if (first and mode != "guess") or mode in {"repeat", "repeat_id"}:
            name = FIXTURE.CASE_TOOLS[case["topic"]] if mode != "unknown_tool" else "issue_refund"
            reference = case["evidence"]["evidence_id"] if mode != "wrong_case" else "other-customer"
            tool_id = "same-id" if mode == "repeat_id" else f"tool-{len(calls)}"
            content = [{"reasoningContent": {"reasoningText": {"text": "Fixture reasoning.", "signature": "fixture"}}},
                       {"toolUse": {"toolUseId": tool_id, "name": name, "input": {"case_reference": reference}}}]
            stop = "tool_use"
        elif mode == "wrong_answer":
            content = [{"text": json.dumps({**case["expected"], "approved": True})}]
        row = {
            "text": "".join(block.get("text", "") for block in content),
            "stop_reason": stop, "response": {"output": {"message": {"role": "assistant", "content": content}}},
            "input_tokens": 100, "output_tokens": 20, "latency_ms": 10, "cost_usd": 0.01,
            "sdk_retries": 0,
        }
        if mode == "unknown_cost":
            row["cost_usd"] = None
        if mode == "sdk_retry":
            row["sdk_retries"] = 1
        return row

    ns = {
        "WORKHORSE": "fixed-model", "REPO_ROOT": ROOT, "SECTION_ROOT": SECTION,
        "pd": pd, "json": json, "time": time, "HTML": str, "escape": escape,
        "display": displayed.append, "pretty_json": lambda value: None,
        "supported": lambda *args: True, "run_case": scripted_model,
    }
    for cell_id in ("high-harness-audit-fixture", "high-harness-skills", "high-harness-after", "high-harness-runner"):
        execute_cell(cell_id, ns)
    return ns, calls, displayed


def test_complete_harness_changes_skills_tools_and_context_but_keeps_rules_and_evidence():
    ns, calls, _ = experiment()
    execute_cell("high-harness-first-run", ns)
    execute_cell("high-harness-audit-live", ns)
    assert len(ns["harness_runs"]) == 10 and len(calls) == 20
    assert all(task["passed"] for task in ns["harness_runs"])
    for task in ns["harness_runs"]:
        assert len(task["calls"]) == 2 and len(task["executions"]) == 1
        assert task["checks"]["case_lookup_succeeded"]
    for call in calls:
        assert call["model"] == "fixed-model"
        assert call["system"][0] == {"text": FIXTURE.SHARED_INSTRUCTIONS}
        assert call["effort"] == "medium" and call["max_tokens"] == 512
        assert "purchase_age_days" not in call["messages"][0]["content"][0]["text"]
        if "-overloaded-" in call["label"]:
            assert len(call["system"]) == 4 and len(call["tool_config"]["tools"]) == 6
        else:
            assert len(call["system"]) == 2 and len(call["tool_config"]["tools"]) == 1
        if len(call["messages"]) > 1:
            assert call["messages"][1]["content"][0]["reasoningContent"]["reasoningText"]["signature"] == "fixture"
    for case in AUDIT.AUDIT_CASES:
        before, after = [task for variant in ns["HARNESS_VARIANTS"] for task in ns["harness_runs"]
                         if task["case_id"] == case["id"] and task["variant"] == variant]
        full = before["executions"][0]
        selected = after["executions"][0]
        assert full["backend_result"] == selected["backend_result"]
        assert full["sent_result"] == full["backend_result"]
        assert selected["sent_result"] == case["evidence"]
        assert "product_profile" not in selected["sent_result"]
    # Completing the remaining-case cell again must not spend on duplicate tasks.
    execute_cell("high-harness-audit-live", ns)
    assert len(calls) == 20


def test_actual_requests_fit_the_native_converse_builder():
    ns, calls, _ = experiment()
    execute_cell("high-harness-first-run", ns)
    model = ModelRegistry(region="us-east-1").resolve("workhorse")
    for call in calls:
        request = build_converse_request(
            model, call["messages"], system=call["system"], tool_config=call["tool_config"],
            effort=call["effort"], max_tokens=call["max_tokens"],
        )
        assert request["toolConfig"]["tools"]


@pytest.mark.parametrize("mode", ["guess", "unknown_tool", "wrong_case", "repeat", "repeat_id", "wrong_answer"])
def test_invalid_or_ungrounded_workflows_never_pass_and_are_bounded(mode):
    ns, calls, _ = experiment(mode)
    execute_cell("high-harness-first-run", ns)
    assert not any(task["passed"] for task in ns["harness_runs"])
    assert len(calls) <= 6
    for task in ns["harness_runs"]:
        assert sum(event["attempted"] for event in task["executions"]) <= 2
        if mode == "guess":
            assert task["checks"]["correct_fields"]
            assert not task["checks"]["case_lookup_succeeded"]
        if mode in {"unknown_tool", "wrong_case"}:
            assert not any(event["attempted"] for event in task["executions"])


@pytest.mark.parametrize("mode", ["normal", "unknown_cost", "sdk_retry", "request_error"])
def test_summary_counts_every_turn_and_preserves_unpriced_results(mode, capsys):
    ns, _, displayed = experiment(mode)
    execute_cell("high-harness-first-run", ns)
    execute_cell("high-harness-audit-live", ns)
    execute_cell("high-harness-answers", ns)
    assert len(displayed[-1].data) == 10
    execute_cell("high-harness-audit-results", ns)
    for totals in ns["harness_totals"].values():
        if mode == "normal":
            assert totals["passed"] == 5
            assert totals["model_calls"] == 10 and totals["tool_calls"] == 5
            assert totals["model_cost_usd"] == pytest.approx(0.1)
            assert totals["cost_per_solved"] == pytest.approx(0.02)
            assert totals["input_tokens"] == 1000
        else:
            assert totals["model_cost_usd"] is None
            assert totals["cost_per_solved"] is None
    if mode == "normal":
        assert "Both harnesses passed every case" in capsys.readouterr().out


def test_dummy_lookup_is_read_only_and_requires_exposed_tool_and_current_reference():
    case = AUDIT.AUDIT_CASES[0]
    saved = copy.deepcopy(case)
    proposal = {"name": "get_return_case", "input": {"case_reference": case["evidence"]["evidence_id"]}}
    result = FIXTURE.execute_dummy_tool(proposal, ["get_return_case"], case)
    result["case"]["receipt"] = False
    assert case == saved
    with pytest.raises(ValueError, match="not exposed"):
        FIXTURE.execute_dummy_tool(proposal, [], case)
    with pytest.raises(ValueError, match="current case"):
        FIXTURE.execute_dummy_tool({**proposal, "input": {"case_reference": "someone-else"}}, ["get_return_case"], case)


def test_partial_comparison_does_not_report_savings(capsys):
    ns, _, displayed = experiment()
    execute_cell("high-harness-first-run", ns)
    count = len(displayed)
    execute_cell("high-harness-audit-results", ns)
    assert len(displayed) == count
    assert "Complete all five cases" in capsys.readouterr().out


def test_skill_files_and_backend_output_do_not_embed_expected_answers():
    ns, _, _ = experiment()
    for name, path in ns["SKILL_PATHS"].items():
        text = ns["SKILL_TEXT"][name]
        assert text.startswith("---\n")
        assert f"name: {path.parent.name}" in text
    for case in AUDIT.AUDIT_CASES:
        tool = FIXTURE.CASE_TOOLS[case["topic"]]
        response = FIXTURE.execute_dummy_tool(
            {"name": tool, "input": {"case_reference": case["evidence"]["evidence_id"]}}, [tool], case,
        )
        assert response["case"] == case["evidence"]
        assert not {"expected", "decision", "next_step", "approved"} & response["case"].keys()
    # All new cells remain ordinary Python, including source navigation.
    book = json.loads(NOTEBOOK.read_text())
    for item in book["cells"]:
        if item["cell_type"] == "code":
            compile("".join(item["source"]), item["id"], "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
