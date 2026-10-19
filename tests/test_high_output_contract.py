"""Focused offline output-contract checks using only tracked HIGH sources."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SECTION = Path(__file__).resolve().parents[1] / "02-optimization-playbook"
spec = importlib.util.spec_from_file_location("high_output_audit", SECTION / "utils/harness_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def execute_cell(cell_id, namespace):
    notebook = json.loads((SECTION / "03-high-effort.ipynb").read_text())
    source = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == cell_id)
    exec(compile(source, f"<{cell_id}>", "exec"), namespace)


def test_harness_and_gateway_share_raw_json_contract():
    contract = audit.RAW_JSON_CONTRACT
    assert all(term in contract.lower() for term in ("raw json", "no prose", "markdown", "code fences"))
    for topic in ("returns", "billing"):
        for variant in ("verbose", "audited"):
            assert contract in audit.instruction_bundle(topic, variant)
    notebook = json.loads((SECTION / "03-high-effort.ipynb").read_text())
    source = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "high-14-f124e12a")
    assignment = next(
        node for node in ast.parse(source).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "GATEWAY_SYSTEM" for target in node.targets)
    )
    namespace = {"RAW_JSON_CONTRACT": contract}
    # Evaluate only the prompt assignment; never run notebook setup or service calls.
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "<gateway-prompt>", "exec"), namespace)
    assert contract in namespace["GATEWAY_SYSTEM"]


@pytest.mark.parametrize("topic", ["returns", "billing"])
def test_audited_bundle_retains_required_rules(topic):
    bundle = audit.instruction_bundle(topic, "audited")
    required = {**audit.GLOBAL_RULES, **audit.SKILLS[topic]}
    assert {"evidence", "untrusted", "approval", "privacy", "format"} <= required.keys()
    assert set(audit.audit_coverage(bundle, topic)) == set(required)
    assert all(audit.audit_coverage(bundle, topic).values())
    for key, text in required.items():
        assert bundle.count(f"[{key}] {text}") == 1
        assert not audit.audit_coverage(bundle.replace(text, ""), topic)[key]
    assert audit.OBSOLETE not in bundle and audit.IRRELEVANT not in bundle


@pytest.mark.parametrize("kind", ["raw", "fenced", "wrong_field"])
def test_diagnostics_distinguish_format_from_field_errors_without_repair(kind):
    expected = audit.AUDIT_CASES[0]["expected"]
    text = json.dumps(expected)
    if kind == "fenced":
        text = "```json\n" + text + "\n```"
    elif kind == "wrong_field":
        text = json.dumps({**expected, "evidence_id": "invented"})
    row = {"text": text, "stop_reason": "end_turn"}
    diagnostics = audit.answer_diagnostics(row, expected)
    assert audit.check_answer(row, expected)["passed"] is (kind == "raw")
    assert row == {"text": text, "stop_reason": "end_turn"}
    if kind == "fenced":
        assert diagnostics["format_error"]
        assert diagnostics["missing_fields"] is None
        assert diagnostics["unexpected_fields"] is None
        assert diagnostics["field_mismatches"] is None
    else:
        assert diagnostics["format_error"] is None
        assert diagnostics["missing_fields"] == diagnostics["unexpected_fields"] == []
        assert diagnostics["field_mismatches"] == (
            {"evidence_id": {"expected": expected["evidence_id"], "actual": "invented"}}
            if kind == "wrong_field" else {}
        )


@pytest.mark.parametrize("approved", [True, 0, "false"])
def test_wrong_approvals_fail_without_coercion_or_repair(approved):
    expected = audit.AUDIT_CASES[0]["expected"]
    text = json.dumps({**expected, "approved": approved})
    row = {"text": text, "stop_reason": "end_turn"}
    diagnostics = audit.answer_diagnostics(row, expected)
    assert diagnostics["format_error"] is None
    assert diagnostics["field_mismatches"] == {"approved": {"expected": False, "actual": approved}}
    checks = audit.check_answer(row, expected)
    assert checks["json_shape"]
    assert not checks["no_approval"] and not checks["passed"]
    assert row == {"text": text, "stop_reason": "end_turn"}


def test_prompt_comparison_escapes_table_strings_without_altering_full_answers(capsys):
    candidate = '<img src=x onerror="alert(1)">'
    answer = "<script>untrusted answer</script>\n" + "Full explanation. " * 100
    rows = [
        {"variant": "baseline", "ticket": "Duplicate charge", "expected": "billing",
         "text": "billing", "correct": True, "stop_reason": "end_turn",
         "cost_usd": 0.01, "latency_ms": 100},
        {"variant": candidate, "ticket": "Duplicate charge", "expected": "billing",
         "text": answer, "correct": False, "stop_reason": "max_tokens",
         "cost_usd": None, "latency_ms": 200, "price_note": "Price unavailable"},
    ]
    original = copy.deepcopy(rows)
    displays = []
    execute_cell("high-apo-comparison-results", {
        "optimization_assessment": rows, "pd": pd, "display": displays.append,
    })

    assert rows == original
    table = displays[0]
    assert list(table.data["Passed"]) == ["1/1", "0/1"]
    assert pd.isna(table.data.iloc[1]["Model cost (USD)"])
    assert pd.isna(table.data.iloc[1]["USD / solved ticket"])
    html = table.to_html()
    assert candidate not in html and "&lt;img" in html
    assert "—" in html
    output = capsys.readouterr().out
    assert answer in output  # No truncation, escaping, parsing, or repair of the recorded answer.
    assert "max_tokens" in output and "Price unavailable" in output




def test_apo_displays_both_prompts_and_compares_metrics_without_a_diff(capsys):
    from html import escape

    text = "<script>service field</script>"
    candidate = "Classify {{ticket}}.\n" + text
    selected_result = {
        "optimizedPromptTemplate": candidate,
        "originalPromptMetrics": {"averageInputTokens": 90, "averageScore": 1.0},
        "optimizedPromptMetrics": {"averageInputTokens": 332, "averageLatencyInSec": 1.2751},
    }
    original = copy.deepcopy(selected_result)
    displays = []
    execute_cell("high-apo-diff", {
        "apo_candidate_template": candidate, "apo_comparison_baseline": "Route {{ticket}}.",
        "apo_selected_result": selected_result, "HTML": str, "escape": escape,
        "pd": pd, "display": displays.append,
    })

    assert selected_result == original
    assert "Route {{ticket}}." in displays[0]
    assert escape(text) in displays[1] and text not in displays[1]
    table = displays[2].data.set_index("Measure")
    assert table.loc["Average input tokens", "Baseline (original)"] == "90.0"
    assert table.loc["Average input tokens", "APO candidate (optimized)"] == "332.0"
    assert table.loc["Mean response time (s)", "Baseline (original)"] == "—"
    assert table.loc["Mean response time (s)", "APO candidate (optimized)"] == "1.275"
    output = capsys.readouterr().out
    assert "Baseline prompt (original)" in output and "APO candidate prompt (optimized)" in output
    assert "--- baseline" not in output and "@@" not in output
