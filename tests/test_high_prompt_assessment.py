"""Run the HIGH prompt assessment with synthetic responses and no model clients."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "02-optimization-playbook/03-high-effort.ipynb"
TICKETS = [
    ("Duplicate payment", "billing"),
    ("Invoice correction", "billing"),
    ("Bluetooth failure", "technical"),
    ("Display flicker", "technical"),
    ("Open box return", "returns"),
    ("Return address", "returns"),
]


def responses(correct=6, *, cost=0.01):
    return [
        {
            "text": expected if index < correct else f"This is {expected}.",
            "stop_reason": "end_turn",
            "cost_usd": cost,
            "latency_ms": 100,
        }
        for index, (_, expected) in enumerate(TICKETS)
    ]


@pytest.fixture
def assess():
    notebook = json.loads(NOTEBOOK.read_text())
    source = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "high-08-f889fbf9")
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "assess_prompts")
    code = compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec")

    def run(baseline, candidate, *, templates=False, tickets=None, prompts=None):
        attempts = [row for pair in zip(baseline, candidate, strict=True) for row in pair]
        run_case = Mock(side_effect=attempts)
        printed = []
        namespace = {
            "HELD_OUT": TICKETS if tickets is None else tickets,
            "run_case": run_case,
            "print": lambda value: printed.append(copy.deepcopy(value)),
        }
        exec(code, namespace)
        rows = namespace["assess_prompts"](
            {"baseline": "Baseline: {{ticket}}", "candidate": "Candidate: {{ticket}}"}
            if prompts is None else prompts,
            "synthetic-model",
            templates=templates,
            model_config={"modelId": "synthetic-model", "inferenceConfig": {"maxTokens": 1024}},
        )
        return SimpleNamespace(
            rows=rows, run_case=run_case,
            failures=[item for item in printed if isinstance(item, dict) and "actual_raw_answer" in item],
            summaries={item["variant"]: item for item in printed
                       if isinstance(item, dict) and "cost_per_correct_ticket" in item},
            comparisons=[item for item in printed if isinstance(item, dict) and "outcome" in item],
        )

    return run


@pytest.mark.parametrize("templates", [False, True])
@pytest.mark.parametrize(("raw", "stop_reason"), [
    ("  This is billing.\n", "end_turn"),
    ("```text\nbilling\n```", "end_turn"),
    ("Billing", "end_turn"),
    ("technical", "end_turn"),
    ("", "end_turn"),
    ("billing", "max_tokens"),
    ("billing", "guardrail_intervened"),
])
def test_mismatch_displays_original_ticket_answer_and_stop_reason(assess, templates, raw, stop_reason):
    candidate = responses()
    candidate[0].update(
        text=raw, stop_reason=stop_reason,
        response={"output": {"message": {"content": [{"text": raw}]}}},
    )
    original = copy.deepcopy(candidate[0])

    result = assess(responses(), candidate, templates=templates)

    assert result.failures == [{
        "variant": "candidate", "ticket": TICKETS[0][0], "expected": TICKETS[0][1],
        "actual_raw_answer": raw, "stop_reason": stop_reason,
    }]
    failed = next(row for row in result.rows if not row["correct"])
    assert failed["ticket"] == TICKETS[0][0]
    assert failed["expected"] == TICKETS[0][1]
    assert all(failed[key] == value for key, value in original.items())
    assert result.summaries["candidate"]["correct"] == 5
    assert result.summaries["candidate"]["completed"] == (6 if stop_reason == "end_turn" else 5)
    assert result.run_case.call_count == len(result.rows) == 12


def test_baseline_mismatches_are_visible_too_and_whitespace_rule_is_unchanged(assess):
    candidate = responses()
    candidate[0]["text"] = " \nbilling\t "
    result = assess(responses(correct=0), candidate)

    assert len(result.failures) == 6
    assert {item["variant"] for item in result.failures} == {"baseline"}
    assert [(item["ticket"], item["expected"]) for item in result.failures] == TICKETS
    assert result.summaries["candidate"]["correct"] == 6
    assert next(row for row in result.rows if row["variant"] == "candidate")["text"] == " \nbilling\t "


@pytest.mark.parametrize(("baseline_correct", "candidate_correct", "candidate_cost", "eligible"), [
    (6, 2, 0.001, False),  # Lower cost does not excuse quality regression.
    (6, 6, 0.02, True),  # A more expensive tie merits review, not automatic adoption.
    (2, 6, 0.02, True),
    (0, 0, 0.01, True),  # No regression alone does not establish an acceptable quality floor.
])
def test_comparison_requires_no_quality_regression_before_tradeoff_review(
    assess, baseline_correct, candidate_correct, candidate_cost, eligible,
):
    result = assess(responses(baseline_correct), responses(candidate_correct, cost=candidate_cost))

    assert len(result.comparisons) == 1
    comparison = result.comparisons[0]
    assert comparison["candidate"] == "candidate"
    assert comparison["baseline_correct"] == baseline_correct
    assert comparison["candidate_correct"] == candidate_correct
    assert comparison["eligible_for_tradeoff_review"] is eligible
    assert comparison["outcome"] == (
        "No quality regression: eligible for cost/latency tradeoff review only."
        if eligible else "Retain baseline: quality regression."
    )


def test_cost_and_latency_include_every_attempt_including_failed_answers(assess):
    candidate = responses(correct=2)
    for index, row in enumerate(candidate, start=1):
        row.update(cost_usd=index / 100, latency_ms=index * 100)
    result = assess(responses(), candidate)

    summary = result.summaries["candidate"]
    assert summary["n"] == 6
    assert summary["correct"] == 2
    assert summary["model_cost"] == pytest.approx(0.21)
    assert summary["cost_per_correct_ticket"] == pytest.approx(0.105)
    assert summary["mean_ms"] == pytest.approx(350)
    assert result.run_case.call_count == 12


@pytest.mark.parametrize("correct", [0, 2, 6])
@pytest.mark.parametrize("unknown_cost", [False, True])
def test_unknown_cost_and_zero_success_never_turn_into_zero_cost(assess, correct, unknown_cost):
    candidate = responses(correct=correct)
    if unknown_cost:
        candidate[-1].update(cost_usd=None, sdk_retries=1, returned_response_cost_usd=0.01)
    result = assess(responses(), candidate)

    summary = result.summaries["candidate"]
    if unknown_cost:
        assert summary["model_cost"] is None
    else:
        assert summary["model_cost"] == pytest.approx(0.06)
    if unknown_cost or not correct:
        assert summary["cost_per_correct_ticket"] is None
    else:
        assert summary["cost_per_correct_ticket"] == pytest.approx(0.06 / correct)
    assert result.summaries["baseline"]["model_cost"] == pytest.approx(0.06)


@pytest.mark.parametrize("overrides", [
    {"tickets": []},
    {"prompts": {}},
    {"prompts": {"candidate": "Candidate"}},
    {"prompts": {"baseline": "Baseline"}},
])
def test_incomplete_comparison_is_rejected_before_any_model_call(assess, overrides):
    # With no synthetic responses, any model call raises instead of returning data.
    with pytest.raises(ValueError):
        assess([], [], **overrides)
