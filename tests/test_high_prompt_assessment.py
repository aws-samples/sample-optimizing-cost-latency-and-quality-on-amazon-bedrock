"""Run the HIGH prompt assessment with synthetic responses and no model clients."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pandas as pd
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
    sources = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}

    def run(baseline, candidate, *, templates=False, tickets=None, prompts=None,
            training=(), model_config=None, supported=True):
        attempts = [row for pair in zip(baseline, candidate, strict=True) for row in pair]
        run_case = Mock(side_effect=attempts)
        printed, displays = [], []
        namespace = {
            "HELD_OUT": TICKETS if tickets is None else tickets,
            "TRAINING": training,
            "copy": copy, "pd": pd,
            "supported": Mock(return_value=supported),
            "run_case": run_case,
            "print": lambda *args: printed.append(copy.deepcopy(args)),
            "display": displays.append,
        }
        exec(compile(sources["high-apo-assessment-tools"], "<assessment-tools>", "exec"), namespace)
        run_case.assert_not_called()
        selected_prompts = (
            {"baseline": "Baseline: {{ticket}}", "candidate": "Candidate: {{ticket}}"}
            if prompts is None else prompts
        )
        original_prompts = copy.deepcopy(selected_prompts)
        rows = namespace["assess_prompts"](
            selected_prompts, "synthetic-model",
            templates=templates,
            model_config=(
                {"modelId": "synthetic-model", "inferenceConfig": {"maxTokens": 1024}}
                if model_config is None else model_config
            ),
        )
        namespace["optimization_assessment"] = rows
        original_rows = copy.deepcopy(rows)
        exec(compile(sources["high-apo-comparison-results"], "<comparison-results>", "exec"), namespace)
        assert rows == original_rows  # Rendering does not repair answers or promote a prompt.
        assert selected_prompts == original_prompts
        return SimpleNamespace(
            rows=rows, run_case=run_case, printed=printed,
            output="\n".join(" ".join(str(value) for value in args) for args in printed),
            summaries=displays[-1].data.set_index("Prompt").to_dict("index"),
            table=displays[-1],
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

    header = f"\nTicket: {TICKETS[0][0]}\nExpected: {TICKETS[0][1]}"
    start = result.printed.index((header,))
    assert result.printed[start:start + 5] == [
        (header,), ("baseline | PASS | end_turn",), ("billing",),
        (f"candidate | FAIL | {stop_reason}",), (raw,),
    ]
    failed = next(row for row in result.rows if not row["correct"])
    assert failed["ticket"] == TICKETS[0][0]
    assert failed["expected"] == TICKETS[0][1]
    assert all(failed[key] == value for key, value in original.items())
    assert result.summaries["candidate"]["Passed"] == "5/6"
    assert result.summaries["candidate"]["Pass rate"] == pytest.approx(5 / 6)
    assert result.summaries["candidate"]["Complete"] == (6 if stop_reason == "end_turn" else 5)
    assert result.run_case.call_count == len(result.rows) == 12


def test_baseline_mismatches_are_visible_too_and_whitespace_rule_is_unchanged(assess):
    candidate = responses()
    candidate[0]["text"] = " \nbilling\t "
    result = assess(responses(correct=0), candidate)

    failures = [row for row in result.rows if not row["correct"]]
    assert len(failures) == 6
    assert {item["variant"] for item in failures} == {"baseline"}
    assert [(item["ticket"], item["expected"]) for item in failures] == TICKETS
    for row in failures:
        header = f"\nTicket: {row['ticket']}\nExpected: {row['expected']}"
        start = result.printed.index((header,))
        assert result.printed[start + 1:start + 3] == [
            ("baseline | FAIL | end_turn",), (row["text"],),
        ]
    assert result.summaries["candidate"]["Passed"] == "6/6"
    assert (" \nbilling\t ",) in result.printed
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

    assert result.summaries["baseline"]["Passed"] == f"{baseline_correct}/6"
    assert result.summaries["candidate"]["Passed"] == f"{candidate_correct}/6"
    assert result.summaries["candidate"]["Pass rate"] == candidate_correct / 6
    assert result.summaries["candidate"]["Model cost (USD)"] == pytest.approx(6 * candidate_cost)
    assert ("retain the baseline" in result.output) is (not eligible)
    assert ("Inspect any newly failed tickets" in result.output) is eligible
    assert "No candidate is auto-adopted" in result.output


def test_cost_and_latency_include_every_attempt_including_failed_answers(assess):
    candidate = responses(correct=2)
    for index, row in enumerate(candidate, start=1):
        row.update(cost_usd=index / 100, latency_ms=index * 100)
    result = assess(responses(), candidate)

    summary = result.summaries["candidate"]
    assert summary["Passed"] == "2/6"
    assert summary["Pass rate"] == pytest.approx(2 / 6)
    assert summary["Model cost (USD)"] == pytest.approx(0.21)
    assert summary["USD / solved ticket"] == pytest.approx(0.105)
    assert summary["Mean latency (ms)"] == pytest.approx(350)
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
        assert pd.isna(summary["Model cost (USD)"])
    else:
        assert summary["Model cost (USD)"] == pytest.approx(0.06)
    if unknown_cost or not correct:
        assert pd.isna(summary["USD / solved ticket"])
    else:
        assert summary["USD / solved ticket"] == pytest.approx(0.06 / correct)
    assert result.summaries["baseline"]["Model cost (USD)"] == pytest.approx(0.06)


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


def test_equal_pass_counts_still_show_newly_failed_tickets(assess):
    baseline, candidate = responses(correct=5), responses()
    candidate[0]["text"] = "technical"
    result = assess(baseline, candidate)

    assert result.summaries["baseline"]["Passed"] == result.summaries["candidate"]["Passed"] == "5/6"
    failed = [(row["variant"], row["ticket"]) for row in result.rows if not row["correct"]]
    assert failed == [("candidate", TICKETS[0][0]), ("baseline", TICKETS[-1][0])]
    assert ("candidate | FAIL | end_turn",) in result.printed
    assert ("baseline | FAIL | end_turn",) in result.printed
    assert "Inspect any newly failed tickets" in result.output


@pytest.mark.parametrize("templates", [False, True])
def test_both_variants_use_the_same_tickets_model_and_controls(assess, templates):
    result = assess(responses(), responses(), templates=templates)
    expected_calls = []
    for ticket, _ in TICKETS:
        for variant in ("baseline", "candidate"):
            prompt = f"{variant.capitalize()}: {{{{ticket}}}}"
            query = prompt.replace("{{ticket}}", ticket) if templates else ticket
            controls = {"max_tokens": 1024}
            if templates:
                controls["additional_model_request_fields"] = None
            else:
                controls["system"] = [{"text": prompt}]
            expected_calls.append(call("synthetic-model", query, label="L14-held-out-" + variant, **controls))
    assert result.run_case.call_args_list == expected_calls


@pytest.mark.parametrize("overrides", [
    {"training": [TICKETS[0]]},
    {"supported": False},
    {"templates": True, "model_config": {"modelId": "other", "inferenceConfig": {"maxTokens": 1024}}},
    {"templates": True, "model_config": {"modelId": "synthetic-model", "inferenceConfig": {}}},
    {"templates": True, "model_config": {
        "modelId": "synthetic-model", "inferenceConfig": {"maxTokens": 1024, "unsupported": 1},
    }},
    {"templates": True, "model_config": {
        "modelId": "synthetic-model", "inferenceConfig": {"maxTokens": 1024}, "unsupported": True,
    }},
])
def test_training_leaks_or_unreproducible_settings_are_rejected_before_calls(assess, overrides):
    with pytest.raises(ValueError):
        assess([], [], **overrides)
