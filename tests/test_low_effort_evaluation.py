"""Exercise notebook evaluation logic without credentials or inference."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from workshop_utils.bedrock import build_converse_request, normalize_usage
from workshop_utils.models import Support, caps, get_model, resolve_model
from workshop_utils.pricing import (
    AmbiguousCacheUsageError,
    InferredPriceError,
    UnknownPriceError,
    calculate_cost,
)

NOTEBOOK = Path(__file__).resolve().parents[1] / "02-optimization-playbook/01-low-effort.ipynb"
_scorer_path = NOTEBOOK.parent / "utils/effort_scoring.py"
_scorer_spec = importlib.util.spec_from_file_location("low_effort_scoring", _scorer_path)
_scorer = importlib.util.module_from_spec(_scorer_spec)
_scorer_spec.loader.exec_module(_scorer)
score_effort_answer = _scorer.score_effort_answer

SONNET = "global.anthropic.claude-sonnet-5"
HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
LUNA = "global.openai.gpt-5.6-luna"


def code_cells():
    return {
        cell["id"]: "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
        if cell["cell_type"] == "code"
    }


def run_comparison(*, include_gpt=False, workhorse=SONNET, truncated_effort=None, formatting=False):
    cells = code_cells()
    functions = [
        node for node in ast.parse(cells["low-setup-9d7ad99d"]).body
        if isinstance(node, ast.FunctionDef) and node.name in {"supported", "run_case"}
    ]
    sent, printed = [], []
    namespace = {
        "WORKHORSE": workhorse, "GPT": LUNA, "RUN_GPT": include_gpt,
        "REGION": "us-east-1", "RUNTIME": object(), "RUN_ID": "offline-evaluation",
        "RUN_ROWS": [], "boto3": SimpleNamespace(__version__="offline"),
        "caps": caps, "resolve_model": resolve_model, "get_model": get_model, "Support": Support,
        "pd": pd, "display": lambda value: None, "SHOW_REQUEST_DETAILS": False,
        "score_effort_answer": score_effort_answer,
        "build_converse_request": build_converse_request, "normalize_usage": normalize_usage,
        "calculate_cost": calculate_cost, "UnknownPriceError": UnknownPriceError,
        "InferredPriceError": InferredPriceError, "AmbiguousCacheUsageError": AmbiguousCacheUsageError,
        "json": json, "hashlib": hashlib, "time": time, "uuid": uuid,
        "datetime": datetime, "UTC": UTC, "response_span": lambda *args, **kwargs: None,
        "print": lambda *args: printed.extend(args),
    }

    def response_fixture(**request):
        sent.append(copy.deepcopy(request))
        question = request["messages"][0]["content"][0]["text"]
        task = next(task for task in namespace["EFFORT_TASKS"] if task["question"] == question)
        fields = request["additionalModelRequestFields"]
        effort = (fields.get("output_config") or fields.get("reasoning"))["effort"]
        output = {"low": 10, "medium": 20, "high": 40}[effort]
        answer = task["expected"]
        if formatting and effort != "low" and task["id"] == "missing-evidence":
            answer = "**ask_for_receipt**\n\nThe receipt is the missing policy requirement."
        if formatting and effort == "high" and task["id"] == "arithmetic" and len(sent) % 2 == 0:
            answer = "**6195**"
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": answer}]}},
            "usage": {"inputTokens": 100, "outputTokens": output, "totalTokens": 100 + output},
            "stopReason": "max_tokens" if effort == truncated_effort else "end_turn",
        }

    @contextmanager
    def observation_fixture(request, *, session_id):
        assert session_id
        yield lambda response: response

    namespace["RUNTIME"] = SimpleNamespace(converse=response_fixture)
    namespace["converse_observation"] = observation_fixture
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    exec(compile(cells["low-effort-options"], str(NOTEBOOK), "exec"), namespace)
    exec(compile(cells["low-reasoning-effort-sweep"], str(NOTEBOOK), "exec"), namespace)
    return namespace, sent, printed


def summarize(namespace, **requirements):
    namespace = {**namespace, **requirements}
    shown, printed = [], []
    namespace.update(
        pd=pd,
        display=shown.append,
        print=lambda *args: printed.extend(args),
    )
    # The presentation dependency is not needed to test the actual calculations.
    tree = ast.parse(code_cells()["low-effort-results"])
    tree.body = [
        node for node in tree.body
        if not (isinstance(node, ast.Import) and any(alias.name == "pandas" for alias in node.names))
    ]
    exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
    return namespace, shown, printed


@pytest.mark.parametrize("include_gpt,expected_requests", [(False, 24), (True, 48)])
def test_repeated_comparison_holds_requests_fixed_except_effort(include_gpt, expected_requests):
    namespace, sent, _ = run_comparison(include_gpt=include_gpt)
    assert len(sent) == expected_requests
    assert len(namespace["effort_rows"]) == expected_requests
    assert all(row["correct"] for row in namespace["effort_rows"])
    assert all(row["effective_effort"] == row["effort"] for row in namespace["effort_rows"])
    for model_id in namespace["EFFORT_MODELS"]:
        by_question = {}
        for request in sent:
            if request["modelId"] != model_id:
                continue
            assert request["inferenceConfig"] == {"maxTokens": 2048}
            normalized = copy.deepcopy(request)
            fields = normalized["additionalModelRequestFields"]
            (fields.get("output_config") or fields.get("reasoning"))["effort"] = "fixed"
            question = request["messages"][0]["content"][0]["text"]
            by_question.setdefault(question, []).append(normalized)
        assert len(by_question) == 4
        assert all(len(group) == 6 and all(r == group[0] for r in group) for group in by_question.values())
    # Independently check the authored arithmetic and discount/shipping answers.
    expected = {task["id"]: task["expected"] for task in namespace["EFFORT_TASKS"]}
    assert expected["arithmetic"] == str(17 * 23 * 19 - 1234)
    assert expected["invoice-total"] == str(int(120 * 0.9 + 5))


def test_lowest_cost_passing_setting_wins_not_highest_effort():
    namespace, _, _ = run_comparison()
    result, _, _printed = summarize(namespace)
    rows = result["effort_summary"]
    assert len(rows) == 3
    assert all(row["passed"] == row["attempts"] == 8 for row in rows)
    assert all(row["pass_rate"] == row["completion_rate"] == 1 for row in rows)
    choice = result["choice"]
    assert choice["effort"] == "low"
    assert rows[0]["mean_cost_per_attempt_usd"] == pytest.approx(rows[0]["total_cost_usd"] / 8)


def test_truncated_exact_answers_fail_quality_and_completion_gates():
    namespace, _, _ = run_comparison(truncated_effort="low")
    result, _, _printed = summarize(namespace)
    low = result["effort_summary"][0]
    assert low["pass_rate"] == low["completion_rate"] == 0
    assert low["cost_per_success_usd"] is None
    assert not low["meets_requirements"]
    assert low["mean_cost_per_attempt_usd"] > 0
    assert result["choice"]["effort"] == "medium"


def test_failed_attempts_are_included_in_cost_per_success():
    namespace, _, _ = run_comparison()
    low = [row for row in namespace["effort_rows"] if row["effort"] == "low"]
    low[0]["correct"] = False
    low[0]["answer_correct"] = False
    low[0]["text"] = "incorrect"
    result, _, _ = summarize(namespace, MIN_PASS_RATE=0.8)
    aggregate = result["effort_summary"][0]
    assert aggregate["passed"] == 7
    assert aggregate["cost_per_success_usd"] == pytest.approx(sum(r["cost_usd"] for r in low) / 7)
    assert aggregate["cost_per_success_usd"] > aggregate["mean_cost_per_attempt_usd"]


def test_unknown_prices_stay_unknown_and_cannot_win():
    namespace, _, _ = run_comparison()
    namespace["effort_rows"][0]["cost_usd"] = None
    result, _, _printed = summarize(namespace)
    low = result["effort_summary"][0]
    assert low["total_cost_usd"] is None
    assert low["mean_cost_per_attempt_usd"] is None
    assert low["cost_per_success_usd"] is None
    assert result["choice"]["effort"] == "medium"


def test_latency_requirement_can_reject_every_configuration():
    namespace, _, _ = run_comparison()
    for row in namespace["effort_rows"]:
        row["latency_ms"] = 100
    result, _, printed = summarize(namespace, MAX_MEAN_LATENCY_MS=50)
    assert not result["priced_candidates"]
    assert not any(row["meets_requirements"] for row in result["effort_summary"])
    assert isinstance(printed[-1], str)


def test_model_without_effort_produces_no_fabricated_measurements():
    namespace, sent, _ = run_comparison(workhorse=HAIKU)
    result, shown, printed = summarize(namespace)
    assert not sent and not result["effort_summary"]
    assert shown == []
    assert isinstance(printed[-1], str)


def test_service_tier_exercise_is_removed():
    cells = code_cells()
    assert "low-22-7857aa47" not in cells
    assert not any("RUN_SERVICE_TIERS" in source for source in cells.values())


def test_chart_uses_measured_values_and_skips_unknown_cost(monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    namespace, _, _ = run_comparison()
    namespace["effort_rows"][0]["cost_usd"] = None
    result, _, _ = summarize(namespace)
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        exec(code_cells()["low-effort-plots"], result)
        axis = result["ax"]
        points = axis.collections[0]
        assert len(points.get_offsets()) == 2
        assert "per task attempt" in axis.get_xlabel()
        assert axis.get_ylabel() == "Task success (%)"
        assert list(points.get_offsets()[:, 1]) == [100, 100]
        assert list(points.get_array()) == [r["mean_latency_ms"] for r in result["plot_rows"]]
        assert len(result["fig"].axes) == 2  # One comparison and one latency colour bar.
        assert result["fig"].axes[1].get_ylabel() == "Mean task latency (ms)"
    finally:
        plt.close("all")


def test_format_variation_does_not_masquerade_as_wrong_reasoning():
    namespace, sent, _ = run_comparison(formatting=True)
    result, _, _ = summarize(namespace)
    assert len(sent) == 24
    assert [r["answer_matches"] for r in result["effort_summary"]] == [8, 8, 8]
    assert [r["format_matches"] for r in result["effort_summary"]] == [8, 6, 5]
    assert [r["passed"] for r in result["effort_summary"]] == [8, 8, 8]
    assert any(row["text"] == "**6195**" for row in namespace["effort_rows"])
    strict, _, _ = summarize(namespace, REQUIRE_EXACT_FORMAT=True)
    assert [r["passed"] for r in strict["effort_summary"]] == [8, 6, 5]
    assert len(sent) == 24  # Rescoring the retained results made no new calls.
    assert [r["total_cost_usd"] for r in strict["effort_summary"]] == [r["total_cost_usd"] for r in result["effort_summary"]]
