from __future__ import annotations

import ast
import copy
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "retry_warranty_quality", ROOT / "02-optimization-playbook/utils/warranty_quality.py"
)
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def result(*, error=False, trace="trace"):
    entry = {
        "evaluatorId": "evaluator",
        "context": {"spanContext": {"traceId": trace, "sessionId": "session"}},
    }
    if error:
        entry.update(errorCode="ValidationException", errorMessage="Evaluator model failed")
    else:
        entry.update(value=1, label="Incomplete", explanation="A policy condition was omitted.")
        entry["tokenUsage"] = {"inputTokens": 10, "outputTokens": 5}
    return {"evaluationResults": [entry], "ResponseMetadata": {"RetryAttempts": 0}}


def row_with(response):
    spans = [{"traceId": "trace", "events": [{"name": "original-evidence"}]}]
    return {
        "text": "Keep the original answer.",
        "trace_id": "trace",
        "session_id": "session",
        "spans": spans,
        "spans_hash": quality.fingerprint(spans),
        "evaluation": response,
        "evaluation_seconds": 2,
        "evaluated_by": "evaluator",
        "evaluation_validated": False,
    }


def notebook_evaluator():
    notebook = json.loads((ROOT / "02-optimization-playbook/03-high-effort.ipynb").read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "def evaluate_warranty(row):" in "".join(cell["source"])
    )
    definition = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_warranty"
    )
    api = Mock(return_value=result())
    pricing = Mock(return_value=SimpleNamespace(total_cost=0.001))
    namespace = {
        "check_warranty_experiment": lambda: None,
        "fingerprint": quality.fingerprint,
        "prepare_evaluation_retry": quality.prepare_evaluation_retry,
        "checked_score": quality.checked_score,
        "WARRANTY_EVALUATOR_ID": "evaluator",
        "WARRANTY_JUDGE_MODEL": "judge",
        "AGENTCORE": SimpleNamespace(evaluate=api),
        "RECOMMENDATION_STATE_PATH": Path("unused-state.json"),
        "warranty_state": {},
        "save_state": Mock(),
        "time": time,
        "normalize_usage": lambda *args, **kwargs: None,
        "calculate_cost": pricing,
        "REGION": "us-east-1",
    }
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "evaluate_warranty", "exec"), namespace)
    return namespace["evaluate_warranty"], api, pricing


def test_service_error_can_be_retried_without_changing_answer_or_evidence():
    original = result(error=True)
    row = row_with(copy.deepcopy(original))
    before = copy.deepcopy(row)
    evaluate, api, pricing = notebook_evaluator()

    assert evaluate(row)["value"] == 1
    api.assert_called_once_with(
        evaluatorId="evaluator",
        evaluationInput={"sessionSpans": before["spans"]},
        evaluationTarget={"traceIds": ["trace"]},
    )
    assert row["text"] == before["text"]
    assert row["spans"] == before["spans"]
    assert row["evaluation_attempts"][0]["response"] == original
    assert row["evaluation_attempts"][0]["seconds"] == 2
    assert row["evaluation_validated"] is True
    assert row["evaluation_cost_usd"] is None
    pricing.assert_not_called()


def test_successful_cached_score_is_not_retried():
    row = row_with(result())
    evaluate, api, _ = notebook_evaluator()
    assert evaluate(row)["value"] == 1
    api.assert_not_called()
    assert row["evaluation_cost_usd"] == 0.001


def test_unrelated_score_is_rejected_instead_of_retrying_for_a_better_result():
    row = row_with(result(trace="unrelated"))
    evaluate, api, _ = notebook_evaluator()
    with pytest.raises(ValueError, match="No usable policy score"):
        evaluate(row)
    api.assert_not_called()
    assert "evaluation_attempts" not in row


def test_third_service_failure_stops_without_another_request():
    row = row_with(result(error=True))
    row["evaluation_attempts"] = [
        {"response": result(error=True), "seconds": 1, "evaluator_id": "evaluator"}
        for _ in range(2)
    ]
    evaluate, api, _ = notebook_evaluator()
    with pytest.raises(ValueError, match="three attempts"):
        evaluate(row)
    api.assert_not_called()
    assert row["evaluation"]["evaluationResults"][0]["errorCode"] == "ValidationException"


def test_service_failure_from_another_evaluator_does_not_trigger_retry():
    row = row_with(result(error=True))
    row["evaluated_by"] = "another-evaluator"
    with pytest.raises(ValueError, match="different evaluator"):
        quality.prepare_evaluation_retry(row, "evaluator")
    assert "evaluation_attempts" not in row
