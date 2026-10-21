from __future__ import annotations

import ast
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "03-developer-journey/utils/ragas_eval.py").read_text())
READ = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "read_trace_for_eval")
TEXT = next(n for n in READ.body if isinstance(n, ast.FunctionDef) and n.name == "_to_text")
NS = {"json": json}
exec(compile(ast.Module(body=[TEXT], type_ignores=[]), "trace_text", "exec"), NS)


@pytest.mark.parametrize(
    "payload",
    [
        "final answer",
        {"message": "final answer"},
        {"response": "final answer"},
        {"role": "assistant", "content": [{"reasoningContent": {}}, {"text": "final answer"}]},
        {"message": json.dumps([{"text": "final answer"}])},
    ],
)
def test_actual_answer_shapes_preserve_text(payload):
    assert NS["_to_text"](payload) == "final answer"


def test_tool_preamble_is_not_the_final_answer():
    blocks = [{"text": "Let me check."}, {"toolUse": {"name": "get_product_info", "input": {}}}]
    assert NS["_to_text"]({"message": json.dumps(blocks)}) == ""
    assert NS["_to_text"](json.dumps({"toolUse": {"name": "get_product_info"}})) == ""


def test_latest_answer_selected_and_plain_tool_evidence_kept(monkeypatch):
    import utils.langfuse_metrics as metrics

    observations = [
        SimpleNamespace(
            type="GENERATION",
            name="chat",
            start_time=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
            output={"message": json.dumps([{"text": "Let me check."}, {"toolUse": {}}])},
        ),
        SimpleNamespace(
            type="GENERATION",
            name="chat",
            start_time=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
            output={"content": [{"text": "The verified answer."}]},
        ),
        SimpleNamespace(
            type="TOOL",
            name="get_technical_support",
            start_time=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
            output="Original retrieved paragraph.",
        ),
    ]
    monkeypatch.setattr(
        metrics,
        "trace_reader",
        lambda: SimpleNamespace(
            trace=SimpleNamespace(get=lambda *a, **k: SimpleNamespace(output=None, observations=observations))
        ),
    )
    namespace = {"_dt": dt, "normalize_trace_tool_names": lambda names: names}
    exec(compile(ast.Module(body=[READ], type_ignores=[]), "trace_reader", "exec"), namespace)
    result = namespace["read_trace_for_eval"]("trace-test")
    assert result == {
        "response": "The verified answer.",
        "tool_calls": ["get_technical_support"],
        "retrieved_contexts": ["Original retrieved paragraph."],
    }


def test_gateway_success_envelope_unwrapped_to_original_evidence():
    payload = {
        "message": json.dumps([{"text": json.dumps({"statusCode": 200, "body": "Original paragraph.\nNext line."})}])
    }
    assert NS["_to_text"](payload) == "Original paragraph.\nNext line."
