"""A completed call must not become a passing CI quality result by itself."""

from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1] / "04-deep-dive-topics"


def test_ci_pass_rate_uses_quality_and_keeps_evaluated_version(monkeypatch):
    spec = importlib.util.spec_from_file_location("langfuse_text_transport", ROOT / "utils/langfuse_text_transport.py")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    monkeypatch.setitem(__import__("sys").modules, "langfuse_text_transport", transport)
    spec = importlib.util.spec_from_file_location("stable_lifecycle_eval", ROOT / "scripts/evaluate_prompt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    items = [
        SimpleNamespace(input={"query": f"ticket {i}"}, expected_output={"should_contain": ["missing"]}, metadata={})
        for i in range(2)
    ]

    class Dataset:
        def __init__(self):
            self.items = items

        def run_experiment(self, **kwargs):
            for item in self.items:
                output = kwargs["task"](item=item)
                for evaluator in kwargs["evaluators"]:
                    evaluator(input=item.input, output=output, expected_output=item.expected_output)

    client = SimpleNamespace(
        get_prompt=lambda *a, **k: SimpleNamespace(compile=lambda: "unchanged instructions", config={}, version=7),
        get_dataset=lambda **k: Dataset(),
        start_as_current_observation=lambda **k: nullcontext(SimpleNamespace(update=lambda **k: None)),
        flush=lambda: None,
    )
    monkeypatch.setattr(module, "_init_langfuse", lambda: client)
    monkeypatch.setattr(module, "_call_agent", lambda *a: {"text": "unhelpful", "input_tokens": 5, "output_tokens": 3})
    monkeypatch.setattr(module, "_evaluate_with_llm_judge", lambda *a: {"score": 0.0, "reason": "does not answer"})
    monkeypatch.setattr(
        module.boto3,
        "client",
        lambda *a, **k: SimpleNamespace(get_parameter=lambda **k: {"Parameter": {"Value": "0.7"}}),
    )
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    result = module.run_evaluation()
    assert result["total"] == result["successful"] == 2
    assert result["pass_rate"] == 0.0 and result["avg_score"] == 0.0
    assert result["prompt_version"] == 7 and result["prompt_name"] == module.PROMPT_NAME
