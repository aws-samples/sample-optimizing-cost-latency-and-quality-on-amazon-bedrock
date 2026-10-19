"""Exercise the authored streaming prompt with an offline event stream."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from workshop_utils.bedrock import build_converse_request
from workshop_utils.models import resolve_model

NOTEBOOK = Path(__file__).parents[1] / "01-fundamentals/01-prompts-101.ipynb"
CELLS = {cell["id"]: "".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text())["cells"]}


def test_streaming_and_optional_comparison_share_reference_and_keep_returned_text():
    selected = resolve_model("global.anthropic.claude-sonnet-5", region="us-east-1")
    streams, requests = [], []

    class Stream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            yield {"contentBlockDelta": {"delta": {"text": ""}}}
            if len(requests) == 1:
                namespace["print"].assert_called_once_with("Waiting for the first text…", flush=True)
            yield {"contentBlockDelta": {"delta": {"text": "Unverified "}}}
            # Each example displays its first chunk before consuming the next.
            namespace["print"].assert_called_with("Unverified ", end="", flush=True)
            yield {"contentBlockDelta": {"delta": {"text": "fixture answer."}}}
            namespace["print"].assert_called_with("fixture answer.", end="", flush=True)
            yield {"messageStop": {"stopReason": "end_turn"}}
            yield {"metadata": {
                "usage": {"inputTokens": 60, "outputTokens": 5},
                "metrics": {"latencyMs": 123.456},
            }}

        def close(self):
            self.closed = True

    def converse_stream(**request):
        requests.append(request)
        if len(requests) > 1:
            # The participant can inspect exactly what is sent before inference.
            printed = namespace["print"].call_args.args[0]
            assert request["messages"][0]["content"][0]["text"] in printed
            assert printed.endswith("\n\nAnswer:")
        stream = Stream()
        streams.append(stream)
        return {"stream": stream}

    namespace = {
        "SELECTED": selected, "MODEL_ID": selected.model_id, "EFFORT": None, "time": time,
        "build_converse_request": build_converse_request, "display": Mock(), "pd": pd,
        "runtime": SimpleNamespace(converse_stream=converse_stream), "print": Mock(),
    }
    exec(compile(CELLS["cell-022"], str(NOTEBOOK), "exec"), namespace)
    assert len(requests) == 1 and streams[0].closed
    assert namespace["latency_sample"]["text"] == "Unverified fixture answer."
    assert namespace["latency_sample"]["usage"]["inputTokens"] == 60
    assert namespace["latency_sample"]["visible_ttft_ms"] is not None
    assert namespace["latency_sample"]["e2e_ms"] >= namespace["latency_sample"]["visible_ttft_ms"]
    namespace["print"].assert_called_with("\n", flush=True)
    # Keep the full answer in the result without duplicating it in the summary.
    assert namespace["display"].call_args.args[0] == {
        key: value for key, value in namespace["latency_sample"].items() if key != "text"
    }
    namespace["print"].reset_mock()
    # Participant opt-in; enable only that flag in the authored comparison cell.
    source = CELLS["cell-024"].replace("RUN_LATENCY_COMPARISON = False", "RUN_LATENCY_COMPARISON = True")
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    assert len(requests) == 3 and all(stream.closed for stream in streams)
    assert [request["inferenceConfig"]["maxTokens"] for request in requests[1:]] == [1024, 1024]
    for request in requests:
        prompt = request["messages"][0]["content"][0]["text"]
        assert namespace["CACHING_REFERENCE"] in prompt
        assert "unchanged prompt prefix" in prompt
        assert "model still generates a new response" in prompt
        assert "does not replay a saved answer" in prompt
        assert request["modelId"] == selected.model_id
    table = namespace["display"].call_args.args[0]
    assert list(table.columns) == [
        "Scenario", "TTFT (ms)", "E2E (ms)", "Provider (ms)", "Input tokens", "Output tokens", "Stop",
    ]
    assert table["Scenario"].tolist() == ["One sentence", "Six numbered paragraphs"]
    for row, sample in zip(table.to_dict("records"), namespace["samples"], strict=True):
        assert row["TTFT (ms)"] == round(sample["visible_ttft_ms"], 1)
        assert row["E2E (ms)"] == round(sample["e2e_ms"], 1)
        assert row["Provider (ms)"] == 123.5
        assert (row["Input tokens"], row["Output tokens"], row["Stop"]) == (60, 5, "end_turn")
        # Presentation must not discard or round the original evidence.
        assert sample["provider_ms"] == 123.456
        assert sample["text"] == "Unverified fixture answer."
    short_prompt = requests[1]["messages"][0]["content"][0]["text"]
    long_prompt = requests[2]["messages"][0]["content"][0]["text"]
    assert "in one sentence." in short_prompt
    assert long_prompt == short_prompt.replace("in one sentence.", "in six numbered paragraphs.")
    assert [call.args[0] for call in namespace["print"].call_args_list] == [
        f"\nOne-sentence prompt:\n{short_prompt}\n\nAnswer:",
        "Unverified ", "fixture answer.", "\n",
        f"\nSix-paragraph prompt:\n{long_prompt}\n\nAnswer:",
        "Unverified ", "fixture answer.", "\n",
    ]


@pytest.mark.parametrize("usage", [None, {}, {"inputTokens": 0, "outputTokens": 0}])
def test_latency_table_preserves_missing_measurements_and_real_zero_tokens(usage):
    sample = {
        "text": "Incomplete fixture.", "stop_reason": "max_tokens", "usage": usage,
        "visible_ttft_ms": None, "e2e_ms": 123.456, "provider_ms": None,
    }
    namespace = {"measure_latency": Mock(return_value=sample),
                 "CACHING_REFERENCE": "Offline reference.", "display": Mock(), "pd": pd, "print": Mock()}
    # Default opt-out must not make any calls.
    exec(compile(CELLS["cell-024"], str(NOTEBOOK), "exec"), namespace)
    namespace["measure_latency"].assert_not_called()
    namespace["display"].assert_not_called()
    source = CELLS["cell-024"].replace("RUN_LATENCY_COMPARISON = False", "RUN_LATENCY_COMPARISON = True")
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["measure_latency"].call_count == 2
    table = namespace["display"].call_args.args[0]
    assert table["TTFT (ms)"].isna().all() and table["Provider (ms)"].isna().all()
    assert table["E2E (ms)"].tolist() == [123.5, 123.5]
    assert table["Stop"].tolist() == ["max_tokens", "max_tokens"]
    for column in ("Input tokens", "Output tokens"):
        assert str(table[column].dtype) == "Int64"
        if usage:
            assert (table[column] == 0).all()
        else:
            assert table[column].isna().all()
    assert sample["usage"] is usage and sample["e2e_ms"] == 123.456
