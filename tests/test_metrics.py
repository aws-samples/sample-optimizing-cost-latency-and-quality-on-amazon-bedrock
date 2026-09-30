from __future__ import annotations

import copy
import sys
from unittest.mock import Mock

import pytest

from workshop_utils.metrics import metrics_dataframe, metrics_records, normalize_usage
from workshop_utils.observability import CONVERSE_SCOPE

STRANDS = "strands.telemetry.tracer"
BOTOCORE = "opentelemetry.instrumentation.botocore.bedrock-runtime"
TRACE = "a" * 32


def span(number, *, scope=STRANDS, operation="chat", parent=None, trace_id=TRACE, usage=True):
    document = {
        "traceId": trace_id,
        "spanId": f"{number:016x}",
        "name": f"{operation} model",
        "scope": {"name": scope},
        "resource": {"attributes": {"service.name": "fixture-agent"}},
        "startTimeUnixNano": 1_000_000_000 + number,
        "endTimeUnixNano": 2_000_000_000 + number,
        "attributes": {
            "session.id": "session-fixture",
            "gen_ai.operation.name": operation,
            "gen_ai.request.model": "model-fixture",
        },
    }
    if parent is not None:
        document["parentSpanId"] = f"{parent:016x}"
    if usage:
        document["attributes"].update(
            {
                "gen_ai.usage.input_tokens": 130 if scope == STRANDS else 10,
                "gen_ai.usage.output_tokens": 8,
                "gen_ai.usage.cache_read.input_tokens": 100,
                "gen_ai.usage.cache_creation.input_tokens": 20,
                "gen_ai.server.request.duration": 400,
            }
        )
    return document


def test_actual_ancestry_dedup_preserves_independent_calls_and_trace_ids():
    root = span(1, operation="invoke_agent")
    chat = span(2, parent=1)
    # Includes an intervening non-model span, as instrumentation can add one.
    intermediate = span(3, operation="execute_event_loop_cycle", parent=2, usage=False)
    botocore = span(4, scope=BOTOCORE, parent=3)
    independent = span(5, scope=BOTOCORE, parent=1)
    another_trace = span(2, trace_id="b" * 32)
    documents = [root, chat, intermediate, botocore, independent, another_trace, copy.deepcopy(chat)]
    records = metrics_records(documents)
    assert {(row["trace_id"], row["span_id"]) for row in records} == {
        (TRACE, f"{2:016x}"),
        (TRACE, f"{5:016x}"),
        ("b" * 32, f"{2:016x}"),
    }
    assert sum(row["input_tokens"] for row in records) == 30
    assert sum(row["cache_read_tokens"] for row in records) == 300
    assert all(row["cost_usd"] is None for row in records)


def test_wrapper_suppresses_child_botocore_but_not_other_calls():
    wrapper = span(1, scope=CONVERSE_SCOPE, operation="invoke_agent")
    child = span(2, scope=BOTOCORE, parent=1)
    independent = span(3, scope=BOTOCORE)
    records = metrics_records([wrapper, child, independent])
    assert len(records) == 2
    assert records[0]["input_tokens"] == 10


def test_sdk_log_records_never_duplicate_usage():
    chat = span(1)
    log_record = {
        "traceId": TRACE,
        "spanId": chat["spanId"],
        "scope": {"name": STRANDS},
        "body": {"input": {"messages": []}, "output": {"messages": []}},
    }
    assert len(metrics_records([chat, log_record])) == 1


def test_pricing_receives_uncached_usage_and_is_not_coupled_to_module():
    pricing = Mock(return_value=0.125)
    (row,) = metrics_records([span(1)], pricing=pricing)
    pricing.assert_called_once_with(
        model_id="model-fixture",
        input_tokens=10,
        output_tokens=8,
        cache_read_tokens=100,
        cache_write_tokens=20,
    )
    assert row["cost_usd"] == 0.125
    assert row["llm_ms"] == 400


def test_unknown_prices_stay_unknown_not_zero():
    pricing = Mock(side_effect=KeyError("unpriced model"))
    assert metrics_records([span(1)], pricing=pricing)[0]["cost_usd"] is None
    pricing.side_effect = None
    pricing.return_value = None
    assert metrics_records([span(1)], pricing=pricing)[0]["cost_usd"] is None
    pricing.return_value = 0
    assert metrics_records([span(1)], pricing=pricing)[0]["cost_usd"] == 0


def test_missing_usage_does_not_get_priced():
    pricing = Mock()
    (row,) = metrics_records([span(1, usage=False)], pricing=pricing)
    assert row["input_tokens"] is None
    assert row["cache_read_tokens"] is None
    assert row["output_tokens"] is None
    assert row["llm_ms"] is None
    assert row["cost_usd"] is None
    pricing.assert_not_called()


def test_botocore_missing_cache_instrumentation_is_not_free_cache():
    document = span(1, scope=BOTOCORE)
    document["attributes"].pop("gen_ai.usage.cache_read.input_tokens")
    document["attributes"].pop("gen_ai.usage.cache_creation.input_tokens")
    pricing = Mock()
    (record,) = metrics_records([document], pricing=pricing)
    assert record["input_tokens"] == 10
    assert record["cache_read_tokens"] is None
    assert record["cache_write_tokens"] is None
    assert record["cost_usd"] is None
    pricing.assert_not_called()


def test_streaming_ttft_is_preserved_nonstreaming_is_unknown():
    nonstreaming, streaming = span(1), span(2)
    streaming["attributes"]["gen_ai.server.time_to_first_token"] = 123.5
    records = metrics_records([nonstreaming, streaming])
    assert records[0]["ttft_ms"] is None
    assert records[1]["ttft_ms"] == 123.5
    assert records[1]["llm_ms"] == 400


def test_legacy_cache_aliases_and_strands_inclusive_semantics():
    usage = normalize_usage(
        {
            "gen_ai.usage.input_tokens": 130,
            "gen_ai.usage.output_tokens": 8,
            "gen_ai.usage.cache_read_input_tokens": 100,
            "gen_ai.usage.cache_write_input_tokens": 20,
        },
        STRANDS,
    )
    assert usage == {"input_tokens": 10, "output_tokens": 8, "cache_read_tokens": 100, "cache_write_tokens": 20}


@pytest.mark.parametrize(
    "attributes, scope, message",
    [
        ({"gen_ai.usage.input_tokens": -1}, STRANDS, "nonnegative"),
        ({"gen_ai.usage.input_tokens": 1.2}, STRANDS, "integer"),
        ({"gen_ai.usage.input_tokens": 1, "gen_ai.usage.cache_read.input_tokens": 100}, STRANDS, "inclusive"),
        ({"gen_ai.usage.input_tokens": 1}, "unknown.framework", "Unsupported"),
        (
            {"gen_ai.usage.cache_read.input_tokens": 1, "gen_ai.usage.cache_read_input_tokens": 2},
            STRANDS,
            "Conflicting",
        ),
    ],
)
def test_unsupported_or_inconsistent_usage_is_explicit(attributes, scope, message):
    with pytest.raises(ValueError, match=message):
        normalize_usage(attributes, scope)


def test_conflicting_span_copies_fail_instead_of_choosing_a_token_count():
    first, second = span(1), span(1)
    second["attributes"]["gen_ai.usage.input_tokens"] = 500
    with pytest.raises(ValueError, match="Conflicting"):
        metrics_records([first, second])


def test_pandas_optional_and_data_unchanged(monkeypatch):
    document = span(1)
    original = copy.deepcopy(document)
    monkeypatch.setitem(sys.modules, "pandas", None)
    assert metrics_dataframe([document]) == metrics_records([document])
    assert document == original


def test_dataframe_has_stable_empty_columns_and_unknowns():
    pd = pytest.importorskip("pandas")
    frame = metrics_dataframe([span(1)])
    assert isinstance(frame, pd.DataFrame)
    assert pd.isna(frame.iloc[0]["ttft_ms"])
    assert pd.isna(frame.iloc[0]["cost_usd"])
    empty = metrics_dataframe([])
    assert list(empty.columns) == list(frame.columns)


def test_unknown_parent_does_not_drop_a_real_call():
    document = span(1, scope=BOTOCORE, parent=99)
    assert len(metrics_records([document])) == 1


def test_ancestor_cycle_is_rejected():
    with pytest.raises(ValueError, match="Cycle"):
        metrics_records([span(1, parent=2), span(2, parent=1)])
