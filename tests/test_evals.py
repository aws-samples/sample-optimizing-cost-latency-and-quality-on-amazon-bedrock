from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from workshop_utils import evals
from workshop_utils.observability import response_span

TRACE = "a" * 32
TRACE2 = "b" * 32
ROOT = "1" * 16
CHAT = "2" * 16
SESSION = "session-fixture"


def span(trace_id=TRACE, span_id=ROOT):
    return response_span(
        "Return policy?",
        "30 days with receipt.",
        session_id=SESSION,
        trace_id=trace_id,
        span_id=span_id,
        start_time_ns=1_000_000_000,
        end_time_ns=2_000_000_000,
    )


def service_result(trace_id=TRACE, **overrides):
    return {
        "evaluatorArn": "arn:aws:bedrock-agentcore:::evaluator/Builtin.Correctness",
        "evaluatorId": "Builtin.Correctness",
        "evaluatorName": "Builtin.Correctness",
        "context": {"spanContext": {"sessionId": SESSION, "traceId": trace_id}},
        "value": 1.0,
        "label": "Correct",
        "explanation": "The answer matches the reference.",
        "tokenUsage": {"inputTokens": 600, "outputTokens": 80, "totalTokens": 680},
        **overrides,
    }


def test_quick_eval_real_sdk_shape_with_stubbed_client():
    boto3 = pytest.importorskip("boto3")
    stubber_module = pytest.importorskip("botocore.stub")
    client = boto3.client(
        "bedrock-agentcore",
        region_name="us-east-1",
        aws_access_key_id="fixture",
        aws_secret_access_key="fixture",
    )
    if "evaluationReferenceInputs" not in client.meta.service_model.operation_model("Evaluate").input_shape.members:
        pytest.skip("Installed botocore predates ground-truth Evaluate; run with workshop refresh SDK pins")
    document = span()
    response = {"evaluationResults": [service_result()]}
    expected = {
        "evaluatorId": "Builtin.Correctness",
        "evaluationInput": {"sessionSpans": [document]},
        "evaluationTarget": {"traceIds": [TRACE]},
        "evaluationReferenceInputs": [
            {
                "context": {"spanContext": {"sessionId": SESSION, "traceId": TRACE}},
                "expectedResponse": {"text": "30 days with receipt."},
            }
        ],
    }
    with stubber_module.Stubber(client) as stubber:
        stubber.add_response("evaluate", response, expected)
        results = evals.quick_eval(
            [document],
            client=client,
            evaluator_id="Builtin.Correctness",
            session_id=SESSION,
            trace_ids=[TRACE],
            ground_truth={TRACE: "30 days with receipt."},
        )
        assert results == response["evaluationResults"]
        stubber.assert_no_pending_responses()


def test_ground_truth_correlated_per_trace_and_inputs_unchanged():
    client = Mock()
    client.evaluate.return_value = {"evaluationResults": [service_result(), service_result(TRACE2)]}
    docs = [span(), span(TRACE2)]
    original = copy.deepcopy(docs)
    evals.quick_eval(
        docs,
        client=client,
        evaluator_id="Builtin.Correctness",
        session_id=SESSION,
        trace_ids=[TRACE, TRACE2],
        ground_truth={TRACE: "Answer one", TRACE2: "Answer two"},
    )
    references = client.evaluate.call_args.kwargs["evaluationReferenceInputs"]
    assert [r["context"]["spanContext"]["traceId"] for r in references] == [TRACE, TRACE2]
    assert [r["expectedResponse"]["text"] for r in references] == ["Answer one", "Answer two"]
    assert docs == original


def test_session_target_uses_omitted_target_and_real_reference_shape():
    document = span()
    document["attributes"].pop("workshop.response_only")
    document["scope"]["name"] = "strands.telemetry.tracer"
    client = Mock()
    client.evaluate.return_value = {
        "evaluationResults": [
            service_result(
                evaluatorId="Builtin.TrajectoryExactOrderMatch",
                evaluatorName="Builtin.TrajectoryExactOrderMatch",
                evaluatorArn="arn:aws:bedrock-agentcore:::evaluator/Builtin.TrajectoryExactOrderMatch",
                context={"spanContext": {"sessionId": SESSION}},
            )
        ]
    }
    evals.quick_eval(
        [document],
        client=client,
        evaluator_id="Builtin.TrajectoryExactOrderMatch",
        session_id=SESSION,
        trace_ids=[TRACE],
        level="session",
        expected_tools=["lookup_order", "create_ticket"],
        assertions=["Order resolved"],
    )
    request = client.evaluate.call_args.kwargs
    assert "evaluationTarget" not in request
    assert request["evaluationReferenceInputs"] == [
        {
            "context": {"spanContext": {"sessionId": SESSION}},
            "expectedTrajectory": {"toolNames": ["lookup_order", "create_ticket"]},
            "assertions": [{"text": "Order resolved"}],
        }
    ]


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"evaluator_id": "Builtin.ToolSelectionAccuracy"}, "unsupported"),
        ({"level": "tool_call"}, "Only trace and session"),
        ({"trace_ids": [TRACE2]}, "outside trace_ids"),
        ({"session_id": "other-session"}, "different session"),
        ({"ground_truth": {TRACE2: "wrong context"}}, "exactly"),
        ({"trace_ids": TRACE}, "nonempty collection"),
    ],
)
def test_invalid_eval_does_not_call_service(overrides, match):
    client = Mock()
    kwargs = {"client": client, "evaluator_id": "Builtin.Correctness", "session_id": SESSION, "trace_ids": [TRACE]}
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        evals.quick_eval([span()], **kwargs)
    client.evaluate.assert_not_called()


def test_response_only_cannot_claim_session_trajectory():
    with pytest.raises(ValueError, match="response-only"):
        evals.quick_eval(
            [span()],
            client=Mock(),
            evaluator_id="Builtin.GoalSuccessRate",
            session_id=SESSION,
            trace_ids=[TRACE],
            level="session",
        )


@pytest.mark.parametrize(
    "result, match",
    [
        ([], "no results"),
        (
            [service_result(errorCode="UnsupportedScope", errorMessage="Unsupported instrumentation")],
            "evaluator error",
        ),
        ([service_result(ignoredReferenceInputFields=["expectedResponse"])], "ignored supplied ground truth"),
    ],
)
def test_failed_or_unsupported_results_never_become_scores(result, match):
    client = Mock()
    client.evaluate.return_value = {"evaluationResults": result}
    with pytest.raises(evals.EvaluationError, match=match) as exc:
        evals.quick_eval(
            [span()],
            client=client,
            evaluator_id="Builtin.Correctness",
            session_id=SESSION,
            trace_ids=[TRACE],
            ground_truth={TRACE: "Expected"},
        )
    assert exc.value.results == result


def test_client_without_evaluate_is_explicitly_unsupported():
    with pytest.raises(TypeError, match="SDK version"):
        evals.quick_eval(
            [span()], client=object(), evaluator_id="Builtin.Correctness", session_id=SESSION, trace_ids=[TRACE]
        )


def test_sdk_missing_ground_truth_fails_before_request():
    client = Mock()
    client.meta.service_model.operation_model.return_value = SimpleNamespace(
        input_shape=SimpleNamespace(members={"evaluationInput": {}, "evaluationTarget": {}})
    )
    with pytest.raises(TypeError, match="cannot send ground truth"):
        evals.quick_eval(
            [span()],
            client=client,
            evaluator_id="Builtin.Correctness",
            session_id=SESSION,
            trace_ids=[TRACE],
            ground_truth={TRACE: "30 days"},
        )
    client.evaluate.assert_not_called()


def test_incomplete_evaluation_results_are_not_a_passing_gate():
    client = Mock()
    client.evaluate.return_value = {"evaluationResults": [service_result()]}
    with pytest.raises(evals.EvaluationError, match="incomplete trace results"):
        evals.quick_eval(
            [span(), span(TRACE2)],
            client=client,
            evaluator_id="Builtin.Correctness",
            session_id=SESSION,
            trace_ids=[TRACE, TRACE2],
        )


class Clock:
    def __init__(self):
        self.now = 0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(evals.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(evals.time, "sleep", clock.sleep)
    return clock


def query_result(documents, status="Complete", **extra):
    return {
        "status": status,
        "results": [[{"field": "@message", "value": json.dumps(doc)}] for doc in documents],
        "statistics": {"recordsMatched": len(documents)},
        **extra,
    }


def logs_client(results):
    client = Mock()
    client.start_query.return_value = {"queryId": "query-fixture"}
    client.get_query_results.side_effect = results
    return client


def collect(client, **overrides):
    kwargs = {
        "client": client,
        "session_id": SESSION,
        "expected_span_ids": {TRACE: [ROOT]},
        "log_groups": ["aws/spans", "/aws/bedrock-agentcore/runtimes/fixture-DEFAULT"],
        "start_time": 1,
        "end_time": 2,
        "timeout_s": 5,
        "poll_interval_s": 1,
    }
    kwargs.update(overrides)
    return evals.collect_runtime_spans(**kwargs)


def test_complete_caller_expected_traces_required_not_just_count(fake_clock):
    root, second = span(), span(TRACE2)
    client = logs_client([query_result([root]), query_result([root, second])])
    docs = collect(client, expected_span_ids={TRACE: [ROOT], TRACE2: [ROOT]})
    assert {doc["traceId"] for doc in docs} == {TRACE, TRACE2}
    assert client.start_query.call_count == 2
    assert fake_clock.now == 1
    request = client.start_query.call_args.kwargs
    assert request["endTime"] == 62
    assert request["logGroupNames"][0] == "aws/spans"
    assert TRACE in request["queryString"] and TRACE2 in request["queryString"]


def test_split_strands_content_must_arrive_before_evaluation(fake_clock):
    root, chat = span(), span(span_id=CHAT)
    chat["scope"]["name"] = "strands.telemetry.tracer"
    chat["attributes"] = {"session.id": SESSION, "gen_ai.operation.name": "chat"}
    chat["parentSpanId"] = ROOT
    log_record = {
        "traceId": TRACE,
        "spanId": CHAT,
        "scope": {"name": "strands.telemetry.tracer"},
        "body": {
            "input": {"messages": [{"role": "user", "content": {"content": "Return policy?"}}]},
            "output": {"messages": [{"role": "assistant", "content": {"message": "30 days"}}]},
        },
    }
    client = logs_client(
        [
            query_result([root, chat]),
            query_result([root, chat, log_record, copy.deepcopy(chat)]),
        ]
    )
    docs = collect(client, expected_span_ids={TRACE: [ROOT, CHAT]})
    assert len(docs) == 3
    assert log_record in docs
    assert client.start_query.call_count == 2


def test_unfinished_span_does_not_meet_manifest(fake_clock):
    unfinished = span()
    unfinished.pop("endTimeUnixNano")
    client = logs_client([query_result([unfinished])] * 5)
    with pytest.raises(evals.IncompleteTracesError) as exc:
        collect(client)
    assert exc.value.missing == {TRACE: [ROOT]}
    assert fake_clock.now == 5


@pytest.mark.parametrize("operation", ["chat", "execute_tool"])
def test_runtime_unified_event_content_satisfies_manifest(fake_clock, operation):
    document = span()
    document["scope"]["name"] = "strands.telemetry.tracer"
    document["attributes"] = {"session.id": SESSION, "gen_ai.operation.name": operation}
    document["events"] = [
        {
            "name": "gen_ai.tool.message" if operation == "execute_tool" else "gen_ai.user.message",
            "attributes": {"content": "{}" if operation == "execute_tool" else "Return policy?"},
        },
        {"name": "gen_ai.choice", "attributes": {"message": '[{"text":"30 days with receipt"}]'}},
    ]
    client = logs_client([query_result([document])])
    assert collect(client) == [document]
    assert client.start_query.call_count == 1


def test_event_metadata_without_output_still_waits(fake_clock):
    document = span()
    document["scope"]["name"] = "strands.telemetry.tracer"
    document["attributes"] = {"session.id": SESSION, "gen_ai.operation.name": "chat"}
    document["events"] = [{"name": "gen_ai.user.message", "attributes": {"content": "Return policy?"}}]
    client = logs_client([query_result([document])] * 5)
    with pytest.raises(evals.IncompleteTracesError):
        collect(client)


def test_partial_results_never_reach_evaluate(fake_clock):
    client = logs_client([query_result([span()])] * 5)
    evaluation = Mock()
    with pytest.raises(evals.IncompleteTracesError) as exc:
        docs = collect(client, expected_span_ids={TRACE: [ROOT, CHAT]})
        evals.quick_eval(
            docs, client=evaluation, evaluator_id="Builtin.Correctness", session_id=SESSION, trace_ids=[TRACE]
        )
    assert exc.value.missing == {TRACE: [CHAT]}
    evaluation.evaluate.assert_not_called()


@pytest.mark.parametrize("status", ["Failed", "Cancelled", "Timeout", "Unknown", None])
def test_query_terminal_failure_is_not_an_empty_success(fake_clock, status):
    client = logs_client([query_result([], status=status)])
    with pytest.raises(evals.SpanCollectionError, match="status"):
        collect(client)
    client.stop_query.assert_called_once_with(queryId="query-fixture")


def test_running_query_is_bounded_and_cancelled(fake_clock):
    client = logs_client([query_result([span()], status="Running")] * 5)
    with pytest.raises(evals.IncompleteTracesError):
        collect(client)
    client.stop_query.assert_called_once()
    assert fake_clock.now == 5


def test_query_transport_error_propagates_and_cancels(fake_clock):
    client = logs_client([RuntimeError("AccessDenied")])
    with pytest.raises(RuntimeError, match="AccessDenied"):
        collect(client)
    client.stop_query.assert_called_once()


def test_truncated_query_rejected_even_when_manifest_found(fake_clock):
    client = logs_client([query_result([span()], statistics={"recordsMatched": 10001})])
    with pytest.raises(evals.SpanCollectionError, match="truncated"):
        collect(client)


def test_malformed_json_rejected(fake_clock):
    client = logs_client(
        [
            {
                "status": "Complete",
                "results": [[{"field": "@message", "value": "{bad json"}]],
            }
        ]
    )
    with pytest.raises(evals.SpanCollectionError, match="malformed"):
        collect(client)


def test_wrong_session_rejected_even_with_right_trace(fake_clock):
    wrong = span()
    wrong["attributes"]["session.id"] = "other"
    client = logs_client([query_result([wrong])])
    with pytest.raises(evals.SpanCollectionError, match="different session"):
        collect(client)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_s": 0},
        {"timeout_s": float("nan")},
        {"poll_interval_s": -1},
        {"expected_span_ids": {}},
        {"expected_span_ids": {TRACE: []}},
        {"log_groups": []},
        {"start_time": 3, "end_time": 2},
    ],
)
def test_invalid_polling_arguments_never_start_queries(kwargs):
    client = Mock()
    with pytest.raises(ValueError):
        collect(client, **kwargs)
    client.start_query.assert_not_called()
