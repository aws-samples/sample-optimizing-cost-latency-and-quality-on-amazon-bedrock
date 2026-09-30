"""Small AgentCore Evaluate adapter and bounded remote span collection.

Request shapes were checked against botocore 1.43.101 / AgentCore 1.23.1.
All clients are injected. No inference, credentials, or initialization on import.
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import datetime
from typing import Any

_TRACE_EVALUATORS = {
    "Correctness",
    "Faithfulness",
    "Helpfulness",
    "ResponseRelevance",
    "Conciseness",
    "Coherence",
    "InstructionFollowing",
    "Refusal",
    "Harmfulness",
    "Stereotyping",
}
_SESSION_EVALUATORS = {
    "GoalSuccessRate",
    "TrajectoryExactOrderMatch",
    "TrajectoryInOrderMatch",
    "TrajectoryAnyOrderMatch",
}


class EvaluationError(RuntimeError):
    """The service did not produce a usable result; no passing score is inferred."""

    def __init__(self, message: str, results: list[dict] | None = None):
        super().__init__(message)
        self.results = results or []


class SpanCollectionError(RuntimeError):
    """A query failed or could not return trustworthy complete input."""


class IncompleteTracesError(TimeoutError):
    """Deadline reached before all caller-expected finished spans were observed."""

    def __init__(self, missing: dict[str, list[str]]):
        self.missing = missing
        super().__init__(f"Remote span collection timed out; missing expected span IDs: {missing}")


def _ids(values: Sequence[str], size: int, name: str) -> list[str]:
    if isinstance(values, str) or not values:
        raise ValueError(f"{name} must be a nonempty collection of IDs")
    result = []
    for value in values:
        if (
            not isinstance(value, str)
            or len(value) != size
            or any(c not in "0123456789abcdef" for c in value)
            or int(value, 16) == 0
        ):
            raise ValueError(f"{name} must contain nonzero lowercase {size}-character hex IDs")
        if value not in result:
            result.append(value)
    return result


def quick_eval(
    span_docs: Sequence[Mapping],
    *,
    client: Any,
    evaluator_id: str,
    session_id: str,
    trace_ids: Sequence[str],
    ground_truth: Mapping[str, str] | None = None,
    level: str = "trace",
    expected_tools: Sequence[str] | None = None,
    assertions: Sequence[str] | None = None,
) -> list[dict]:
    """Evaluate supplied documents, not an agent invocation or a log query.

    trace_ids and session_id explicitly bound the evaluated input. Trace-level
    ground_truth maps each selected trace to expected response text. Session-level
    evaluation omits evaluationTarget (the API has no sessionIds target); optional
    assertions and expected_tools apply to the supplied session. Only trace and
    session levels are supported here; tool-call evaluators must use the SDK.

    A response_span can check the observed answer immediately. It cannot establish
    remote tool/trajectory quality. For Runtime work first collect_runtime_spans
    with a complete expected-span manifest. Supplied documents are otherwise the
    caller's completeness responsibility. Judge tokens remain separate from
    application usage. Service errors and ignored ground truth fail explicitly.
    """
    traces = _ids(trace_ids, 32, "trace_ids")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be explicit and nonempty")
    if level not in {"trace", "session"}:
        raise ValueError("Only trace and session evaluation levels are supported")
    if not isinstance(evaluator_id, str) or not evaluator_id:
        raise ValueError("evaluator_id is required")
    if evaluator_id.startswith("Builtin."):
        supported = _TRACE_EVALUATORS if level == "trace" else _SESSION_EVALUATORS
        if evaluator_id.removeprefix("Builtin.") not in supported:
            raise ValueError(f"{evaluator_id} is unsupported at level={level}")
    if level == "trace" and len(traces) > 10:
        raise ValueError("Evaluate accepts at most 10 trace targets per call")
    if not callable(getattr(client, "evaluate", None)):
        raise TypeError("Injected client must support bedrock-agentcore.evaluate; check its SDK version")
    if not span_docs or len(span_docs) > 20000:
        raise ValueError("Supply between 1 and 20000 span documents")
    documents = []
    seen_traces = set()
    for document in span_docs:
        if not isinstance(document, Mapping):
            raise ValueError("Each span document must be a mapping")
        if document.get("traceId") not in traces:
            raise ValueError("Input contains a trace outside trace_ids")
        _ids([document.get("spanId")], 16, "spanId")
        attributes = document.get("attributes", {})
        document_session = attributes.get("session.id")
        # SDK conversation log records share IDs but omit session.id.
        if document_session is not None and document_session != session_id:
            raise ValueError("Input contains a different session")
        if document_session == session_id and "startTimeUnixNano" in document:
            seen_traces.add(document["traceId"])
        if level == "session" and attributes.get("workshop.response_only"):
            raise ValueError("A response-only observation cannot evaluate a session trajectory")
        documents.append(copy.deepcopy(dict(document)))
    if seen_traces != set(traces):
        raise ValueError("Every target trace needs a span explicitly attributed to session_id")

    references = []
    if level == "trace":
        if expected_tools is not None or assertions is not None:
            raise ValueError("expected_tools and assertions require session-level evaluation")
        if ground_truth is not None:
            if set(ground_truth) != set(traces):
                raise ValueError("ground_truth must contain exactly the targeted trace IDs")
            for trace_id in traces:
                expected = ground_truth[trace_id]
                if not isinstance(expected, str) or not 1 <= len(expected) <= 100000:
                    raise ValueError("Expected response text must contain 1..100000 characters")
                references.append(
                    {
                        "context": {"spanContext": {"sessionId": session_id, "traceId": trace_id}},
                        "expectedResponse": {"text": expected},
                    }
                )
    else:
        if ground_truth is not None:
            raise ValueError("ground_truth expected responses require trace-level evaluation")
        reference = {"context": {"spanContext": {"sessionId": session_id}}}
        if expected_tools is not None:
            if (
                isinstance(expected_tools, str)
                or len(expected_tools) > 1000
                or any(not isinstance(item, str) or not 1 <= len(item) <= 500 for item in expected_tools)
            ):
                raise ValueError("expected_tools must contain at most 1000 nonempty tool names")
            reference["expectedTrajectory"] = {"toolNames": list(expected_tools)}
        if assertions is not None:
            if (
                isinstance(assertions, str)
                or not 1 <= len(assertions) <= 100
                or any(not isinstance(item, str) or not 1 <= len(item) <= 100000 for item in assertions)
            ):
                raise ValueError("assertions must contain 1..100 nonempty text assertions")
            reference["assertions"] = [{"text": item} for item in assertions]
        if len(reference) > 1:
            references.append(reference)
    request = {"evaluatorId": evaluator_id, "evaluationInput": {"sessionSpans": documents}}
    if level == "trace":
        request["evaluationTarget"] = {"traceIds": traces}
    if references:
        request["evaluationReferenceInputs"] = references
        model = getattr(getattr(client, "meta", None), "service_model", None)
        if model is not None:
            members = model.operation_model("Evaluate").input_shape.members
            if isinstance(members, Mapping) and "evaluationReferenceInputs" not in members:
                raise TypeError(
                    "This SDK cannot send ground truth (evaluationReferenceInputs); "
                    "install the workshop's current boto3/botocore pins"
                )
    response = client.evaluate(**request)
    results = response.get("evaluationResults")
    if not isinstance(results, list) or not results:
        raise EvaluationError("Evaluate returned no results")
    if any(result.get("errorCode") or result.get("errorMessage") for result in results):
        raise EvaluationError("Evaluate returned an evaluator error; inspect exception.results", results)
    if any(result.get("ignoredReferenceInputFields") for result in results):
        raise EvaluationError("Evaluator ignored supplied ground truth; inspect exception.results", results)
    scored_traces = set()
    for result in results:
        context = result.get("context", {}).get("spanContext", {})
        if result.get("evaluatorId") != evaluator_id or context.get("sessionId") != session_id:
            raise EvaluationError("Evaluate returned a result for an unexpected evaluator/session", results)
        if level == "trace":
            if context.get("traceId") not in traces:
                raise EvaluationError("Evaluate returned an unexpected trace target", results)
            scored_traces.add(context["traceId"])
        if result.get("value") is None and result.get("label") is None:
            raise EvaluationError("Evaluate returned a result without a score or label", results)
    if level == "trace" and scored_traces != set(traces):
        raise EvaluationError("Evaluate returned incomplete trace results", results)
    return results


def _epoch(value: datetime | float) -> float:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("Datetime query boundaries must include a timezone")
        value = value.timestamp()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Query boundaries must be nonnegative epoch seconds or aware datetimes")
    return value


def _finished(document: Mapping) -> bool:
    start, end = document.get("startTimeUnixNano"), document.get("endTimeUnixNano")
    return isinstance(start, int) and isinstance(end, int) and 0 <= start <= end


def _has_content(document: Mapping, content_ids: set[tuple[str, str]]) -> bool:
    """Strands model/tool spans need payloads, not only span metadata."""
    attributes = document.get("attributes", {})
    if document.get("scope", {}).get("name") != "strands.telemetry.tracer":
        return True
    if attributes.get("gen_ai.operation.name") not in {"chat", "execute_tool"}:
        return True
    if (document["traceId"], document["spanId"]) in content_ids:
        return True
    # ADOT's unified content mode keeps Strands message payloads on span events.
    # Live Runtime spans (Strands 1.57 / ADOT 0.20) use gen_ai.*.message for
    # inputs and gen_ai.choice for the completed model/tool result.
    events = document.get("events") or []
    input_names = (
        {"gen_ai.tool.message"}
        if attributes.get("gen_ai.operation.name") == "execute_tool"
        else {"gen_ai.system.message", "gen_ai.user.message", "gen_ai.assistant.message", "gen_ai.tool.message"}
    )
    has_event_input = any(
        event.get("name") in input_names and event.get("attributes", {}).get("content") is not None
        for event in events
    )
    has_event_output = any(
        event.get("name") == "gen_ai.choice" and event.get("attributes", {}).get("message") is not None
        for event in events
    )
    if has_event_input and has_event_output:
        return True
    return any(
        input_key in attributes and output_key in attributes
        for input_key, output_key in (
            ("gen_ai.input.messages", "gen_ai.output.messages"),
            ("gen_ai.task.input", "gen_ai.task.output"),
            ("gen_ai.tool.call.arguments", "gen_ai.tool.call.result"),
        )
    )


def collect_runtime_spans(
    *,
    client: Any,
    session_id: str,
    expected_span_ids: Mapping[str, Sequence[str]],
    log_groups: Sequence[str],
    start_time: datetime | float,
    end_time: datetime | float,
    timeout_s: float = 120,
    poll_interval_s: float = 2,
) -> list[dict]:
    """Poll Logs Insights until every caller-expected span has finished.

    The manifest maps trace IDs to ALL span IDs required for the evaluation, not
    just root IDs or a count. Completion means completeness against that manifest,
    not discovery of an unknowable remote trajectory. Include both ``aws/spans``
    and the Runtime log group for split telemetry. Query boundaries cover the
    invocation; an ingestion buffer of 60 seconds is added to end_time.

    AgentCore 1.23.1's CloudWatchAgentSpanCollector returns on first nonempty data,
    which may be partial. This helper reruns completed queries until the manifest
    is met. It preserves SDK conversation log documents sharing trace/span IDs.
    For split telemetry, a span also needs its input/output log record before it
    can count as ready (or inline gen_ai input/output content in unified mode).

    timeout_s bounds polling and sleeps, not an in-flight synchronous SDK request.
    Inject a Logs client with bounded connect/read timeouts and retries (e.g.
    botocore Config(connect_timeout=2, read_timeout=5, retries={"max_attempts": 0})).
    Timeout raises with missing IDs; it never returns partial spans for evaluation.
    """
    if not isinstance(session_id, str) or not session_id or any(c in session_id for c in "\r\n"):
        raise ValueError("session_id must be nonempty and contain no newlines")
    if not expected_span_ids or not isinstance(expected_span_ids, Mapping):
        raise ValueError("expected_span_ids must map every expected trace to required span IDs")
    manifest = {
        _ids([trace_id], 32, "trace IDs")[0]: set(_ids(span_ids, 16, "span IDs"))
        for trace_id, span_ids in expected_span_ids.items()
    }
    if (
        isinstance(log_groups, str)
        or not 1 <= len(log_groups) <= 50
        or any(not isinstance(group, str) or not group for group in log_groups)
    ):
        raise ValueError("Supply 1..50 explicit log group names")
    for name, value in (("timeout_s", timeout_s), ("poll_interval_s", poll_interval_s)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    start, end = _epoch(start_time), _epoch(end_time)
    if end < start:
        raise ValueError("end_time must not precede start_time")
    for method in ("start_query", "get_query_results", "stop_query"):
        if not callable(getattr(client, method, None)):
            raise TypeError(f"Injected Logs client must support {method}")
    # Trace filtering also retains split-telemetry log records with no session.id.
    # Selection is verified against session-attributed spans below.
    query = (
        "fields @message\n| filter traceId in "
        + json.dumps(list(manifest))
        + "\n| filter ispresent(scope.name) and ispresent(spanId)\n| sort @timestamp asc"
    )
    deadline = time.monotonic() + timeout_s
    missing = {trace_id: sorted(ids) for trace_id, ids in manifest.items()}
    while time.monotonic() < deadline:
        query_id = client.start_query(
            logGroupNames=list(dict.fromkeys(log_groups)),
            startTime=math.floor(start),
            endTime=math.ceil(end) + 60,
            queryString=query,
            limit=10000,
        )["queryId"]
        completed = False
        try:
            while time.monotonic() < deadline:
                result = client.get_query_results(queryId=query_id)
                status = result.get("status")
                if status == "Complete":
                    completed = True
                    break
                if status not in {"Scheduled", "Running"}:
                    raise SpanCollectionError(f"Logs Insights query {query_id} ended with status {status!r}")
                time.sleep(min(poll_interval_s, max(0, deadline - time.monotonic())))
        finally:
            if not completed:
                # Preserve the original failure/timeout; cancellation is best effort.
                with suppress(Exception):
                    client.stop_query(queryId=query_id)
        if not completed or time.monotonic() >= deadline:
            break
        rows = result.get("results", [])
        if len(rows) >= 10000 or result.get("statistics", {}).get("recordsMatched", 0) > len(rows):
            raise SpanCollectionError("Logs Insights result is truncated; narrow the invocation time window")
        documents = {}
        for row in rows:
            messages = [field.get("value") for field in row if field.get("field") == "@message"]
            if len(messages) != 1:
                raise SpanCollectionError("Logs Insights row lacks a unique @message")
            try:
                document = json.loads(messages[0])
            except (TypeError, json.JSONDecodeError) as exc:
                raise SpanCollectionError("Logs Insights returned malformed span JSON") from exc
            if not isinstance(document, dict) or document.get("traceId") not in manifest:
                raise SpanCollectionError("Logs Insights returned an unexpected document")
            sid = document.get("attributes", {}).get("session.id")
            if sid is not None and sid != session_id:
                raise SpanCollectionError("A requested trace belongs to a different session")
            _ids([document.get("spanId")], 16, "spanId")
            documents[json.dumps(document, sort_keys=True)] = document
        docs = list(documents.values())
        content_ids = {
            (document["traceId"], document["spanId"])
            for document in docs
            if isinstance(document.get("body"), dict) and "input" in document["body"] and "output" in document["body"]
        }
        present = {trace_id: set() for trace_id in manifest}
        for document in docs:
            if (
                _finished(document)
                and document.get("attributes", {}).get("session.id") == session_id
                and _has_content(document, content_ids)
            ):
                present[document["traceId"]].add(document["spanId"])
        missing = {
            trace_id: sorted(ids - present[trace_id]) for trace_id, ids in manifest.items() if ids - present[trace_id]
        }
        if not missing:
            return docs
        time.sleep(min(poll_interval_s, max(0, deadline - time.monotonic())))
    raise IncompleteTracesError(missing)
