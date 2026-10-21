"""Execute LOW's authored cells against local clients; never execute setup."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import time
import uuid
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from workshop_utils.bedrock import build_converse_request, normalize_usage
from workshop_utils.models import Support, UnsupportedFeatureError, caps, get_model, resolve_model
from workshop_utils.observability import response_span
from workshop_utils.pricing import (
    AmbiguousCacheUsageError,
    InferredPriceError,
    UnknownPriceError,
    calculate_cost,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "02-optimization-playbook/01-low-effort.ipynb"
TRACE = "a" * 32
SPAN = "b" * 16


def cells():
    return {
        cell["id"]: "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
        if cell["cell_type"] == "code"
    }


def execute(cell_id, namespace):
    exec(compile(cells()[cell_id], str(NOTEBOOK), "exec"), namespace)


@pytest.fixture
def lab():
    calls, printed, recorded, displayed = [], [], [], []
    active = []
    raw_usage = {"inputTokens": 100, "outputTokens": 10}
    retry_attempts = [0]

    @contextmanager
    def observation(request, *, session_id):
        active.append((request, session_id))

        def record(response):
            recorded.append(response)
            return {
                **response,
                "_trace": {"session_id": session_id, "trace_id": TRACE, "span_id": SPAN},
            }

        try:
            yield record
        finally:
            active.pop()

    def converse(**request):
        assert active and active[-1][0] == request
        calls.append(copy.deepcopy(request))
        content = [{"text": "billing"}]
        if "toolConfig" in request:
            content = [{"toolUse": {
                "toolUseId": "tool-1", "name": "classify_email",
                "input": {"issue": "damaged", "sentiment": "negative", "action": "replace"},
            }}]
        return {
            "output": {"message": {"role": "assistant", "content": content}},
            "usage": copy.deepcopy(raw_usage),
            "stopReason": "tool_use" if "toolConfig" in request else "end_turn",
            "serviceTier": {"type": "default"},
            "ResponseMetadata": {"RetryAttempts": retry_attempts[0]},
        }

    def capture_print(*args, **kwargs):
        for value in args:
            if isinstance(value, str):
                with suppress(ValueError):
                    value = json.loads(value)
            printed.append(value)

    ns = {
        "REPO_ROOT": ROOT, "REGION": "us-east-1", "RUN_ID": "native-offline", "RUN_ROWS": [],
        "RUNTIME": SimpleNamespace(converse=converse),
        "boto3": SimpleNamespace(__version__="offline"),
        "converse_observation": observation, "response_span": response_span,
        "build_converse_request": build_converse_request, "normalize_usage": normalize_usage,
        "caps": caps, "resolve_model": resolve_model, "get_model": get_model, "Support": Support,
        "pd": pd, "display": displayed.append, "SHOW_REQUEST_DETAILS": True,
        "RUN_PREFIX_CHANGE": False, "RUN_SELF_REFINE": False,
        "calculate_cost": calculate_cost, "UnknownPriceError": UnknownPriceError,
        "InferredPriceError": InferredPriceError, "AmbiguousCacheUsageError": AmbiguousCacheUsageError,
        "SMALL": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "WORKHORSE": "global.anthropic.claude-sonnet-5",
        "GPT": "global.openai.gpt-5.6-luna", "RUN_GPT": False,
        "hashlib": hashlib, "json": json, "time": time, "uuid": uuid,
        "datetime": datetime, "UTC": UTC,
        "print": capture_print,
    }
    functions = [
        node for node in ast.parse(cells()["low-setup-9d7ad99d"]).body
        if isinstance(node, ast.FunctionDef)
    ]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), ns)
    return SimpleNamespace(ns=ns, calls=calls, printed=printed, recorded=recorded,
                           usage=raw_usage, retry_attempts=retry_attempts, displayed=displayed)


def test_native_converse_records_the_same_single_response(lab):
    row = lab.ns["run_case"](lab.ns["WORKHORSE"], "Classify.", label="native",
                            additional_model_request_fields={"output_config": {"effort": "low"}})
    assert len(lab.calls) == len(lab.recorded) == len(lab.ns["RUN_ROWS"]) == 1
    assert row["request"] == lab.calls[0]
    assert row["response"]["usage"] == lab.recorded[0]["usage"]
    assert row["effective_effort"] == "low"
    assert row["span"]["traceId"] == TRACE and row["span"]["spanId"] == SPAN
    assert row["span"]["attributes"]["session.id"] == row["response"]["_trace"]["session_id"]
    assert row["cost_usd"] > 0 and row["sdk_retries"] == 0 and row["ttft_ms"] is None
    assert row["usage_complete"] and row["returned_response_cost_usd"] == row["cost_usd"]
    assert row["request_sha256"] == hashlib.sha256(
        json.dumps(row["request"], sort_keys=True).encode()
    ).hexdigest()


def test_sdk_retry_keeps_returned_response_cost_but_not_a_false_total(lab):
    lab.retry_attempts[0] = 1
    row = lab.ns["run_case"](lab.ns["WORKHORSE"], "Classify.", label="retry")
    assert row["sdk_retries"] == 1 and row["usage_complete"] is False
    assert row["cost_usd"] is None
    assert row["returned_response_cost_usd"] > 0
    assert isinstance(row["price_note"], str)
    assert "earlier attempts is unknown" in row["price_note"]
    execute("low-03-88a27752", lab.ns)
    reports = lab.ns["selection_summary"]
    assert reports and all(p["Model cost (USD)"] is None and p["USD / solved task"] is None for p in reports)


def test_native_request_still_validates_before_inference(lab):
    with pytest.raises(UnsupportedFeatureError):
        lab.ns["run_case"](lab.ns["WORKHORSE"], "Classify.", label="invalid", temperature=0)
    assert not lab.calls and not lab.recorded and not lab.ns["RUN_ROWS"]


def test_representative_cells_preserve_call_counts_and_show_sent_fields(lab):
    for cell_id, count in (
        ("low-03-88a27752", 8),
        ("low-05-9ee8d1ad", 2),
        ("low-06-761f76d3", 6),
        ("low-08-d0e20408", 0),
        ("low-prompted-json", 1),
        ("low-forced-tool", 1),
        ("low-native-schema", 2),
        ("low-10-0a475b0d", 2),
        ("low-temperature-comparison", 9),
        ("low-stop-sequence-comparison", 2),
    ):
        before = len(lab.calls)
        execute(cell_id, lab.ns)
        assert len(lab.calls) - before == count, cell_id
    assert len(lab.calls) == len(lab.recorded) == len(lab.ns["RUN_ROWS"])
    printed = [item for item in lab.printed if isinstance(item, dict)]
    assert [p["Converse request"] for p in printed if "Converse request" in p] == [
        lab.calls[0], lab.calls[4],
    ]
    for field in ("toolConfig", "outputConfig", "inferenceConfig"):
        displays = [p[field] for p in printed if field in p]
        assert displays
        assert all(any(request.get(field) == value for request in lab.calls) for value in displays)


def test_email_contract_preserves_raw_failures_and_checks_every_required_detail(lab):
    execute("low-05-9ee8d1ad", lab.ns)
    check = lab.ns["email_extraction_checks"]
    correct = {
        "issue": ["cracked case", "left earbud won't charge"], "sentiment": "negative",
        "action": "replacement", "deadline": "before the weekend",
    }
    assert all(check(json.dumps(correct), "end_turn").values())
    assert not all(check(json.dumps(correct), "max_tokens").values())
    for field, wrong in [
        ("issue", ["cracked case"]), ("issue", ["left earbud won't charge"]),
        ("issue", "defective product"), ("sentiment", "positive"),
        ("action", "refund"), ("deadline", None), ("deadline", "next month"),
        ("extra", "unsupported"),
    ]:
        assert not all(check(json.dumps({**correct, field: wrong}), "end_turn").values())
    compressed = '{"issue":"defective product","sentiment":"negative","action":"replace"}'
    for invalid in [compressed, "```json\n" + json.dumps(correct) + "\n```", "[]", "null", "billing"]:
        assert not all(check(invalid, "end_turn").values())
    # The model stub returned failures: no correction, stripping, or hidden retry.
    assert len(lab.calls) == len(lab.ns["prompt_design_rows"]) == 2
    assert all(not result["passed"] and result["row"]["text"] == "billing"
               for result in lab.ns["prompt_design_rows"])
    assert all(request["modelId"] == lab.ns["SMALL"] for request in lab.calls)
    assert all("outputConfig" not in request for request in lab.calls)  # This comparison teaches prompting.
    assert "all defect phrases, verbatim" in lab.ns["STRUCTURED_PROMPT"]
    assert "deadline (requested timing, verbatim)" in lab.ns["STRUCTURED_PROMPT"]


CLASSIFIED_EMAIL = {
    "issue": "cracked screen; the charger is missing",
    "sentiment": "negative", "action": "replacement",
}


def run_output_methods(lab, *, data=None, text=None, tool_calls=None,
                       text_stop="end_turn", tool_stop="tool_use"):
    """Supply responses to the existing native call recorder, never repair them."""
    payload = copy.deepcopy(CLASSIFIED_EMAIL if data is None else data)
    raw = json.dumps(payload) if text is None else text
    proposed = copy.deepcopy(tool_calls) if tool_calls is not None else [{
        "toolUseId": "tool-1", "name": "classify_email", "input": payload,
    }]
    original_converse = lab.ns["RUNTIME"].converse

    def converse(**request):
        response = original_converse(**request)
        is_tool = "toolConfig" in request
        response["output"]["message"]["content"] = (
            [{"toolUse": call} for call in proposed] if is_tool else [{"text": raw}]
        )
        response["stopReason"] = tool_stop if is_tool else text_stop
        return response

    lab.ns["RUN_GPT"] = True
    lab.ns["RUNTIME"].converse = converse
    for cell_id in ("low-08-d0e20408", "low-prompted-json", "low-forced-tool", "low-native-schema"):
        execute(cell_id, lab.ns)
    return lab.ns["output_method_rows"]


def test_output_methods_share_customer_request_contract_and_report_content_separately(lab):
    rows = run_output_methods(lab)
    assert len(rows) == len(lab.calls) == 5  # Prompt, strict tool, native Haiku, Sonnet and GPT.
    assert all(r["schema_valid"] and r["semantic_valid"] and r["passed"] for r in rows)
    assert all(all(r["semantic_checks"].values()) for r in rows)
    assert [r["model_id"] for r in rows] == [
        lab.ns["SMALL"], lab.ns["SMALL"], lab.ns["SMALL"], lab.ns["WORKHORSE"], lab.ns["GPT"],
    ]
    properties = lab.ns["SCHEMA"]["properties"]
    assert properties["issue"]["type"] == "string"
    assert all(properties[name].get("description") for name in properties)
    assert properties["action"]["enum"] == ["replacement", "refund", "repair", "other", "none"]
    assert "customer" in properties["action"]["description"].lower()
    assert "approval" in properties["action"]["description"].lower()
    assert all(request["system"][0]["text"].startswith(lab.ns["EXTRACTION_RULES"]) for request in lab.calls)
    native = [json.loads(request["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"])
              for request in lab.calls if "outputConfig" in request]
    tool_schema = lab.calls[1]["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]
    assert native == [lab.ns["SCHEMA"]] * 3
    assert tool_schema == lab.ns["SCHEMA"]
    assert lab.calls[1]["toolConfig"]["tools"][0]["toolSpec"]["strict"] is True
    assert len(rows) == 5 and all("semantic_checks" in row for row in rows)


@pytest.mark.parametrize("issue", [
    "cracked screen; missing charger",
    "screen is cracked; charger is missing",
    "cracked screen; charger missing from order",
    "cracked screen on arrival; charger missing from package",
    "the screen is cracked on arrival; the charger is missing from the box",
])
def test_output_methods_accept_equivalent_specific_defect_phrases(lab, issue):
    rows = run_output_methods(lab, data={**CLASSIFIED_EMAIL, "issue": issue})
    assert all(row["semantic_valid"] and row["passed"] for row in rows)


@pytest.mark.parametrize("field,value,failed_check,schema_valid", [
    ("issue", "damaged product and missing item", "cracked_screen", True),
    ("issue", "cracked screen", "missing_charger", True),
    ("issue", "the charger is missing", "cracked_screen", True),
    ("issue", "screen is not cracked; missing charger", "cracked_screen", True),
    ("issue", "cracked screen; charger is not missing", "missing_charger", True),
    ("issue", "screen is not cracked on arrival; missing charger", "cracked_screen", True),
    ("issue", "cracked screen; charger is not missing from package", "missing_charger", True),
    ("issue", "cracked screen; charger missing from package but later found", "missing_charger", True),
    ("sentiment", "neutral", "negative_sentiment", True),
    ("action", "refund", "requested_replacement", True),
    ("action", "replacement request", "requested_replacement", False),
    ("action", "Process replacement order and ship new unit", "requested_replacement", False),
    ("action", "Send replacement order", "requested_replacement", False),
    ("action", "Arrange replacement", "requested_replacement", False),
])
def test_output_methods_retain_raw_wrong_facts_and_fulfilment_assumptions(
    lab, field, value, failed_check, schema_valid,
):
    payload = {**CLASSIFIED_EMAIL, field: value}
    rows = run_output_methods(lab, data=payload)
    assert all(r["schema_valid"] is schema_valid for r in rows)
    assert all(not r["semantic_checks"][failed_check] and not r["passed"] for r in rows)
    assert all(r["data"] == payload for r in rows)
    assert len(lab.calls) == 5  # No rewrite, retry, model substitution or tool execution.
    for result in rows:
        row = result["row"]
        if row["label"] == "L02-strict-tool":
            assert row["response"]["output"]["message"]["content"][0]["toolUse"]["input"] == payload
        else:
            assert row["text"] == json.dumps(payload)


@pytest.mark.parametrize("raw", [
    "```json\n" + json.dumps(CLASSIFIED_EMAIL) + "\n```",
    "Here is the result: " + json.dumps(CLASSIFIED_EMAIL),
    "[]", "null",
])
def test_output_methods_never_repair_fences_or_nonobjects(lab, raw):
    rows = run_output_methods(lab, text=raw)
    text_rows = [r for r in rows if r["method"] != "L02-strict-tool"]
    assert len(text_rows) == 4
    assert all(not r["schema_valid"] and not r["passed"] for r in text_rows)
    assert all(r["raw"] == r["row"]["text"] == raw for r in text_rows)
    assert rows[1]["passed"]  # The separate valid tool proposal remains valid.


@pytest.mark.parametrize("names", [[], ["wrong_tool"], ["classify_email", "classify_email"]])
def test_output_methods_require_exactly_one_expected_forced_tool(lab, names):
    proposals = [
        {"toolUseId": f"call-{i}", "name": name, "input": copy.deepcopy(CLASSIFIED_EMAIL)}
        for i, name in enumerate(names)
    ]
    rows = run_output_methods(lab, tool_calls=proposals)
    result = rows[1]
    assert result["tool_calls"] == len(proposals)
    assert result["method_checks"]["one_expected_tool"] is False
    assert not result["passed"]
    assert result["raw"] == proposals
    assert len(lab.calls) == 5


@pytest.mark.parametrize("text_stop,tool_stop", [
    ("max_tokens", "max_tokens"), ("guardrail_intervened", "end_turn"),
])
def test_output_methods_require_normal_completion_even_with_correct_payload(lab, text_stop, tool_stop):
    rows = run_output_methods(lab, text_stop=text_stop, tool_stop=tool_stop)
    assert all(r["schema_valid"] and r["semantic_valid"] for r in rows)
    assert all(not r["method_checks"]["complete"] and not r["passed"] for r in rows)


def test_self_refine_requests_only_customer_reply_and_keeps_three_call_measurement(lab):
    lab.ns["RUN_SELF_REFINE"] = True
    execute("low-18-d729885a", lab.ns)
    assert len(lab.calls) == 3
    draft, critique, revision = [
        request["messages"][0]["content"][0]["text"] for request in lab.calls
    ]
    assert "charged twice for October" in draft
    assert "do NOT rewrite" in critique
    assert "Return only the customer-facing reply" in revision
    assert "Do not include critique, change notes" in revision and "Changes made" in revision
    assert "Do not claim a refund has been issued" in revision
    report = lab.ns["refinement_summary"]
    assert report["calls"] == 3
    assert report["workflow_cost_usd"] == pytest.approx(sum(r["cost_usd"] for r in lab.ns["refinement_rows"]))


@pytest.mark.parametrize("retry_attempts", [0, 1])
def test_cache_displays_raw_counters_once_without_an_extra_call(lab, retry_attempts):
    lab.usage.update({
        "cacheReadInputTokens": 80, "cacheWriteInputTokens": 20,
        "cacheDetails": [{"ttl": "5m", "inputTokens": 20}],
    })
    lab.retry_attempts[0] = retry_attempts
    execute("low-cache-ttl-choice", lab.ns)
    execute("low-12-f22b2184", lab.ns)
    execute("low-cache-history", lab.ns)
    execute("low-cache-change", lab.ns)  # Remains gated off.
    assert len(lab.calls) == 4
    lab.ns["RUN_PREFIX_CHANGE"] = True
    execute("low-cache-change", lab.ns)
    assert len(lab.calls) == 5
    # One metadata object per answer, preserving native counters and TTL details.
    displays = [p for p in lab.printed if isinstance(p, dict)]
    assert len(displays) == 5
    for metadata, row in zip(displays, lab.ns["RUN_ROWS"], strict=True):
        assert metadata["api"] == "Converse"
        assert metadata["model"] == row["model_id"]
        assert metadata["usage"] == row["response"]["usage"] == lab.usage
        assert metadata["latency_ms"] == round(row["latency_ms"], 1)
        assert metadata["stop_reason"] == row["stop_reason"]
        assert metadata["estimated_cost_usd"] == (
            round(row["cost_usd"], 8) if row["cost_usd"] is not None else None
        )
        if retry_attempts:
            assert metadata["estimated_cost_usd"] is None
            assert row["price_note"] in lab.printed
    questions = lab.ns["cache_questions"]
    assert [p for p in lab.printed if isinstance(p, str) and p.startswith("\nQuestion:")] == [
        f"\nQuestion: {question}\nbilling" for question in questions * 2 + questions[:1]
    ]
    assert all(row["cache_read_tokens"] == 80 and row["cache_write_tokens"] == 20
               and row["cache_write_by_ttl"] == {"5m": 20} for row in lab.ns["RUN_ROWS"])
    assert lab.calls[3]["messages"][1] == lab.ns["RUN_ROWS"][2]["response"]["output"]["message"]
    assert lab.calls[0]["system"] == lab.calls[1]["system"]
    assert lab.calls[4]["system"][0]["text"].startswith("Reference edition 2.")


@pytest.fixture
def evaluation(lab):
    row = lab.ns["run_case"](lab.ns["SMALL"], "Classify this charge.", label="selection")
    session_id = row["span"]["attributes"]["session.id"]
    results = [{
        "evaluatorArn": "arn:aws:bedrock-agentcore:::evaluator/Builtin.Correctness",
        "evaluatorId": "Builtin.Correctness", "evaluatorName": "Correctness",
        "context": {"spanContext": {"sessionId": session_id, "traceId": TRACE}},
        "value": 0.0, "label": "Incorrect",
        "tokenUsage": {"inputTokens": 600, "outputTokens": 80, "totalTokens": 680},
    }]
    response = {"evaluationResults": results}
    requests, flushes = [], []
    members = {"evaluationReferenceInputs": object()}

    def evaluate(**request):
        requests.append(copy.deepcopy(request))
        return response

    lab.ns.update(
        RUN_EVAL=True, selection_rows=[row], LABELLED_TICKETS=[("Classify this charge.", "billing")],
        AGENTCORE=SimpleNamespace(
            evaluate=evaluate,
            meta=SimpleNamespace(service_model=SimpleNamespace(
                operation_model=lambda operation: SimpleNamespace(
                    input_shape=SimpleNamespace(members=members)
                )
            )),
        ),
        OBS=SimpleNamespace(flush=lambda: flushes.append(True) or True, backend="agentcore",
                            region="us-east-1", service_name="playbook-low"),
    )
    return SimpleNamespace(lab=lab, requests=requests, response=response, results=results,
                           members=members, flushes=flushes, row=row, session_id=session_id)


def expected_evaluate_request(evaluation):
    return {
        "evaluatorId": "Builtin.Correctness",
        "evaluationInput": {"sessionSpans": [evaluation.row["span"]]},
        "evaluationTarget": {"traceIds": [TRACE]},
        "evaluationReferenceInputs": [{
            "context": {"spanContext": {"sessionId": evaluation.session_id, "traceId": TRACE}},
            "expectedResponse": {"text": "billing"},
        }],
    }


def test_inline_evaluate_targets_existing_response_and_keeps_judge_usage_separate(evaluation):
    execute("low-24-098e0e64", evaluation.lab.ns)
    assert evaluation.requests == [expected_evaluate_request(evaluation)]
    assert len(evaluation.lab.calls) == len(evaluation.lab.ns["RUN_ROWS"]) == 1
    assert evaluation.row["input_tokens"] == 100 and evaluation.row["output_tokens"] == 10
    assert evaluation.results in evaluation.lab.printed
    assert evaluation.flushes == [True]


@pytest.mark.parametrize("backend", ["agentcore", "both", "langfuse", "none"])
def test_low_console_link_opens_the_service_page_without_running_evaluation(evaluation, backend):
    evaluation.lab.ns["RUN_EVAL"] = False
    evaluation.lab.ns["OBS"].backend = backend
    execute("low-24-098e0e64", evaluation.lab.ns)
    assert not evaluation.requests and evaluation.flushes == [True]
    if backend in {"agentcore", "both"}:
        assert evaluation.lab.displayed[-1].data == (
            "[Open playbook-low in CloudWatch GenAI Observability]"
            "(https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1"
            "#gen-ai-observability/agent-core/agent/playbook-low?serviceName=playbook-low&source=agents)"
        )
    else:
        assert not evaluation.lab.displayed


@pytest.mark.parametrize("change", ["disabled", "empty", "no_span"])
def test_evaluate_gates_make_no_request(evaluation, change):
    if change == "disabled":
        evaluation.lab.ns["RUN_EVAL"] = False
    elif change == "empty":
        evaluation.lab.ns["selection_rows"] = []
    else:
        evaluation.row["span"] = None
    execute("low-24-098e0e64", evaluation.lab.ns)
    assert not evaluation.requests and evaluation.flushes == [True]


@pytest.mark.parametrize("failure", [
    "missing_results", "empty_results", "errorCode", "errorMessage", "ignoredReferenceInputFields",
    "wrong_evaluator", "wrong_session", "wrong_trace", "missing_score",
])
def test_evaluate_rejects_unusable_or_mismatched_results(evaluation, failure):
    result = evaluation.results[0]
    if failure == "missing_results":
        evaluation.response.clear()
    elif failure == "empty_results":
        evaluation.response["evaluationResults"] = []
    elif failure in {"errorCode", "errorMessage", "ignoredReferenceInputFields"}:
        result[failure] = ["expectedResponse"] if failure == "ignoredReferenceInputFields" else "failure"
    elif failure == "wrong_evaluator":
        result["evaluatorId"] = "Builtin.Helpfulness"
    elif failure in {"wrong_session", "wrong_trace"}:
        key = "sessionId" if failure == "wrong_session" else "traceId"
        result["context"]["spanContext"][key] = "c" * 32
    else:
        result.pop("value")
        result.pop("label")
    with pytest.raises(RuntimeError):
        execute("low-24-098e0e64", evaluation.lab.ns)
    assert len(evaluation.requests) == 1


@pytest.mark.parametrize("failure", ["trace", "span", "session", "answer", "reference", "sdk"])
def test_evaluate_rejects_invalid_input_before_call(evaluation, failure):
    span = evaluation.row["span"]
    if failure in {"trace", "span"}:
        span["traceId" if failure == "trace" else "spanId"] = "0" * (32 if failure == "trace" else 16)
    elif failure == "session":
        span["attributes"]["session.id"] = ""
    elif failure == "answer":
        span["attributes"]["gen_ai.task.output"] = ""
    elif failure == "reference":
        evaluation.lab.ns["LABELLED_TICKETS"] = [("question", "")]
    else:
        evaluation.members.clear()
    with pytest.raises((ValueError, RuntimeError)):
        execute("low-24-098e0e64", evaluation.lab.ns)
    assert not evaluation.requests


def test_inline_evaluate_matches_installed_sdk_shape(evaluation):
    boto3 = pytest.importorskip("boto3")
    stub = pytest.importorskip("botocore.stub")
    client = boto3.client("bedrock-agentcore", region_name="us-east-1",
                          aws_access_key_id="offline", aws_secret_access_key="offline")
    evaluation.lab.ns["AGENTCORE"] = client
    try:
        with stub.Stubber(client) as stubber:
            stubber.add_response("evaluate", evaluation.response, expected_evaluate_request(evaluation))
            execute("low-24-098e0e64", evaluation.lab.ns)
            stubber.assert_no_pending_responses()
    finally:
        client.close()


@pytest.mark.parametrize("settings_id,experiment_id,flag", [
    ("low-prefix-options", "low-cache-change", "RUN_PREFIX_CHANGE"),
    ("low-responses-options", "low-14-53f48222", "RUN_GPT_RESPONSES"),
    ("low-refine-options", "low-18-d729885a", "RUN_SELF_REFINE"),
    ("low-bounded-options", "low-20-0e46be9c", "RUN_BOUNDED_TASK"),
])
def test_visible_optional_settings_default_off_without_calls(lab, settings_id, experiment_id, flag):
    execute(settings_id, lab.ns)
    assert lab.ns[flag] is False
    lab.ns["RUN_GPT"] = True  # Enabling provider comparisons must not enable a separate optional API.
    execute(experiment_id, lab.ns)
    assert not lab.calls
    assert "environ.get(\"RUN_" not in cells()[experiment_id]


def test_visible_gpt_option_controls_the_model_comparison(lab):
    options = cells()["low-options"].replace("RUN_GPT = False", "RUN_GPT = True")
    exec(compile(options, str(NOTEBOOK), "exec"), lab.ns)
    execute("low-03-88a27752", lab.ns)
    assert len(lab.calls) == 12
    assert {call["modelId"] for call in lab.calls} == {lab.ns["SMALL"], lab.ns["WORKHORSE"], lab.ns["GPT"]}


def test_unparseable_output_is_not_reported_as_wrong_facts(lab):
    rows = run_output_methods(lab, text="```json\n" + json.dumps(CLASSIFIED_EMAIL) + "\n```")
    for row in rows:
        if row["method"] != "L02-strict-tool":
            assert row["schema_valid"] is False
            assert row["semantic_valid"] is None
            assert all(value is None for value in row["semantic_checks"].values())
            assert row["passed"] is False
