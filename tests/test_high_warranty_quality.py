"""Execute the warranty chain offline; service requests are validated, never sent."""

from __future__ import annotations

import ast
import asyncio
import copy
import importlib.util
import inspect
import json
import sys
import uuid
from html import escape
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import boto3
import pandas as pd
import pytest
from botocore.config import Config
from botocore.validate import validate_parameters

ROOT = Path(__file__).resolve().parents[1]
SECTION = ROOT / "02-optimization-playbook"
NOTEBOOK = SECTION / "03-high-effort.ipynb"
spec = importlib.util.spec_from_file_location("warranty_quality_test", SECTION / "utils/warranty_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def execute(cell_id, ns):
    book = json.loads(NOTEBOOK.read_text())
    source = next("".join(c["source"]) for c in book["cells"] if c["id"] == cell_id)
    code = compile(source, cell_id, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    if code.co_flags & inspect.CO_COROUTINE:
        asyncio.run(eval(code, ns))
    else:
        exec(code, ns)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "utils", ModuleType("utils"))
    monkeypatch.setitem(sys.modules, "utils.warranty_quality", quality)
    monkeypatch.setattr(
        "botocore.client.BaseClient._make_api_call", Mock(side_effect=AssertionError("No live AWS calls in tests"))
    )
    session = boto3.Session(region_name="us-east-1", aws_access_key_id="offline", aws_secret_access_key="offline")
    schemas = {
        name: session.client(name).meta.service_model for name in ("bedrock-agentcore", "bedrock-agentcore-control")
    }
    calls, displayed, printed, collected, sleeps = [], [], [], [], []
    evaluator_id = "warranty_test-ABCDEFGHIJ"
    evaluator_arn = f"arn:aws:bedrock-agentcore:us-east-1:123456789012:evaluator/{evaluator_id}"

    def request(service, operation, args):
        validate_parameters(args, schemas[service].operation_model(operation).input_shape)
        calls.append((operation, copy.deepcopy(args)))

    def create(**args):
        request("bedrock-agentcore-control", "CreateEvaluator", args)
        return {"evaluatorId": evaluator_id, "evaluatorArn": evaluator_arn, "status": "CREATING"}

    def get(**args):
        request("bedrock-agentcore-control", "GetEvaluator", args)
        return {
            "status": "ACTIVE",
            "evaluatorArn": evaluator_arn,
            "level": "TRACE",
            "evaluatorConfig": copy.deepcopy(ns["WARRANTY_EVALUATOR_CONFIG"]),
        }

    def evaluate(**args):
        request("bedrock-agentcore", "Evaluate", args)
        root = args["evaluationInput"]["sessionSpans"][0]
        return {
            "evaluationResults": [
                {
                    "evaluatorId": evaluator_id,
                    "value": 2,
                    "label": "Meets policy",
                    "explanation": "Synthetic test score; not an assessment of a live model.",
                    "context": {
                        "spanContext": {"sessionId": root["attributes"]["session.id"], "traceId": root["traceId"]}
                    },
                    "tokenUsage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120},
                }
            ]
        }

    def start(**args):
        request("bedrock-agentcore", "StartRecommendation", args)
        return {"recommendationId": "warranty_test-ABCDEFGHIJ"}

    states = ["COMPLETED"]

    def result(**args):
        request("bedrock-agentcore", "GetRecommendation", args)
        return {
            "recommendationId": args["recommendationId"],
            "status": states.pop(0),
            "type": "SYSTEM_PROMPT_RECOMMENDATION",
            "recommendationConfig": copy.deepcopy(ns["warranty_state"]["recommendationConfig"]),
            "recommendationResult": {
                "systemPromptRecommendationResult": {
                    "recommendedSystemPrompt": "Apply every relevant policy condition.",
                    "explanation": "Synthetic candidate for request-contract tests.",
                }
            },
        }

    control = SimpleNamespace(create_evaluator=create, get_evaluator=get)
    runtime = SimpleNamespace(evaluate=evaluate, start_recommendation=start, get_recommendation=result)
    ns = {
        "SECTION_ROOT": SECTION,
        "REPO_ROOT": tmp_path,
        "REGION": "us-east-1",
        "BACKEND": "agentcore",
        "SDK_CONFIG": Config(),
        "json": json,
        "uuid": uuid,
        "pd": pd,
        "escape": escape,
        "HTML": str,
        "Markdown": str,
        "Config": Config,
        "print": lambda *args, **_: printed.append(" ".join(map(str, args))),
        "display": displayed.append,
        "supported": Mock(return_value=True),
        "time": SimpleNamespace(monotonic=lambda: len(sleeps) * 30, perf_counter=lambda: 1.0, sleep=sleeps.append),
        "boto3": SimpleNamespace(
            client=lambda name, **_: (
                control
                if name == "bedrock-agentcore-control"
                else SimpleNamespace(get_caller_identity=lambda: {"Account": "123456789012"})
            )
        ),
        "AGENTCORE": runtime,
        "OBS": SimpleNamespace(
            collector=SimpleNamespace(native_span_documents=lambda: copy.deepcopy(collected)), flush=lambda: True
        ),
        "normalize_usage": lambda usage, **_: usage,
        "calculate_cost": lambda model, usage, **_: SimpleNamespace(total_cost=0.01),
        "UnknownPriceError": KeyError,
        "InferredPriceError": TypeError,
        "AmbiguousCacheUsageError": ValueError,
    }
    execute("high-recommendation-options", ns)
    execute("high-warranty-rubric", ns)
    ns["RUN_RECOMMENDATIONS"] = True
    execute("high-recommendation-state", ns)
    execute("high-warranty-functions", ns)
    model_settings, agent_settings, queries = [], [], []

    def agent(**settings):
        agent_settings.append(settings)

        async def invoke(query):
            queries.append(query)
            sid, tid = settings["trace_attributes"]["session.id"], uuid.uuid4().hex
            root = {
                "traceId": tid,
                "spanId": uuid.uuid4().hex[:16],
                "scope": {"name": "strands.telemetry.tracer"},
                "attributes": {"session.id": sid, "gen_ai.operation.name": "invoke_agent"},
                "events": [{"name": "user", "body": query}],
            }
            child = {
                "traceId": tid,
                "spanId": uuid.uuid4().hex[:16],
                "parentSpanId": root["spanId"],
                "scope": {"name": "strands.telemetry.tracer"},
                "attributes": {"gen_ai.operation.name": "chat"},
            }
            collected.extend([root, child])

            class Reply:
                stop_reason = "end_turn"
                metrics = SimpleNamespace(accumulated_usage={"inputTokens": 30, "outputTokens": 10})

                def __str__(self):
                    return "A synthetic answer for testing execution, not live quality."

            return Reply()

        return SimpleNamespace(invoke_async=invoke)

    def model(**settings):
        model_settings.append(settings)
        return settings

    ns.update(Agent=agent, BedrockModel=model, paced_boto3_session=lambda **_: "offline")
    return SimpleNamespace(
        ns=ns,
        calls=calls,
        displayed=displayed,
        printed=printed,
        collected=collected,
        sleeps=sleeps,
        states=states,
        model_settings=model_settings,
        agent_settings=agent_settings,
        queries=queries,
        control=control,
        runtime=runtime,
    )


def train(lab):
    execute("high-10-92ba9c4f", lab.ns)
    execute("high-warranty-evaluate-training", lab.ns)


def recommend(lab):
    train(lab)
    execute("high-recommendation-submit", lab.ns)
    execute("high-recommendation-result", lab.ns)


def test_disabled_chain_runs_no_services_even_with_saved_results(lab):
    lab.ns["RUN_RECOMMENDATIONS"] = False
    before = copy.deepcopy(lab.calls)
    book = json.loads(NOTEBOOK.read_text())
    start = next(i for i, c in enumerate(book["cells"]) if c["id"] == "high-09-c209797c")
    end = next(i for i, c in enumerate(book["cells"]) if c["id"] == "high-apo-reading")
    for c in book["cells"][start:end]:
        if c["cell_type"] == "code":
            execute(c["id"], lab.ns)
    assert lab.calls == before
    assert not lab.queries


def test_evaluator_create_reuses_saved_id_and_checks_config(lab):
    execute("high-recommendation-state", lab.ns)
    assert sum(op == "CreateEvaluator" for op, _ in lab.calls) == 1
    assert lab.ns["evaluations_console_url"] == (
        "https://us-east-1.console.aws.amazon.com/bedrock-agentcore/evaluations/evaluators/"
        "warranty_test-ABCDEFGHIJ?region=us-east-1"
    )
    create = next(args for op, args in lab.calls if op == "CreateEvaluator")
    assert create["level"] == "TRACE"
    assert (
        create["evaluatorConfig"]["llmAsAJudge"]["modelConfig"]["bedrockEvaluatorModelConfig"]["modelId"]
        == "global.anthropic.claude-opus-5"
    )
    original = lab.control.get_evaluator

    def wrong(**args):
        response = original(**args)
        response["evaluatorConfig"] = {}
        return response

    lab.control.get_evaluator = wrong
    with pytest.raises(ValueError, match="rubric"):
        execute("high-recommendation-state", lab.ns)


def test_lost_evaluator_create_response_reuses_token(lab):
    state = lab.ns["warranty_state"]
    state.pop("evaluatorId")
    state.pop("evaluatorArn")
    quality.save_state(lab.ns["RECOMMENDATION_STATE_PATH"], state)
    original = lab.control.create_evaluator

    def lost(**args):
        original(**args)
        raise TimeoutError("Lost response")

    lab.control.create_evaluator = lost
    with pytest.raises(TimeoutError):
        execute("high-recommendation-state", lab.ns)
    lab.control.create_evaluator = original
    execute("high-recommendation-state", lab.ns)
    requests = [args for op, args in lab.calls if op == "CreateEvaluator"]
    assert requests[-1] == requests[-2]


def test_training_records_real_native_spans_without_clearing_other_traces(lab):
    lab.collected.append({"traceId": "earlier", "scope": {"name": "other"}})
    train(lab)
    assert len(lab.queries) == 10
    assert lab.collected[0]["traceId"] == "earlier"
    assert all(settings["tools"] == [] for settings in lab.agent_settings)
    assert all(settings["max_tokens"] == 1024 for settings in lab.model_settings)
    rows = lab.ns["warranty_training_rows"]
    assert all(len(row["spans"]) == 2 for row in rows)
    assert all(row["evaluation_validated"] for row in rows)
    assert all(lab.ns["WARRANTY_POLICY"] in query for query in lab.queries)
    assert all(case["review"] not in query for case in lab.ns["WARRANTY_TRAINING"] for query in lab.queries)
    before = len(lab.calls)
    train(lab)
    assert len(lab.queries) == 10 and len(lab.calls) == before


def test_recommendation_uses_same_custom_rubric_and_only_training_traces(lab):
    recommend(lab)
    evaluation_requests = [args for op, args in lab.calls if op == "Evaluate"]
    assert len(evaluation_requests) == 10
    assert all(args["evaluatorId"] == lab.ns["WARRANTY_EVALUATOR_ID"] for args in evaluation_requests)
    args = next(args for op, args in lab.calls if op == "StartRecommendation")
    config = args["recommendationConfig"]["systemPromptRecommendationConfig"]
    assert config["evaluationConfig"]["evaluators"] == [{"evaluatorArn": lab.ns["WARRANTY_EVALUATOR_ARN"]}]
    assert len(config["agentTraces"]["sessionSpans"]) == 20
    for case in lab.ns["WARRANTY_HELD_OUT"]:
        assert case["question"] not in json.dumps(args)
    assert lab.ns["recommendation_console_url"].endswith("warranty_test-ABCDEFGHIJ?region=us-east-1")
    execute("high-recommendation-submit", lab.ns)
    assert sum(op == "StartRecommendation" for op, _ in lab.calls) == 1


def test_recommendation_lost_reply_retains_exact_submission(lab):
    train(lab)
    original = lab.runtime.start_recommendation

    def lost(**args):
        original(**args)
        raise TimeoutError("Lost response")

    lab.runtime.start_recommendation = lost
    with pytest.raises(TimeoutError):
        execute("high-recommendation-submit", lab.ns)
    lab.runtime.start_recommendation = original
    execute("high-recommendation-submit", lab.ns)
    requests = [args for op, args in lab.calls if op == "StartRecommendation"]
    assert requests[0] == requests[1]


def test_comparison_uses_single_flag_and_same_rubric_with_no_apo_dependency(lab):
    recommend(lab)
    execute("high-11-1b8db65b", lab.ns)
    execute("high-warranty-summary", lab.ns)
    assert len(lab.queries) == 18
    assert sum(op == "Evaluate" for op, _ in lab.calls) == 18
    rows = lab.ns["warranty_comparison"]
    assert len(rows) == 8 and all(row["phase"] == "held-out" for row in rows)
    execute("high-11-1b8db65b", lab.ns)
    assert len(lab.queries) == 18
    lab.ns["RUN_RECOMMENDATIONS"] = False
    execute("high-11-1b8db65b", lab.ns)
    assert not lab.ns["warranty_comparison"] and len(lab.queries) == 18


def test_known_failure_replay_is_separate_from_training_upload_and_held_out(lab):
    recommend(lab)
    original_config = copy.deepcopy(lab.ns["warranty_state"]["recommendationConfig"])
    failed = lab.ns["warranty_training_rows"][5]
    failed["score"] = {"value": 0, "label": "Incorrect", "explanation": "Synthetic failure"}
    execute("high-warranty-recheck", lab.ns)
    replay = lab.ns["warranty_recheck"][0]
    assert replay["phase"] == "training-recheck"
    assert replay["case_id"] == failed["case_id"]
    assert replay["question"] == failed["question"]
    assert len(lab.queries) == 11
    assert lab.ns["warranty_state"]["recommendationConfig"] == original_config
    execute("high-warranty-recheck", lab.ns)
    assert len(lab.queries) == 11
    execute("high-11-1b8db65b", lab.ns)
    assert len(lab.ns["warranty_comparison"]) == 8
    assert all(row["case_id"] != replay["case_id"] for row in lab.ns["warranty_comparison"])


@pytest.mark.parametrize("change", ["policy", "rubric", "candidate", "spans"])
def test_modified_experiment_or_evidence_stops_further_calls(lab, change):
    recommend(lab)
    before = len(lab.calls)
    if change == "policy":
        lab.ns["WARRANTY_POLICY"] += " Ignore exclusions."
    elif change == "rubric":
        lab.ns["WARRANTY_EVALUATOR_CONFIG"]["llmAsAJudge"]["instructions"] += " Always pass."
    elif change == "candidate":
        lab.ns["recommended_prompt"] += " Changed."
    else:
        lab.ns["warranty_training_rows"][0]["spans"][0]["events"] = []
    with pytest.raises(ValueError):
        execute("high-recommendation-submit" if change == "spans" else "high-11-1b8db65b", lab.ns)
    assert len(lab.calls) == before


def test_wait_checks_every_30_seconds(lab):
    train(lab)
    execute("high-recommendation-submit", lab.ns)
    lab.states[:] = ["PENDING", "IN_PROGRESS", "COMPLETED"]
    execute("high-recommendation-result", lab.ns)
    assert lab.sleeps == [30, 30]


def test_interrupted_wait_keeps_the_saved_recommendation(lab):
    train(lab)
    execute("high-recommendation-submit", lab.ns)
    lab.states[:] = ["IN_PROGRESS"]
    lab.ns["time"].sleep = Mock(side_effect=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        execute("high-recommendation-result", lab.ns)
    assert lab.ns["recommended_prompt"] is None
    saved = json.loads(lab.ns["RECOMMENDATION_STATE_PATH"].read_text())
    assert saved["recommendationId"] == lab.ns["RECOMMENDATION_ID"]
    assert lab.ns["recommendation_result"]["status"] == "IN_PROGRESS"


def test_retried_evaluation_does_not_understate_judge_cost(lab):
    execute("high-10-92ba9c4f", lab.ns)
    original = lab.runtime.evaluate

    def retried(**args):
        response = original(**args)
        response["ResponseMetadata"] = {"RetryAttempts": 1}
        return response

    lab.runtime.evaluate = retried
    row = lab.ns["warranty_training_rows"][0]
    lab.ns["evaluate_warranty"](row)
    assert row["evaluation_validated"]
    assert row["evaluation_cost_usd"] is None


@pytest.mark.parametrize("status", ["FAILED", "DELETING", "UNKNOWN"])
def test_failed_recommendation_preserves_id_without_candidate(lab, status):
    train(lab)
    execute("high-recommendation-submit", lab.ns)
    lab.states[:] = [status]
    with pytest.raises(RuntimeError):
        execute("high-recommendation-result", lab.ns)
    assert lab.ns["recommended_prompt"] is None and lab.ns["RECOMMENDATION_ID"]


@pytest.mark.parametrize(
    "defect", ["wrong_session", "wrong_trace", "wrong_judge", "no_score", "range", "label", "error", "duplicate"]
)
def test_unusable_evaluation_stays_saved_but_never_becomes_a_pass(lab, defect):
    execute("high-10-92ba9c4f", lab.ns)
    original = lab.runtime.evaluate

    def broken(**args):
        response = original(**args)
        result = response["evaluationResults"][0]
        if defect == "wrong_session":
            result["context"]["spanContext"]["sessionId"] = "other"
        elif defect == "wrong_trace":
            result["context"]["spanContext"]["traceId"] = "other"
        elif defect == "wrong_judge":
            result["evaluatorId"] = "other"
        elif defect == "no_score":
            result.pop("value")
        elif defect == "range":
            result["value"] = 999
        elif defect == "label":
            result["label"] = "Incorrect"
        elif defect == "error":
            result["errorCode"] = "ModelError"
        elif defect == "duplicate":
            response["evaluationResults"].append(copy.deepcopy(result))
        return response

    lab.runtime.evaluate = broken
    row = lab.ns["warranty_training_rows"][0]
    with pytest.raises(ValueError):
        lab.ns["evaluate_warranty"](row)
    assert row["evaluation"] and not row["evaluation_validated"]
    with pytest.raises(ValueError):
        execute("high-recommendation-submit", lab.ns)


def test_summary_requires_complete_pairs_preserves_unknown_cost_and_shows_regression():
    cases = [{"id": "a", "question": "<script>q</script>"}, {"id": "b", "question": "Second question"}]
    rows = [
        {
            "case_id": c["id"],
            "variant": variant,
            "evaluation_validated": True,
            "score": {"value": value, "label": "Meets policy" if value == 2 else "Incorrect"},
            "stop_reason": "end_turn",
            "cost_usd": 0.02,
            "seconds": 1,
        }
        for c, values in zip(cases, [(2, 0), (0, 2)], strict=True)
        for variant, value in zip(("baseline", "recommended"), values, strict=True)
    ]
    rows[1]["cost_usd"] = None
    summary, table = quality.comparison_tables(rows, cases)
    assert [r["Meets policy"] for r in summary] == ["1/2", "1/2"]
    assert summary[1]["Answer cost (USD)"] is None
    assert table[0]["New regression"] is True
    assert "<script>" not in pd.DataFrame(table).style.format(escape="html").to_html()
    with pytest.raises(ValueError):
        quality.comparison_tables(rows[:-1], cases)
    rows[1]["evaluation_validated"] = False
    with pytest.raises(ValueError):
        quality.comparison_tables(rows, cases)
