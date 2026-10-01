"""Exercise authored APO cells offline: job reuse, result identity and held-out checks."""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import boto3
import pytest
from botocore.config import Config
from botocore.validate import validate_parameters

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "02-optimization-playbook/03-high-effort.ipynb"
ACCOUNT = "123456789012"
JOB = f"arn:aws:bedrock:us-east-1:{ACCOUNT}:advanced-prompt-optimization-job/example"


def cell_source(cell_id):
    return next(
        "".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"] if c["id"] == cell_id
    )


def execute(cell_id, ns, *, submit=False):
    text = cell_source(cell_id)
    if submit:
        text = text.replace("RUN_APO = False", "RUN_APO = True", 1)
    exec(compile(text, str(NOTEBOOK), "exec"), ns)


@pytest.fixture
def lab(tmp_path):
    calls, uploads, printed = [], [], []
    status = {}
    output = {}
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="offline", aws_secret_access_key="offline"
    )
    client = session.client("bedrock")

    def create(**request):
        validate_parameters(request, client.meta.service_model.operation_model(
            "CreateAdvancedPromptOptimizationJob").input_shape)
        calls.append(copy.deepcopy(request))
        status.update(jobArn=JOB, jobStatus="InProgress", **copy.deepcopy(request))
        return {"jobArn": JOB}

    def get(**request):
        validate_parameters(request, client.meta.service_model.operation_model(
            "GetAdvancedPromptOptimizationJob").input_shape)
        assert request == {"jobIdentifier": JOB}
        return copy.deepcopy(status)

    def get_object(**request):
        expected = ns["apo_state"]["prefix"] + "output/example/advanced_prompt_optimization_results.jsonl"
        assert request == {"Bucket": "workshop-test", "Key": expected}
        return {"Body": io.BytesIO((json.dumps(output) + "\n").encode())}

    s3 = SimpleNamespace(
        get_bucket_location=lambda **_: {"LocationConstraint": None},
        put_object=lambda **kwargs: uploads.append(kwargs),
        get_object=get_object,
    )
    sts = SimpleNamespace(get_caller_identity=lambda: {"Account": ACCOUNT})
    ns = {
        "REPO_ROOT": tmp_path, "REGION": "us-east-1", "SDK_CONFIG": Config(),
        "json": json, "hashlib": hashlib, "uuid": uuid,
        "boto3": SimpleNamespace(client=lambda name, **_: {"s3": s3, "sts": sts}[name]),
        "BEDROCK": SimpleNamespace(create_advanced_prompt_optimization_job=create,
                                   get_advanced_prompt_optimization_job=get),
        "parameter_or_env": lambda *_: "workshop-test",
        "print": lambda *args, **_: printed.extend(args),
    }
    execute("high-apo-input", ns)
    yield SimpleNamespace(ns=ns, calls=calls, uploads=uploads, printed=printed,
                          status=status, output=output, s3=s3)
    client.close()


def completed(lab):
    execute("high-apo-submit", lab.ns, submit=True)
    lab.status["jobStatus"] = "Completed"
    lab.output.update({
        "promptTemplateId": lab.ns["APO_INPUT"]["templateId"],
        "promptOptimizationResults": [{
            "modelId": lab.ns["APO_MODEL_ID"], "status": "Completed",
            "optimizedPromptTemplate": "Return the correct category for {{ticket}}.",
        }],
    })


def test_preview_has_no_s3_upload_or_job_and_excludes_held_out(lab):
    execute("high-apo-submit", lab.ns)
    execute("high-apo-status", lab.ns)
    execute("high-apo-result", lab.ns)
    assert not lab.calls and not lab.uploads
    assert lab.ns["APO_JOB_ARN"] is None
    assert not lab.ns["APO_STATE_PATH"].exists()
    data = json.loads(lab.ns["APO_INPUT_BODY"])
    training = {s["inputVariables"][0]["ticket"] for s in data["evaluationSamples"]}
    assert len(training) == 6
    assert training.isdisjoint(ticket for ticket, _ in lab.ns["HELD_OUT"])
    assert data["customLLMJConfig"] and "steeringCriteria" not in data


def test_resume_reuses_persisted_job_and_input(lab):
    execute("high-apo-submit", lab.ns, submit=True)
    state = json.loads(lab.ns["APO_STATE_PATH"].read_text())
    assert state["jobArn"] == JOB
    assert state["clientToken"] == lab.calls[0]["clientToken"]
    execute("high-apo-submit", lab.ns, submit=True)
    assert len(lab.calls) == len(lab.uploads) == 1
    assert lab.ns["APO_JOB_ARN"] == JOB


def test_timed_out_create_reuses_saved_token(lab):
    create = lab.ns["BEDROCK"].create_advanced_prompt_optimization_job
    seen = []

    def timeout(**request):
        seen.append(request)
        raise TimeoutError("reply lost")

    lab.ns["BEDROCK"].create_advanced_prompt_optimization_job = timeout
    with pytest.raises(TimeoutError):
        execute("high-apo-submit", lab.ns, submit=True)
    lab.ns["BEDROCK"].create_advanced_prompt_optimization_job = create
    execute("high-apo-submit", lab.ns, submit=True)
    assert lab.calls[0] == seen[0]


def test_changed_config_and_wrong_region_cannot_submit_another_job(lab):
    execute("high-apo-submit", lab.ns, submit=True)
    lab.ns["APO_MODEL_CONFIGS"][0]["inferenceConfig"]["maxTokens"] = 512
    with pytest.raises(ValueError, match="differs"):
        execute("high-apo-submit", lab.ns, submit=True)
    assert len(lab.calls) == 1
    lab.s3.get_bucket_location = lambda **_: {"LocationConstraint": "us-west-2"}
    with pytest.raises(ValueError, match="same Region"):
        execute("high-apo-submit", lab.ns, submit=True)
    assert len(lab.calls) == 1


def test_completed_result_is_selected_and_incomplete_job_clears_candidate(lab):
    completed(lab)
    execute("high-apo-result", lab.ns)
    assert lab.ns["apo_candidate_template"].endswith("{{ticket}}.")
    assert lab.ns["apo_comparison_baseline"] == lab.ns["APO_BASELINE_TEMPLATE"]
    lab.ns["APO_MODEL_CONFIGS"][0]["inferenceConfig"]["maxTokens"] = 512
    assert lab.ns["apo_comparison_config"]["inferenceConfig"]["maxTokens"] == 1024
    for status in ("InProgress", "Failed", "PartiallyCompleted", "Stopped"):
        lab.status["jobStatus"] = status
        execute("high-apo-result", lab.ns)
        assert lab.ns["apo_candidate_template"] is None


def test_edited_dataset_does_not_relabel_old_job_results(lab):
    completed(lab)
    lab.ns["APO_INPUT_BODY"] = b'{"promptTemplate": "Always say billing"}\n'
    with pytest.raises(ValueError, match="dataset has changed"):
        execute("high-apo-result", lab.ns)
    assert lab.ns["apo_candidate_template"] is None


@pytest.mark.parametrize("problem", ["wrong_model", "wrong_template", "duplicate", "missing_variable",
                                    "extra_variable", "failure", "failed_status", "missing_status",
                                    "unknown_status", "changed_configuration"])
def test_unrelated_or_unusable_results_cannot_be_assessed(lab, problem):
    completed(lab)
    result = lab.output["promptOptimizationResults"][0]
    if problem == "wrong_model":
        result["modelId"] = "another-model"
    elif problem == "wrong_template":
        lab.output["promptTemplateId"] = "another-template"
    elif problem == "duplicate":
        lab.output["promptOptimizationResults"].append(copy.deepcopy(result))
    elif problem == "missing_variable":
        result["optimizedPromptTemplate"] = "Return billing."
    elif problem == "extra_variable":
        result["optimizedPromptTemplate"] += " {{unknown}}"
    elif problem == "failure":
        result["failureReason"] = "model access denied"
    elif problem == "failed_status":
        result["status"] = "Failed"
    elif problem == "missing_status":
        result.pop("status")
    elif problem == "unknown_status":
        result["status"] = "Unknown"
    else:
        lab.status["modelConfigurations"][0]["inferenceConfig"]["maxTokens"] = 512
    with pytest.raises(ValueError):
        execute("high-apo-result", lab.ns)
    assert lab.ns["apo_candidate_template"] is None


@pytest.mark.parametrize("change", ["job", "dataset", "config"])
def test_comparison_rejects_a_candidate_retained_from_another_experiment(lab, change):
    completed(lab)
    execute("high-apo-result", lab.ns)
    if change == "job":
        lab.ns["APO_JOB_ARN"] = None
    elif change == "dataset":
        lab.ns["APO_INPUT_BODY"] = b"changed"
    else:
        lab.ns["APO_MODEL_CONFIGS"][0]["inferenceConfig"]["maxTokens"] = 512
    with pytest.raises(ValueError, match="experiment changed"):
        execute("high-08-f889fbf9", lab.ns)


def test_held_out_comparison_uses_full_template_and_same_actual_settings(lab):
    calls = []
    ns = lab.ns
    ns["apo_candidate_template"] = "Classify {{ticket}}."
    ns["APO_MODEL_CONFIGS"][0]["inferenceConfig"]["maxTokens"] = 512
    ns["APO_MODEL_CONFIGS"][0]["additionalModelRequestFields"] = {"output_config": {"effort": "medium"}}
    ns["apo_comparison_config"] = copy.deepcopy(ns["APO_MODEL_CONFIGS"][0])
    ns["apo_comparison_baseline"] = ns["APO_BASELINE_TEMPLATE"]
    ns["APO_JOB_ARN"] = ns["apo_candidate_job_arn"] = JOB
    ns["apo_candidate_input_sha"] = hashlib.sha256(ns["APO_INPUT_BODY"]).hexdigest()
    expected_by_ticket = dict(ns["HELD_OUT"])

    def run_case(model_id, query, *, label, **controls):
        calls.append((model_id, query, controls))
        answer = next(expected for ticket, expected in expected_by_ticket.items() if ticket in query)
        return {"label": label, "text": answer, "stop_reason": "end_turn",
                "cost_usd": 0.001, "latency_ms": 100, "span": None}

    ns["run_case"] = run_case
    execute("high-08-f889fbf9", ns)
    assert len(calls) == 12
    assert all(r["correct"] for r in ns["optimization_assessment"])
    for model, query, controls in calls:
        assert model == ns["APO_MODEL_ID"]
        assert "{{ticket}}" not in query and "system" not in controls
        assert controls["max_tokens"] == 512
        assert controls["additional_model_request_fields"] == ns["APO_MODEL_CONFIGS"][0]["additionalModelRequestFields"]
    reports = [x for x in lab.printed if isinstance(x, dict) and "cost_per_correct_ticket" in x]
    assert len(reports) == 2
    assert all(r["cost_per_correct_ticket"] == pytest.approx(0.001) for r in reports)


def test_zero_success_has_no_cost_per_correct_ticket(lab):
    lab.ns.update(
        apo_candidate_template="Classify {{ticket}}.",
        apo_comparison_config=copy.deepcopy(lab.ns["APO_MODEL_CONFIGS"][0]),
        apo_comparison_baseline=lab.ns["APO_BASELINE_TEMPLATE"],
        APO_JOB_ARN=JOB, apo_candidate_job_arn=JOB,
        apo_candidate_input_sha=hashlib.sha256(lab.ns["APO_INPUT_BODY"]).hexdigest(),
        run_case=lambda *args, **kwargs: {
            "label": kwargs["label"], "text": "wrong", "stop_reason": "end_turn",
            "cost_usd": 0.01, "latency_ms": 100, "span": None,
        },
    )
    execute("high-08-f889fbf9", lab.ns)
    reports = [x for x in lab.printed if isinstance(x, dict) and "cost_per_correct_ticket" in x]
    assert all(r["cost_per_correct_ticket"] is None and r["correct"] == 0 for r in reports)


@pytest.mark.parametrize("failure", [None, "ignored_reference", "wrong_trace", "no_results", "error"])
def test_agentcore_evaluation_uses_recorded_answer_and_ground_truth(lab, failure):
    calls = []
    trace = "a" * 32
    span = {"traceId": trace, "spanId": "b" * 16, "attributes": {"session.id": "session"}}
    row = {"variant": "APO candidate", "expected": "billing", "correct": True, "span": span}
    session = boto3.Session(region_name="us-east-1", aws_access_key_id="offline", aws_secret_access_key="offline")
    client = session.client("bedrock-agentcore")
    result = {
        "evaluatorId": "Builtin.Correctness", "value": 1.0,
        "context": {"spanContext": {"sessionId": "session", "traceId": trace}},
    }
    if failure == "ignored_reference":
        result["ignoredReferenceInputFields"] = ["expectedResponse"]
    elif failure == "wrong_trace":
        result["context"]["spanContext"]["traceId"] = "c" * 32
    elif failure == "error":
        result["errorCode"] = "ModelError"

    def evaluate(**request):
        validate_parameters(request, client.meta.service_model.operation_model("Evaluate").input_shape)
        calls.append(request)
        return {"evaluationResults": [] if failure == "no_results" else [result]}

    lab.ns.update(RUN_EVAL=False, optimization_assessment=[row],
                  AGENTCORE=SimpleNamespace(evaluate=evaluate))
    execute("high-apo-evaluation", lab.ns)
    assert not calls
    lab.ns["RUN_EVAL"] = True
    try:
        if failure:
            with pytest.raises(RuntimeError):
                execute("high-apo-evaluation", lab.ns)
        else:
            execute("high-apo-evaluation", lab.ns)
        assert len(calls) == 1
        assert calls[0]["evaluationInput"] == {"sessionSpans": [span]}
        assert calls[0]["evaluationReferenceInputs"][0]["expectedResponse"] == {"text": "billing"}
    finally:
        client.close()


def test_setup_keeps_native_converse_and_evaluation_outside_assessment():
    setup = ast.parse(cell_source("high-setup-87841c4c"))
    calls = {ast.unparse(n.func) for n in ast.walk(setup) if isinstance(n, ast.Call)}
    assert "RUNTIME.converse" in calls and "traced_converse" not in calls
    assessment = cell_source("high-08-f889fbf9")
    assert "quick_eval" not in assessment and "AGENTCORE.evaluate" not in assessment
