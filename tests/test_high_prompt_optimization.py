"""Exercise authored APO cells offline: job reuse, result identity and held-out checks."""

from __future__ import annotations

import ast
import asyncio
import copy
import hashlib
import inspect
import io
import json
import uuid
from html import escape
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import boto3
import pandas as pd
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError
from botocore.validate import validate_parameters

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "02-optimization-playbook/03-high-effort.ipynb"
ACCOUNT = "123456789012"
JOB = f"arn:aws:bedrock:us-east-1:{ACCOUNT}:advanced-prompt-optimization-job/example"


def cell_source(cell_id):
    return next(
        "".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"] if c["id"] == cell_id
    )


def execute(cell_id, ns):
    code = compile(cell_source(cell_id), f"{NOTEBOOK.name}:{cell_id}", "exec",
                   flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    if code.co_flags & inspect.CO_COROUTINE:
        asyncio.run(eval(code, ns))
    else:
        exec(code, ns)


def submit(lab):
    lab.ns["RUN_APO"] = True
    execute("high-apo-state", lab.ns)
    execute("high-apo-submit", lab.ns)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    calls, uploads, printed, displays, reads = [], [], [], [], []
    status = {}
    output = {}
    monkeypatch.setattr("botocore.client.BaseClient._make_api_call",
                        Mock(side_effect=AssertionError("Unexpected live AWS request")))
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
        reads.append(request)
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
        "json": json, "hashlib": hashlib, "uuid": uuid, "ClientError": ClientError,
        "pd": pd, "display": displays.append, "HTML": str, "Markdown": str, "escape": escape,
        "time": SimpleNamespace(monotonic=lambda: 0, sleep=Mock(side_effect=AssertionError("Unexpected wait"))),
        "boto3": SimpleNamespace(client=lambda name, **_: {"s3": s3, "sts": sts}[name]),
        "BEDROCK": SimpleNamespace(create_advanced_prompt_optimization_job=create,
                                   get_advanced_prompt_optimization_job=get),
        "parameter_or_env": lambda *_: "workshop-test",
        "print": lambda *args, **_: printed.extend(args),
        "supported": Mock(return_value=True),
        "run_case": Mock(side_effect=AssertionError("Unexpected model call")),
    }
    try:
        for cell_id in ("high-apo-input", "high-apo-dataset", "high-apo-options", "high-apo-state",
                        "high-apo-assessment-tools"):
            execute(cell_id, ns)
        yield SimpleNamespace(ns=ns, calls=calls, uploads=uploads, printed=printed, displays=displays,
                              status=status, output=output, s3=s3, reads=reads, sts=sts)
    finally:
        client.close()


def completed(lab):
    submit(lab)
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


def test_section_defaults_ignore_environment_toggles_and_make_no_calls(lab, monkeypatch):
    for name in ("RUN_APO", "RUN_AGENTCORE_RECOMMENDATIONS", "RUN_AGENTCORE_EVALUATION",
                 "RUN_PROMPT_OPTIMIZATION"):
        monkeypatch.setenv(name, "1")
    notebook = json.loads(NOTEBOOK.read_text())
    start = next(i for i, c in enumerate(notebook["cells"]) if c["id"] == "high-06-09fbec88")
    end = next(i for i, c in enumerate(notebook["cells"]) if c["id"] == "high-apo-comparison-results")
    lab.ns["AGENTCORE"] = SimpleNamespace(
        evaluate=Mock(side_effect=AssertionError("Evaluation must be opted in")),
        start_recommendation=Mock(side_effect=AssertionError("Recommendations must be opted in")),
    )
    lab.ns["boto3"].client = Mock(side_effect=AssertionError("Default cells must not create clients"))
    for cell in notebook["cells"][start:end + 1]:
        if cell["cell_type"] == "code":
            execute(cell["id"], lab.ns)
    assert not lab.calls and not lab.uploads and not lab.reads
    lab.ns["boto3"].client.assert_not_called()
    lab.ns["run_case"].assert_not_called()
    lab.ns["AGENTCORE"].evaluate.assert_not_called()
    lab.ns["AGENTCORE"].start_recommendation.assert_not_called()
    assert not (lab.ns["REPO_ROOT"] / ".workshop-state").exists()


def test_preparation_does_not_submit_and_submission_rechecks_opt_in(lab):
    lab.ns["RUN_APO"] = True
    execute("high-apo-state", lab.ns)
    saved = json.loads(lab.ns["APO_STATE_PATH"].read_text())
    assert saved["clientToken"] and "jobArn" not in saved
    assert not lab.calls and not lab.uploads
    lab.ns["RUN_APO"] = False
    execute("high-apo-submit", lab.ns)
    assert not lab.calls and not lab.uploads
    lab.ns["RUN_APO"] = True
    execute("high-apo-submit", lab.ns)
    assert len(lab.calls) == len(lab.uploads) == 1
    assert lab.calls[0]["clientToken"] == saved["clientToken"]


@pytest.mark.parametrize("change", ["dataset", "config", "region", "bucket"])
def test_submission_rejects_edits_after_preparation_before_upload(lab, change):
    lab.ns["RUN_APO"] = True
    execute("high-apo-state", lab.ns)
    if change == "dataset":
        lab.ns["APO_INPUT_BODY"] = b"changed"
    elif change == "config":
        lab.ns["APO_MODEL_CONFIGS"][0]["inferenceConfig"]["maxTokens"] = 512
    elif change == "region":
        lab.ns["REGION"] = "us-west-2"
    else:
        lab.ns["APO_BUCKET"] = "other-bucket"
    with pytest.raises(ValueError):
        execute("high-apo-submit", lab.ns)
    assert not lab.calls and not lab.uploads


@pytest.mark.parametrize("missing", ["operation", "capability"])
def test_unsupported_apo_stops_before_state_or_upload(lab, missing):
    if missing == "operation":
        del lab.ns["BEDROCK"].create_advanced_prompt_optimization_job
    else:
        lab.ns["supported"].return_value = False
    with pytest.raises(RuntimeError if missing == "operation" else ValueError):
        submit(lab)
    assert not lab.calls and not lab.uploads
    assert not lab.ns["APO_STATE_PATH"].exists()


def test_resume_reuses_persisted_job_and_input(lab):
    submit(lab)
    state = json.loads(lab.ns["APO_STATE_PATH"].read_text())
    assert state["jobArn"] == JOB
    assert state["clientToken"] == lab.calls[0]["clientToken"]
    submit(lab)
    assert len(lab.calls) == len(lab.uploads) == 1
    assert lab.ns["APO_JOB_ARN"] == JOB


@pytest.mark.parametrize("scope", ["account", "bucket"])
def test_saved_experiment_cannot_be_reused_in_another_scope(lab, scope):
    submit(lab)
    if scope == "account":
        lab.sts.get_caller_identity = lambda: {"Account": "210987654321"}
    else:
        lab.ns["APO_BUCKET_NAME"] = "different-bucket"
    with pytest.raises(ValueError):
        submit(lab)
    assert len(lab.calls) == len(lab.uploads) == 1
    assert lab.ns["APO_JOB_ARN"] is None


def test_timed_out_create_reuses_saved_token(lab):
    create = lab.ns["BEDROCK"].create_advanced_prompt_optimization_job
    seen = []

    def timeout(**request):
        seen.append(request)
        raise TimeoutError("reply lost")

    lab.ns["BEDROCK"].create_advanced_prompt_optimization_job = timeout
    with pytest.raises(TimeoutError):
        submit(lab)
    lab.ns["BEDROCK"].create_advanced_prompt_optimization_job = create
    submit(lab)
    assert lab.calls[0] == seen[0]


def test_denied_submission_retains_retry_token_without_claiming_a_job(lab):
    create = lab.ns["BEDROCK"].create_advanced_prompt_optimization_job
    denial = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "Denied"}},
                        "CreateAdvancedPromptOptimizationJob")
    lab.ns["BEDROCK"].create_advanced_prompt_optimization_job = Mock(side_effect=denial)
    with pytest.raises(ClientError) as caught:
        submit(lab)
    assert caught.value is denial
    saved = json.loads(lab.ns["APO_STATE_PATH"].read_text())
    assert "jobArn" not in saved and lab.ns["APO_JOB_ARN"] is None
    lab.ns["BEDROCK"].create_advanced_prompt_optimization_job = create
    submit(lab)
    assert lab.calls[0]["clientToken"] == saved["clientToken"]


def test_changed_config_and_wrong_region_cannot_submit_another_job(lab):
    submit(lab)
    lab.ns["APO_MODEL_CONFIGS"][0]["inferenceConfig"]["maxTokens"] = 512
    with pytest.raises(ValueError, match="differs"):
        submit(lab)
    assert len(lab.calls) == 1
    lab.s3.get_bucket_location = lambda **_: {"LocationConstraint": "us-west-2"}
    with pytest.raises(ValueError, match="same Region"):
        submit(lab)
    assert len(lab.calls) == 1


def test_completed_result_is_selected_and_incomplete_job_clears_candidate(lab):
    completed(lab)
    execute("high-apo-result", lab.ns)
    assert lab.ns["apo_candidate_template"].endswith("{{ticket}}.")
    assert lab.ns["apo_comparison_baseline"] == lab.ns["APO_BASELINE_TEMPLATE"]
    assert lab.ns["apo_candidate_sha"] == hashlib.sha256(lab.ns["apo_candidate_template"].encode()).hexdigest()
    assert json.loads(lab.ns["apo_result_text"]) == lab.output
    assert lab.ns["template_results"] == [lab.output]
    lab.ns["APO_MODEL_CONFIGS"][0]["inferenceConfig"]["maxTokens"] = 512
    assert lab.ns["apo_comparison_config"]["inferenceConfig"]["maxTokens"] == 1024
    for status in ("InProgress", "Failed", "PartiallyCompleted", "Stopped"):
        lab.status["jobStatus"] = status
        execute("high-apo-result", lab.ns)
        assert lab.ns["apo_candidate_template"] is None
    assert len(lab.reads) == 1  # Incomplete jobs never fetch/accept partial S3 results.


def test_edited_dataset_does_not_relabel_old_job_results(lab):
    completed(lab)
    lab.ns["APO_INPUT_BODY"] = b'{"promptTemplate": "Always say billing"}\n'
    with pytest.raises(ValueError, match="dataset has changed"):
        execute("high-apo-result", lab.ns)
    assert lab.ns["apo_candidate_template"] is None


@pytest.mark.parametrize("problem", ["wrong_model", "wrong_template", "duplicate", "missing_variable",
                                    "extra_variable", "failure", "failed_status", "missing_status",
                                    "unknown_status", "changed_configuration", "wrong_job",
                                    "wrong_input_location", "wrong_output_location"])
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
    elif problem == "changed_configuration":
        lab.status["modelConfigurations"][0]["inferenceConfig"]["maxTokens"] = 512
    elif problem == "wrong_job":
        lab.status["jobArn"] = JOB + "-other"
    elif problem == "wrong_input_location":
        lab.status["inputConfig"]["s3Uri"] = "s3://other/input.jsonl"
    else:
        lab.status["outputConfig"]["s3Uri"] = "s3://other/output/"
    with pytest.raises(ValueError):
        execute("high-apo-result", lab.ns)
    assert lab.ns["apo_candidate_template"] is None


@pytest.mark.parametrize("change", ["job", "dataset", "config", "candidate", "baseline", "input_object"])
def test_comparison_rejects_a_candidate_retained_from_another_experiment(lab, change):
    completed(lab)
    execute("high-apo-result", lab.ns)
    if change == "job":
        lab.ns["APO_JOB_ARN"] = None
    elif change == "dataset":
        lab.ns["APO_INPUT_BODY"] = b"changed"
    elif change == "config":
        lab.ns["APO_MODEL_CONFIGS"][0]["inferenceConfig"]["maxTokens"] = 512
    elif change == "candidate":
        lab.ns["apo_candidate_template"] += " Changed locally."
    elif change == "baseline":
        lab.ns["apo_comparison_baseline"] = "Always return billing."
    else:
        lab.ns["APO_INPUT"]["promptTemplate"] = "Always return billing."
    with pytest.raises(ValueError, match="experiment changed"):
        execute("high-08-f889fbf9", lab.ns)
    lab.ns["run_case"].assert_not_called()


def test_completed_candidate_comparison_respects_the_single_run_apo_switch(lab):
    completed(lab)
    execute("high-apo-result", lab.ns)
    lab.ns["RUN_APO"] = False
    execute("high-08-f889fbf9", lab.ns)
    assert lab.ns["apo_candidate_template"]
    assert lab.ns["optimization_assessment"] == []
    lab.ns["run_case"].assert_not_called()


def test_apo_wait_polls_every_30_seconds_and_returns_only_at_completion(lab):
    submit(lab)
    original_get = lab.ns["BEDROCK"].get_advanced_prompt_optimization_job
    statuses = iter(["InProgress", "InProgress", "Completed"])
    sleeps = []

    def get(**request):
        lab.status["jobStatus"] = next(statuses)
        return original_get(**request)

    lab.ns["BEDROCK"].get_advanced_prompt_optimization_job = get
    lab.ns["time"] = SimpleNamespace(monotonic=lambda: len(sleeps) * 30, sleep=sleeps.append)
    execute("high-apo-status", lab.ns)
    assert sleeps == [30, 30]
    assert lab.ns["apo_status"] == "Completed"
    assert lab.ns["apo_job_status"]["jobArn"] == JOB
    assert len(lab.calls) == 1  # Waiting never creates another job.


@pytest.mark.parametrize("status", ["Failed", "PartiallyCompleted", "Stopped", "Deleting", "Unexpected"])
def test_apo_wait_ends_on_failure_without_accepting_a_candidate(lab, status):
    submit(lab)
    lab.status.update(jobStatus=status, failureMessage="Recorded failure detail")
    with pytest.raises(RuntimeError):
        execute("high-apo-status", lab.ns)
    assert lab.ns["apo_candidate_template"] is None
    assert lab.ns["apo_job_status"]["failureMessage"] == "Recorded failure detail"
    lab.ns["time"].sleep.assert_not_called()


def test_apo_wait_interruption_preserves_the_submitted_job(lab):
    submit(lab)
    lab.ns["time"] = SimpleNamespace(monotonic=lambda: 0, sleep=Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        execute("high-apo-status", lab.ns)
    assert lab.ns["APO_JOB_ARN"] == JOB
    assert json.loads(lab.ns["APO_STATE_PATH"].read_text())["jobArn"] == JOB
    assert len(lab.calls) == 1


def test_submission_displays_the_verified_apo_job_route(lab):
    submit(lab)
    assert lab.ns["apo_console_url"] == (
        "https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1"
        "#/prompt-optimization/example"
    )
    assert any("Open this APO job in the console" in value for value in lab.displays if isinstance(value, str))


def test_rough_baseline_does_not_change_training_labels_or_leak_held_out_tickets(lab):
    prompt = lab.ns["BASELINE_PROMPT"]
    assert len(prompt) > 800
    assert all(label in prompt for label in ("billing", "technical", "returns"))
    assert all(ticket not in prompt for ticket, _ in lab.ns["HELD_OUT"])
    assert lab.ns["APO_STATE_PATH"].name == "playbook-apo-opus5-v1.json"
    assert lab.ns["APO_INPUT"]["customLLMJConfig"]["customLLMJModelId"] == "anthropic.claude-opus-5"














def test_comparison_checks_uploaded_tickets_even_if_training_global_changes(lab):
    completed(lab)
    execute("high-apo-result", lab.ns)
    lab.ns.update(RUN_APO=True, HELD_OUT=[lab.ns["TRAINING"][0]], TRAINING=[])
    with pytest.raises(ValueError):
        execute("high-08-f889fbf9", lab.ns)
    lab.ns["run_case"].assert_not_called()


def test_held_out_comparison_uses_full_template_and_same_actual_settings(lab):
    calls = []
    ns = lab.ns
    ns["APO_MODEL_CONFIGS"][0]["inferenceConfig"]["maxTokens"] = 512
    ns["APO_MODEL_CONFIGS"][0]["additionalModelRequestFields"] = {"output_config": {"effort": "medium"}}
    completed(lab)
    execute("high-apo-result", ns)
    expected_by_ticket = dict(ns["HELD_OUT"])

    def run_case(model_id, query, *, label, **controls):
        calls.append((model_id, query, controls))
        answer = next(expected for ticket, expected in expected_by_ticket.items() if ticket in query)
        return {"label": label, "text": answer, "stop_reason": "end_turn",
                "cost_usd": 0.001, "latency_ms": 100, "span": None}

    ns["run_case"] = run_case
    execute("high-08-f889fbf9", ns)
    execute("high-apo-comparison-results", ns)
    assert len(calls) == 12
    assert all(r["correct"] for r in ns["optimization_assessment"])
    for model, query, controls in calls:
        assert model == ns["APO_MODEL_ID"]
        assert "{{ticket}}" not in query and "system" not in controls
        assert controls["max_tokens"] == 512
        assert controls["additional_model_request_fields"] == ns["APO_MODEL_CONFIGS"][0]["additionalModelRequestFields"]
    report = lab.displays[-1].data
    assert list(report["Prompt"]) == ["baseline", "APO candidate"]
    assert list(report["Passed"]) == ["6/6", "6/6"]
    assert list(report["USD / solved ticket"]) == pytest.approx([0.001, 0.001])


def test_zero_success_has_no_cost_per_correct_ticket(lab):
    completed(lab)
    execute("high-apo-result", lab.ns)
    lab.ns.update(
        RUN_APO=True,
        run_case=lambda *args, **kwargs: {
            "label": kwargs["label"], "text": "wrong", "stop_reason": "end_turn",
            "cost_usd": 0.01, "latency_ms": 100, "span": None,
        },
    )
    execute("high-08-f889fbf9", lab.ns)
    execute("high-apo-comparison-results", lab.ns)
    report = lab.displays[-1].data
    assert len(report) == 2
    assert list(report["Passed"]) == ["0/6", "0/6"]
    assert report["USD / solved ticket"].isna().all()
    assert list(report["Model cost (USD)"]) == pytest.approx([0.06, 0.06])






def test_setup_keeps_native_converse_and_evaluation_outside_assessment():
    setup = ast.parse(cell_source("high-setup-87841c4c"))
    calls = {ast.unparse(n.func) for n in ast.walk(setup) if isinstance(n, ast.Call)}
    assert "RUNTIME.converse" in calls and "traced_converse" not in calls
    assessment = cell_source("high-apo-assessment-tools") + cell_source("high-08-f889fbf9")
    assert "quick_eval" not in assessment and "AGENTCORE.evaluate" not in assessment
