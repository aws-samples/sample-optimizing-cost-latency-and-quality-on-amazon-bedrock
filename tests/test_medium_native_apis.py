"""Execute MEDIUM notebook paths offline against pinned SDKs and synthetic responses."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import io
import json
import sys
import time
import uuid
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import boto3
import pytest
from botocore.config import Config
from botocore.endpoint import Endpoint
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from botocore.validate import validate_parameters

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "02-optimization-playbook/02-medium-effort.ipynb"
CELLS = {
    cell["id"]: "".join(cell["source"])
    for cell in json.loads(NOTEBOOK.read_text())["cells"]
}
MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
KB_ID = "VECTOR1234"
MANAGED_ID = "MANAGED123"
QUERY = "What is the warranty on a refurbished laptop?"
PASSAGE = {
    "content": {"text": "Refurbished laptops have a 90-day warranty."},
    "location": {"type": "S3", "s3Location": {"uri": "s3://fixture/policy.txt"}},
    "metadata": {"product_line": "laptops", "doc_type": "policy"},
    "score": 0.75,
}


@pytest.fixture(autouse=True)
def no_aws_transport(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Notebook tests must never reach an AWS endpoint")

    monkeypatch.setattr(Endpoint, "make_request", forbidden)


@pytest.fixture
def sdk_session():
    return boto3.Session(
        aws_access_key_id="offline", aws_secret_access_key="offline",
        region_name="us-east-1",
    )


def execute(cell_id, namespace):
    exec(compile(CELLS[cell_id], f"{NOTEBOOK}:{cell_id}", "exec"), namespace)


def retrieval_request(kb_id=KB_ID, *, kind="vector", count=3, **extra):
    return {
        "knowledgeBaseId": kb_id, "retrievalQuery": {"text": QUERY},
        "retrievalConfiguration": {
            f"{kind}SearchConfiguration": {"numberOfResults": count, **extra},
        },
    }


def cohere_preflight_fixture(availability=None):
    order, bodies = [], []
    availability = availability or {
        "modelId": "cohere.rerank-v3-5:0",
        "agreementAvailability": {"status": "AVAILABLE"},
        "authorizationStatus": "AUTHORIZED",
        "entitlementAvailability": "AVAILABLE",
        "regionAvailability": "AVAILABLE",
    }

    def invoke(**request):
        order.append("invoke")
        body = io.BytesIO(json.dumps({
            "results": [{"index": 0, "relevance_score": 0.98},
                        {"index": 1, "relevance_score": 0.12}],
        }).encode())
        bodies.append(body)
        return {"body": body, "contentType": "application/json"}

    def inspect_availability(**request):
        order.append("availability")
        return copy.deepcopy(availability)

    return {
        "RUNTIME": SimpleNamespace(invoke_model=Mock(side_effect=invoke)),
        "BEDROCK": SimpleNamespace(get_foundation_model_availability=Mock(side_effect=inspect_availability)),
        "_cohere_order": order, "_cohere_bodies": bodies,
    }


def test_vector_filter_rerank_and_managed_requests_keep_query_and_sources(sdk_session):
    client = sdk_session.client("bedrock-agent-runtime")
    metadata_filter = {
        "andAll": [
            {"equals": {"key": "doc_type", "value": "policy"}},
            {"in": {"key": "product_line", "value": ["laptops", "all"]}},
        ],
    }
    reranking = {
        "type": "BEDROCK_RERANKING_MODEL",
        "bedrockRerankingConfiguration": {
            "numberOfRerankedResults": 3,
            "modelConfiguration": {
                "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0",
            },
        },
    }
    requests = [retrieval_request(count=k, filter=metadata_filter) for k in (5, 4, 3)]
    requests += [
        retrieval_request(count=5),
        retrieval_request(count=5, filter=metadata_filter),
        retrieval_request(count=10, filter=metadata_filter),
        retrieval_request(count=10, filter=metadata_filter, rerankingConfiguration=reranking),
        retrieval_request(MANAGED_ID, kind="managed"),
    ]
    control = SimpleNamespace(get_knowledge_base=Mock(return_value={
        "knowledgeBase": {"knowledgeBaseConfiguration": {"type": "MANAGED"}},
    }))
    printed = []
    namespace = {
        **cohere_preflight_fixture(),
        "KB_ID": KB_ID, "QUERY": QUERY, "REGION": "us-east-1",
        "KB_RUNTIME": client, "KB_CONTROL": control,
        "os": SimpleNamespace(environ={
            "RUN_RERANK": "1", "RUN_MANAGED_KB": "1", "MANAGED_KB_ID": MANAGED_ID,
        }),
        "print": printed.append,
    }
    with Stubber(client) as stubber:
        for request in requests:
            stubber.add_response("retrieve", {"retrievalResults": [PASSAGE]}, request)
        execute("medium-12-e12b8aef", namespace)
        stubber.assert_no_pending_responses()
    assert namespace["hits"] == [PASSAGE]
    assert printed[0]["sources"] == [PASSAGE["location"]]
    assert printed[-1] == PASSAGE
    assert namespace["retrieval_candidates"] == [PASSAGE]
    assert namespace["retrieval_only_hits"] == [PASSAGE]
    assert namespace["reranked_hits"] == [PASSAGE]
    control.get_knowledge_base.assert_called_once_with(knowledgeBaseId=MANAGED_ID)


@pytest.mark.parametrize("count", [0, 1, 2, 5])
def test_rerank_comparison_bounds_context_without_padding_small_fixture(count):
    hits = [{**PASSAGE, "content": {"text": f"Policy passage {i}"}} for i in range(count)]
    retrieval = Mock(side_effect=lambda **kwargs: {
        "retrievalResults": hits[:3] if "rerankingConfiguration" in
        kwargs["retrievalConfiguration"]["vectorSearchConfiguration"] else hits,
    })
    namespace = {
        **cohere_preflight_fixture(),
        "KB_ID": KB_ID, "QUERY": QUERY, "REGION": "us-east-1",
        "KB_RUNTIME": SimpleNamespace(retrieve=retrieval),
        "os": SimpleNamespace(environ={"RUN_RERANK": "1"}),
        "print": Mock(),
    }
    execute("medium-12-e12b8aef", namespace)
    baseline, reranked = [call.kwargs for call in retrieval.call_args_list[-2:]]
    reranked_config = copy.deepcopy(reranked["retrievalConfiguration"]["vectorSearchConfiguration"])
    reranked_config.pop("rerankingConfiguration")
    assert baseline["retrievalConfiguration"]["vectorSearchConfiguration"] == reranked_config
    assert baseline["retrievalQuery"] == reranked["retrievalQuery"] == {"text": QUERY}
    assert baseline["knowledgeBaseId"] == reranked["knowledgeBaseId"] == KB_ID
    assert namespace["retrieval_candidates"] == hits
    assert namespace["retrieval_only_hits"] == namespace["reranked_hits"] == hits[:3]


def rerank_error_fixture(code, message):
    error = ClientError({
        "Error": {"Code": code, "Message": message},
        "ResponseMetadata": {"HTTPStatusCode": 400, "RequestId": "synthetic-rerank-error"},
    }, "Retrieve")

    def retrieve(**request):
        config = request["retrievalConfiguration"].get("vectorSearchConfiguration", {})
        if "rerankingConfiguration" in config:
            raise error
        return {"retrievalResults": [copy.deepcopy(PASSAGE)]}

    client = SimpleNamespace(retrieve=Mock(side_effect=retrieve))
    control = SimpleNamespace(get_knowledge_base=Mock(return_value={
        "knowledgeBase": {"knowledgeBaseConfiguration": {"type": "MANAGED"}},
    }))
    namespace = {
        **cohere_preflight_fixture(),
        "KB_ID": KB_ID, "QUERY": QUERY, "REGION": "us-east-1",
        "KB_RUNTIME": client, "KB_CONTROL": control,
        "os": SimpleNamespace(environ={
            "RUN_RERANK": "1", "RUN_MANAGED_KB": "1", "MANAGED_KB_ID": MANAGED_ID,
        }),
        "print": Mock(),
        # A previous successful cell must not leave a stale reranked result.
        "reranked_hits": [PASSAGE], "rerank_status": "available",
    }
    return namespace, error


def test_cohere_direct_preflight_consumes_body_and_checks_availability_before_kb_rerank(sdk_session):
    namespace, _ = rerank_error_fixture("unused", "unused")
    order = namespace["_cohere_order"]

    def retrieve(**request):
        if "rerankingConfiguration" in request["retrievalConfiguration"].get("vectorSearchConfiguration", {}):
            order.append("kb-rerank")
        return {"retrievalResults": [PASSAGE]}

    namespace["KB_RUNTIME"].retrieve.side_effect = retrieve
    execute("medium-12-e12b8aef", namespace)
    request = namespace["RUNTIME"].invoke_model.call_args.kwargs
    namespace["RUNTIME"].invoke_model.assert_called_once()
    assert request["modelId"] == "cohere.rerank-v3-5:0"
    assert request["contentType"] == request["accept"] == "application/json"
    assert json.loads(request["body"]) == {
        "api_version": 2, "query": QUERY,
        "documents": ["Refurbished products include a 90-day warranty.",
                      "Returns require original packaging."],
        "top_n": 2,
    }
    client = sdk_session.client("bedrock-runtime")
    control = sdk_session.client("bedrock")
    try:
        validate_parameters(request, client.meta.service_model.operation_model("InvokeModel").input_shape)
        validate_parameters(
            namespace["BEDROCK"].get_foundation_model_availability.call_args.kwargs,
            control.meta.service_model.operation_model("GetFoundationModelAvailability").input_shape,
        )
    finally:
        client.close()
        control.close()
    assert order == ["invoke", "availability", "kb-rerank"]
    assert all(body.closed for body in namespace["_cohere_bodies"])
    assert namespace["cohere_preflight"]["results"][0]["relevance_score"] == 0.98
    assert namespace["rerank_status"] == "available"
    assert any(call.args[0].get("cohere_preflight_scores") for call in namespace["print"].call_args_list
               if call.args and isinstance(call.args[0], dict))


@pytest.mark.parametrize("unready", [
    {"agreementAvailability": {"status": "NOT_AVAILABLE"}},
    {"authorizationStatus": "NOT_AUTHORIZED"},
    {"entitlementAvailability": "NOT_AVAILABLE"},
    {"regionAvailability": "NOT_AVAILABLE"},
])
def test_unready_cohere_does_not_claim_readiness_or_call_kb_rerank_but_managed_continues(unready):
    namespace, _ = rerank_error_fixture("unused", "unused")
    availability = {
        "agreementAvailability": {"status": "AVAILABLE"}, "authorizationStatus": "AUTHORIZED",
        "entitlementAvailability": "AVAILABLE", "regionAvailability": "AVAILABLE", **unready,
    }
    namespace.update(cohere_preflight_fixture(availability))
    execute("medium-12-e12b8aef", namespace)
    assert namespace["rerank_status"] == "activation_pending"
    assert namespace["reranked_hits"] is None
    assert namespace["retrieval_only_hits"] == [PASSAGE]
    assert not any("rerankingConfiguration" in call.kwargs["retrievalConfiguration"].get(
        "vectorSearchConfiguration", {}) for call in namespace["KB_RUNTIME"].retrieve.call_args_list)
    namespace["KB_CONTROL"].get_knowledge_base.assert_called_once_with(knowledgeBaseId=MANAGED_ID)
    assert any("rerun this cell" in str(call.args) for call in namespace["print"].call_args_list)


def test_non_cohere_override_never_receives_cohere_payload_or_availability_probe():
    namespace, _ = rerank_error_fixture("unused", "unused")
    namespace["os"].environ["RERANK_MODEL_ARN"] = (
        "arn:aws:bedrock:us-east-1::foundation-model/example.other-reranker:0"
    )
    namespace["KB_RUNTIME"].retrieve.side_effect = lambda **kwargs: {"retrievalResults": [PASSAGE]}
    execute("medium-12-e12b8aef", namespace)
    namespace["RUNTIME"].invoke_model.assert_not_called()
    namespace["BEDROCK"].get_foundation_model_availability.assert_not_called()
    assert namespace["rerank_status"] == "available"


@pytest.mark.parametrize("failure", ["read", "invalid-json", "no-scores"])
def test_cohere_body_is_closed_on_failure_without_claiming_ready(failure):
    namespace, _ = rerank_error_fixture("unused", "unused")
    if failure == "read":
        body = Mock(read=Mock(side_effect=OSError("Synthetic read failure")))
        expected = OSError
    else:
        body = io.BytesIO(b"not-json" if failure == "invalid-json" else b'{"results":[]}')
        expected = json.JSONDecodeError if failure == "invalid-json" else RuntimeError
    namespace["RUNTIME"].invoke_model.side_effect = None
    namespace["RUNTIME"].invoke_model.return_value = {"body": body}
    with pytest.raises(expected):
        execute("medium-12-e12b8aef", namespace)
    if failure == "read":
        body.close.assert_called_once()
    else:
        assert body.closed
    namespace["BEDROCK"].get_foundation_model_availability.assert_not_called()
    assert namespace["rerank_status"] != "available"


@pytest.mark.parametrize("code,message", [
    ("ValidationException", "BedrockRuntime returned Status Code: 403; "
     "not authorized for aws-marketplace:Subscribe and aws-marketplace:ViewSubscriptions."),
    ("ValidationException", "BedrockRuntime AccessDeniedException: aws-marketplace:ViewSubscriptions required."),
    ("ValidationException", "BedrockRuntime 403: AWS Marketplace Subscribe/ViewSubscriptions permissions missing."),
    ("AccessDeniedException", "Missing permission aws-marketplace:Subscribe."),
])
def test_known_rerank_access_failure_preserves_control_error_and_continues_managed(code, message):
    namespace, error = rerank_error_fixture(code, message)
    execute("medium-12-e12b8aef", namespace)
    assert namespace["rerank_status"] == "unavailable"
    assert namespace["reranked_hits"] is None
    assert namespace["rerank_error"] == error.response
    assert namespace["retrieval_candidates"] == namespace["retrieval_only_hits"] == [PASSAGE]
    calls = namespace["KB_RUNTIME"].retrieve.call_args_list
    rerank_calls = [
        call.kwargs for call in calls
        if "rerankingConfiguration" in call.kwargs["retrievalConfiguration"].get("vectorSearchConfiguration", {})
    ]
    assert len(rerank_calls) == 1  # No fallback model/Region or retry.
    assert rerank_calls[0]["retrievalConfiguration"]["vectorSearchConfiguration"][
        "rerankingConfiguration"]["bedrockRerankingConfiguration"]["modelConfiguration"]["modelArn"] == (
            "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"
        )
    assert calls[-1].kwargs == retrieval_request(MANAGED_ID, kind="managed")
    namespace["KB_CONTROL"].get_knowledge_base.assert_called_once_with(knowledgeBaseId=MANAGED_ID)
    reports = [call.args[0] for call in namespace["print"].call_args_list
               if call.args and isinstance(call.args[0], dict)]
    unavailable = next(row for row in reports if row.get("rerank_status") == "unavailable")
    assert unavailable["raw_error"] == error.response
    assert unavailable["error"] == str(error)
    assert unavailable["rerank_model_arn"] == namespace["RERANK_MODEL_ARN"]
    assert not any(row.get("variant") == "reranked" for row in reports)
    assert any(row.get("variant") == "retrieval-only" and row["hits"] == [PASSAGE] for row in reports)


@pytest.mark.parametrize("code,message", [
    ("ValidationException", "Invalid metadata filter."),
    ("ValidationException", "Reranking model is unsupported in this Region."),
    ("ValidationException", "BedrockRuntime 403: model access denied."),
    ("AccessDeniedException", "Not authorized to perform bedrock:Retrieve."),
    ("ValidationException", "Invalid filter value aws-marketplace:Subscribe."),
    ("ThrottlingException", "aws-marketplace:Subscribe returned 403."),
])
def test_other_rerank_errors_propagate_without_becoming_unavailable(code, message):
    namespace, error = rerank_error_fixture(code, message)
    with pytest.raises(ClientError) as raised:
        execute("medium-12-e12b8aef", namespace)
    assert raised.value is error
    assert namespace["rerank_status"] != "unavailable"
    assert namespace["reranked_hits"] is None
    namespace["KB_CONTROL"].get_knowledge_base.assert_not_called()


def test_workshop_filter_selects_warranty_from_actual_provisioned_documents():
    import yaml

    # Parse the fixture literal, never execute its provisioning handler.
    template = yaml.load(
        (ROOT / "03-developer-journey/prerequisite/infrastructure.yaml").read_text(),
        Loader=yaml.BaseLoader,
    )
    code = template["Resources"]["KnowledgeBaseDocumentFunction"]["Properties"]["Code"]["ZipFile"]
    assignment = next(
        node for node in ast.parse(code).body
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DATASETS" for target in node.targets
        )
    )
    documents = ast.literal_eval(assignment.value)["playbook"]
    assert len(documents) == 5
    assert all(set(metadata) == {"doc_type", "product_line"} for _, metadata in documents.values())
    namespace = {
        "KB_ID": KB_ID, "QUERY": QUERY,
        "KB_RUNTIME": SimpleNamespace(retrieve=Mock(return_value={"retrievalResults": []})),
        "os": SimpleNamespace(environ={}), "print": Mock(),
    }
    execute("medium-12-e12b8aef", namespace)

    def matches(metadata, rule):
        if "andAll" in rule:
            return all(matches(metadata, child) for child in rule["andAll"])
        operation, condition = next(iter(rule.items()))
        actual = metadata.get(condition["key"])
        if operation == "equals":
            return actual == condition["value"]
        assert operation == "in"
        return actual in condition["value"]

    eligible = {
        name: text for name, (text, metadata) in documents.items()
        if matches(metadata, namespace["policy_filter"])
    }
    assert set(eligible) == {"returns-faq.txt", "warranty-faq.txt"}
    assert "90-day warranty" in eligible["warranty-faq.txt"]


def test_managed_type_check_stops_before_retrieval():
    retrieve = Mock(side_effect=AssertionError("Wrong KB type must not be queried"))
    namespace = {
        "KB_ID": "", "QUERY": QUERY,
        "KB_RUNTIME": SimpleNamespace(retrieve=retrieve),
        "KB_CONTROL": SimpleNamespace(get_knowledge_base=Mock(return_value={
            "knowledgeBase": {"knowledgeBaseConfiguration": {"type": "VECTOR"}},
        })),
        "os": SimpleNamespace(environ={"RUN_MANAGED_KB": "1", "MANAGED_KB_ID": MANAGED_ID}),
        "print": Mock(),
    }
    with pytest.raises(ValueError, match="MANAGED"):
        execute("medium-12-e12b8aef", namespace)
    retrieve.assert_not_called()


def test_optional_rag_is_disabled_without_configuration_or_sdk_calls():
    namespace = {"os": SimpleNamespace(environ={}), "print": Mock()}
    execute("medium-strands-rag-integration", namespace)
    assert namespace["RUN_STRANDS_RAG"] is False
    assert "rag_agent" not in namespace


def test_optional_rag_requires_existing_kb_before_constructing_agent():
    namespace = {
        "os": SimpleNamespace(environ={"RUN_STRANDS_RAG": "1"}),
        "KB_ID": "", "Agent": Mock(),
    }
    with pytest.raises(ValueError, match="existing vector KB"):
        execute("medium-strands-rag-integration", namespace)
    namespace["Agent"].assert_not_called()


def converse_response(*, tool_ids=()):
    content = [
        {"toolUse": {"toolUseId": tool_id, "name": "retrieve_workshop_policy", "input": {}}}
        for tool_id in tool_ids
    ] or [{"text": "90 days; source: s3://fixture/policy.txt"}]
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": "tool_use" if tool_ids else "end_turn",
        "usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120},
        "metrics": {"latencyMs": 1},
    }


@pytest.mark.parametrize("repeat_tools", [False, True])
def test_real_strands_tool_loop_respects_retrieval_and_model_budgets(sdk_session, repeat_tools, monkeypatch):
    from strands import Agent
    from strands.models import BedrockModel

    kb_runtime = sdk_session.client("bedrock-agent-runtime")
    printed, models = [], []
    config = Config(
        connect_timeout=10, read_timeout=120,
        retries={"mode": "standard", "total_max_attempts": 3},
    )
    with ExitStack() as stack:
        kb_stubber = stack.enter_context(Stubber(kb_runtime))
        kb_stubber.add_response("retrieve", {"retrievalResults": [PASSAGE]}, retrieval_request())

        def model_factory(**kwargs):
            model = BedrockModel(**kwargs)
            models.append(model)
            stubber = stack.enter_context(Stubber(model.client))
            # Duplicate calls in the first turn must not issue duplicate KB requests.
            stubber.add_response("converse", converse_response(tool_ids=("first", "duplicate")))
            stubber.add_response(
                "converse", converse_response(tool_ids=("again",) if repeat_tools else ()),
            )
            return model

        # The self-contained notebook cell imports these dependencies directly.
        # Patch those public imports while retaining the real Strands agent loop.
        monkeypatch.setattr("strands.models.BedrockModel", model_factory)
        monkeypatch.setattr("workshop_utils.pacing.paced_boto3_session", lambda **kwargs: sdk_session)
        namespace = {
            "os": SimpleNamespace(environ={"RUN_STRANDS_RAG": "1"}),
            "KB_ID": KB_ID, "KB_RUNTIME": kb_runtime, "QUERY": QUERY,
            "SMALL": MODEL_ID, "REGION": "us-east-1", "SDK_CONFIG": config,
            "supported": lambda *args: True,
            "Agent": Agent, "BedrockModel": model_factory,
            "paced_boto3_session": lambda **kwargs: sdk_session,
            "print": printed.append,
        }
        execute("medium-strands-rag-integration", namespace)
        kb_stubber.assert_no_pending_responses()

    assert namespace["rag_calls"] == {"model_calls": 2, "retrieval_calls": 1}
    assert namespace["rag_retrieval_result"]["retrievalResults"] == [PASSAGE]
    assert printed[-1]["sources"] == [PASSAGE["location"]]
    assert models[0].config["max_tokens"] == 512
    assert models[0].config["streaming"] is False
    assert models[0].client.meta.config.retries["total_max_attempts"] == 3
    if repeat_tools:
        assert printed[0]["stop_reason"] == "limit_turns"
        assert printed[0]["answer"] is None
    else:
        assert printed[0]["stop_reason"] == "end_turn"
        assert "90 days" in printed[0]["answer"]
        assert printed[0]["agent_raw_usage"]["inputTokens"] == 200


def test_managed_guardrail_request_explicitly_anonymizes_email_in_both_directions(sdk_session):
    # Execute the authored call only: no readiness polling, model calls or cleanup.
    tree = ast.parse(CELLS["medium-09-b459042d"])
    creation = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "create_guardrail")
    control = SimpleNamespace(create_guardrail=Mock(return_value={"guardrailId": "synthetic"}))
    eval(compile(ast.Expression(body=creation), str(NOTEBOOK), "eval"),
         {"BEDROCK": control, "uuid": uuid})
    control.create_guardrail.assert_called_once()
    request = control.create_guardrail.call_args.kwargs
    client = sdk_session.client("bedrock")
    try:
        validate_parameters(
            request, client.meta.service_model.operation_model("CreateGuardrail").input_shape,
        )
    finally:
        client.close()
    email, = [entry for entry in request["sensitiveInformationPolicyConfig"]["piiEntitiesConfig"]
              if entry["type"] == "EMAIL"]
    assert email["action"] == "ANONYMIZE"
    assert email["inputEnabled"] is True and email["outputEnabled"] is True
    assert email["inputAction"] == email["outputAction"] == "ANONYMIZE"


@pytest.mark.parametrize("fail_model", [False, True])
def test_guardrail_trial_displays_native_config_trace_and_always_cleans_up(fail_model):
    trace = {"inputAssessment": {"fixture": {"sensitiveInformationPolicy": {"piiEntities": []}}}}
    control = SimpleNamespace(
        create_guardrail=Mock(return_value={"guardrailId": "fixture-guardrail"}),
        get_guardrail=Mock(return_value={"status": "READY"}),
        delete_guardrail=Mock(),
    )
    runtime = SimpleNamespace(apply_guardrail=Mock(return_value={
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "My email is {EMAIL}."}],
        "usage": {"sensitiveInformationPolicyUnits": 1},
    }))
    printed = []

    def run_case(model_id, text, **controls):
        if fail_model:
            raise RuntimeError("Synthetic model failure")
        return {
            "request": {"guardrailConfig": controls["guardrail_config"]},
            "response": {"stopReason": "end_turn", "trace": {"guardrail": trace}},
        }

    namespace = {
        "os": SimpleNamespace(environ={"RUN_MANAGED_GUARDRAIL": "1"}),
        "SMALL": MODEL_ID, "supported": lambda *args: True, "uuid": uuid,
        "BEDROCK": control, "RUNTIME": runtime, "run_case": run_case,
        "poll_status": lambda fetch, **kwargs: fetch(),
        "show": Mock(), "print": printed.append,
    }
    if fail_model:
        with pytest.raises(RuntimeError, match="Synthetic model failure"):
            execute("medium-09-b459042d", namespace)
    else:
        execute("medium-09-b459042d", namespace)
        inline = [item for item in printed if "guardrailConfig" in item]
        assert len(inline) == 2
        assert inline[0]["guardrailConfig"] == {
            "guardrailIdentifier": "fixture-guardrail", "guardrailVersion": "DRAFT", "trace": "enabled",
        }
        assert all(item["guardrail_trace"] == trace for item in inline)
        assert runtime.apply_guardrail.call_args.kwargs["content"] == [
            {"text": {"text": "My email is alex@example.com."}},
        ]
    control.delete_guardrail.assert_called_once_with(guardrailIdentifier="fixture-guardrail")


def test_context_preview_exposes_actual_request_and_character_trigger_without_calls():
    from workshop_utils.bedrock import build_converse_request
    from workshop_utils.models import resolve_model

    spec = importlib.util.spec_from_file_location(
        "medium_preview_context", ROOT / "02-optimization-playbook/utils/context_management.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        namespace = {
            "os": SimpleNamespace(environ={}), "SMALL": MODEL_ID,
            "json": json, "REGION": "us-east-1",
            "build_converse_request": build_converse_request, "resolve_model": resolve_model,
            **{name: getattr(module, name) for name in (
                "Exchange", "RollingContext", "answer_checks", "call_totals",
                "SUMMARY_SCHEMA", "ANSWER_SCHEMA",
            )},
            "print": Mock(),
        }
        tree = ast.parse(CELLS["medium-context-fixture"])
        tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)]
        exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
    finally:
        del sys.modules[spec.name]
    preview = namespace["print"].call_args.args[0]
    state = namespace["preview_state"]
    assert namespace["context_request_preview"]["messages"] == state.messages(namespace["CHECKPOINTS"][4])
    assert namespace["context_request_preview"]["outputConfig"] == namespace["ANSWER_OUTPUT_CONFIG"]
    assert namespace["summary_request_preview"]["outputConfig"] == namespace["SUMMARY_OUTPUT_CONFIG"]
    assert preview["context_characters"] == state.context_chars(
        namespace["CHECKPOINTS"][4], namespace["ANSWER_SYSTEM"],
    )
    assert preview["would_compact"] == (
        preview["older_exchanges_eligible"] and preview["size_trigger_reached"]
    )
    assert len(preview["first_user_text_preview"]) <= 180
    assert len(json.dumps(preview)) < 1500
    assert namespace["RUN_CONTEXT"] is False


@pytest.mark.parametrize("retries", [0, 1])
def test_run_case_records_native_response_trace_and_retry_cost_uncertainty(retries):
    from workshop_utils.bedrock import build_converse_request, normalize_usage
    from workshop_utils.models import resolve_model
    from workshop_utils.pricing import (
        AmbiguousCacheUsageError,
        InferredPriceError,
        UnknownPriceError,
        calculate_cost,
    )

    response = converse_response()
    response["ResponseMetadata"] = {"RetryAttempts": retries}
    runtime = SimpleNamespace(converse=Mock(return_value=copy.deepcopy(response)))
    observed_requests = []
    trace = {"session_id": "session-fixture", "trace_id": "a" * 32, "span_id": "b" * 16}

    @contextmanager
    def observation(request, *, session_id):
        observed_requests.append((request, session_id))
        yield lambda result: {**result, "_trace": trace}

    namespace = {
        "REGION": "us-east-1", "RUNTIME": runtime, "RUN_ID": "offline", "RUN_ROWS": [],
        "resolve_model": resolve_model, "build_converse_request": build_converse_request,
        "normalize_usage": normalize_usage, "calculate_cost": calculate_cost,
        "AmbiguousCacheUsageError": AmbiguousCacheUsageError,
        "InferredPriceError": InferredPriceError, "UnknownPriceError": UnknownPriceError,
        "converse_observation": observation, "response_span": Mock(return_value={"span": "fixture"}),
        "json": json, "hashlib": hashlib, "time": time, "uuid": uuid,
        "datetime": datetime, "UTC": UTC, "boto3": SimpleNamespace(__version__="offline"),
    }
    function = next(
        node for node in ast.parse(CELLS["medium-setup-a4d9ea22"]).body
        if isinstance(node, ast.FunctionDef) and node.name == "run_case"
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    row = namespace["run_case"](MODEL_ID, QUERY, label="native", max_tokens=512)
    runtime.converse.assert_called_once_with(**row["request"])
    assert observed_requests[0][0] is row["request"]
    assert observed_requests[0][1]
    assert row["response"]["_trace"] == trace
    assert namespace["response_span"].call_args.kwargs["trace_id"] == trace["trace_id"]
    assert row["returned_response_cost_usd"] > 0
    assert row["usage_complete"] is (retries == 0)
    assert row["request_sha256"] == hashlib.sha256(
        json.dumps(row["request"], sort_keys=True).encode(),
    ).hexdigest()
    if retries:
        assert row["cost_usd"] is None
        assert "earlier attempts is unknown" in row["price_note"]
    else:
        assert row["cost_usd"] == row["returned_response_cost_usd"]


def test_native_operations_stay_visible_and_batch_has_no_executable_code():
    expected = {
        "medium-11-4840f476": {"get_knowledge_base", "get_data_source", "list_ingestion_jobs"},
        "medium-07-ae99d687": {"invoke_guardrail_checks"},
        "medium-09-b459042d": {"create_guardrail", "get_guardrail", "apply_guardrail", "delete_guardrail"},
        "medium-15-83e801ba": {"get_memory", "create_event", "list_events"},
        "medium-16-2cf6a06a": {"retrieve_memory_records"},
        "medium-memory-scope-corrections": {"retrieve_memory_records"},
    }
    for cell_id, methods in expected.items():
        calls = {
            node.func.attr for node in ast.walk(ast.parse(CELLS[cell_id]))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert methods <= calls
    assert ast.parse(CELLS["medium-19-7f7febf8"]).body == []
    assert "retrieve_passages" not in "\n".join(CELLS.values())
    assert "quick_eval" not in CELLS["medium-setup-a4d9ea22"]
    assert "attach_boto3(RUNTIME, get_workshop_pacer())" in CELLS["medium-setup-a4d9ea22"]
