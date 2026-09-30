"""Execute MEDIUM notebook paths offline against pinned SDKs and synthetic responses."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
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
from botocore.stub import Stubber

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
    requests = [retrieval_request(count=k) for k in (5, 4, 3)]
    requests += [
        retrieval_request(count=5),
        retrieval_request(count=5, filter=metadata_filter),
        retrieval_request(count=10, rerankingConfiguration=reranking),
        retrieval_request(MANAGED_ID, kind="managed"),
    ]
    control = SimpleNamespace(get_knowledge_base=Mock(return_value={
        "knowledgeBase": {"knowledgeBaseConfiguration": {"type": "MANAGED"}},
    }))
    printed = []
    namespace = {
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
    control.get_knowledge_base.assert_called_once_with(knowledgeBaseId=MANAGED_ID)


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
    spec = importlib.util.spec_from_file_location(
        "medium_preview_context", ROOT / "02-optimization-playbook/utils/context_management.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        namespace = {
            "os": SimpleNamespace(environ={}), "SMALL": MODEL_ID,
            **{name: getattr(module, name) for name in (
                "Exchange", "RollingContext", "answer_checks", "call_totals",
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
