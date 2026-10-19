"""Offline contracts for HIGH's instruction audit and read-only MCP execution."""

from __future__ import annotations

import ast
import asyncio
import copy
import importlib.util
import json
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from botocore.exceptions import ClientError
from mcp.types import CallToolResult, ListToolsResult

ROOT = Path(__file__).resolve().parents[1]
SECTION = ROOT / "02-optimization-playbook"


def load_helper(name):
    spec = importlib.util.spec_from_file_location(f"high_test_{name}", SECTION / "utils" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_helper("harness_audit")
gateway = load_helper("gateway")
TARGET = "SupportTools"
CATALOG = gateway.discovery_catalog(TARGET)


def notebook_cell(cell_id):
    notebook = json.loads((SECTION / "03-high-effort.ipynb").read_text())
    return "".join(next(cell["source"] for cell in notebook["cells"] if cell["id"] == cell_id))


def execute_cell(cell_id, namespace, *, source=None):
    """Execute actual split cells, including top-level await, with isolated local imports."""
    # The Journey also has a utils package; resolve these imports to the HIGH helpers.
    modules = {"utils": ModuleType("utils"), "utils.gateway": gateway, "utils.harness_audit": audit}
    with patch.dict(sys.modules, modules):
        result = eval(compile(
            notebook_cell(cell_id) if source is None else source,
            cell_id, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        ), namespace)
        if asyncio.iscoroutine(result):
            asyncio.run(result)


def proposal(name="get_return_policy", args=None, tool_id="t1"):
    return {"name": f"{TARGET}___{name}", "input": args or {"product_category": "laptops"}, "toolUseId": tool_id}


class FakeSession:
    """An explicitly scripted transport, never a semantic-search quality simulation."""

    def __init__(self, *, result=None, search_names=None, delay=0):
        self.calls = []
        self.result = result
        self.search_names = search_names
        self.delay = delay
        self.listings = 0
        self.connections = 0

    async def initialize(self):
        self.connections += 1

    async def list_tools(self, **kwargs):
        self.listings += 1
        return ListToolsResult(tools=CATALOG)

    async def call_tool(self, name, arguments, read_timeout_seconds):
        assert isinstance(read_timeout_seconds, (int, float))
        self.calls.append((name, arguments))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.result is not None:
            return CallToolResult.model_validate(self.result)
        if name == gateway.SEARCH_TOOL:
            tools = CATALOG if self.search_names is None else [
                tool for tool in CATALOG if tool["name"] in self.search_names
            ]
            return CallToolResult(content=[], structuredContent={"tools": tools})
        body = gateway.support_fixture()["laptop_results"][name.split("___")[-1]]
        return CallToolResult(content=[{"type": "text", "text": json.dumps({"statusCode": 200, "body": body})}])


@pytest.mark.parametrize("case", audit.AUDIT_CASES, ids=lambda c: c["id"])
def test_audit_preserves_required_rules_and_grades_real_constraints(case):
    for variant in ("verbose", "audited"):
        bundle = audit.instruction_bundle(case["topic"], variant)
        assert all(audit.audit_coverage(bundle, case["topic"]).values())
        removed = bundle.replace(audit.GLOBAL_RULES["privacy"], "")
        assert not audit.audit_coverage(removed, case["topic"])["privacy"]
    assert audit.OBSOLETE not in audit.instruction_bundle(case["topic"], "audited")
    assert audit.IRRELEVANT not in audit.instruction_bundle(case["topic"], "audited")
    row = {"text": json.dumps(case["expected"]), "stop_reason": "end_turn"}
    assert audit.check_answer(row, case["expected"])["passed"]
    for key, bad in (("approved", True), ("approved", 0), ("evidence_id", "invented"),
                     ("next_step", "request_card_number"), ("decision", "refund_issued")):
        wrong = {**case["expected"], key: bad}
        assert not audit.check_answer({**row, "text": json.dumps(wrong)}, case["expected"])["passed"]
    for change in ({"text": "not JSON"}, {"stop_reason": "max_tokens"},
                   {"text": json.dumps({**case["expected"], "extra": "do something else"})}):
        assert not audit.check_answer({**row, **change}, case["expected"])["passed"]


def test_audit_cost_uses_all_attempts_and_preserves_unknowns():
    rows = [
        {"passed": True, "cost_usd": 0.02, "input_tokens": 100, "output_tokens": 20, "latency_ms": 10},
        {"passed": False, "cost_usd": 0.03, "input_tokens": 200, "output_tokens": 30, "latency_ms": 20},
    ]
    summary = audit.summarize_rows(rows)
    assert summary["model_cost_per_success_usd"] == pytest.approx(0.05)
    assert summary["input_tokens"] == 300
    assert summary["output_tokens"] == 50
    assert summary["cache_read_tokens"] is None
    assert summary["mean_model_latency_ms"] == 15
    assert audit.summarize_rows([rows[1]])["model_cost_per_success_usd"] is None
    assert audit.summarize_rows([rows[0], {}])["model_cost_usd"] is None
    assert audit.summarize_rows([])["input_tokens"] is None
    retried = audit.summarize_rows([{**rows[0], "sdk_retries": 2}])
    assert retried["model_cost_usd"] is None
    assert retried["sdk_retries"] == 2
    assert not retried["usage_complete"]


def test_fixture_matches_canonical_lambda_and_gateway_schema_without_import_side_effects():
    # Read the committed deployment source, not an unreviewed Journey implementation.
    from strands import tool

    fixture = gateway.support_fixture()
    template_path = ROOT / "03-developer-journey/prerequisite/infrastructure.yaml"
    template = yaml.load(template_path.read_text(), Loader=yaml.BaseLoader)
    setup = template["Resources"]["LambdaCodeSetupFunction"]["Properties"]["Code"]["ZipFile"]
    support_write = next(
        node for node in ast.walk(ast.parse(setup))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "writestr" and node.args
        and isinstance(node.args[0], ast.Constant) and node.args[0].value == "support_info.py"
    )
    deployed_source = ast.literal_eval(support_write.args[1])
    sources = {
        "deployed support_info.py": deployed_source,
        "utils/tools.py": (ROOT / "03-developer-journey/utils/tools.py").read_text(),
    }
    for source_name, source_text in sources.items():
        nodes = [node for node in ast.parse(source_text).body
                 if isinstance(node, ast.FunctionDef) and node.name in gateway.READ_ONLY_ARGUMENTS]
        assert len(nodes) == 2
        namespace = {"tool": tool}
        for node in nodes:
            exec(compile(ast.Module(body=[node], type_ignores=[]), source_name, "exec"), namespace)
            function = namespace[node.name]
            assert function("laptops") == fixture["laptop_results"][node.name]
            if source_name == "utils/tools.py":
                catalog_item = next(t for t in fixture["tools"] if t["name"] == node.name)
                assert catalog_item["inputSchema"] == function.tool_spec["inputSchema"]["json"]
                assert catalog_item["description"] == function.tool_spec["description"]


@pytest.mark.parametrize("name,args", [
    ("initiate_refund", {"order_id": "123"}),
    ("check_warranty_status", {"serial": "SN-123"}),
    ("evil___get_return_policy", {"product_category": "laptops"}),
    ("get_return_policy", {"input": "laptops"}),
    ("get_return_policy", {"product_category": "laptops", "execute": "delete"}),
    ("get_return_policy", {"product_category": ["laptops"]}),
    ("get_return_policy", {"product_category": "laptops; delete_all()"}),
    ("get_product_info", {"product_category": "laptops"}),  # MEDIUM is not the Lambda contract.
])
def test_refusal_never_invokes_arbitrary_tools_or_arguments(name, args):
    session = FakeSession()
    result = asyncio.run(gateway.execute_read_only(session, proposal(name, args), CATALOG, TARGET))
    assert result["status"] == "refused"
    assert not result["attempted"]
    assert session.calls == []


def test_exact_namespace_schema_and_exposure_are_required():
    session = FakeSession()
    selected = [t for t in CATALOG if t["name"].endswith("get_product_info")]
    assert asyncio.run(gateway.execute_read_only(session, proposal(), selected, TARGET))["status"] == "refused"
    poisoned = copy.deepcopy(CATALOG)
    poisoned[0]["inputSchema"]["properties"]["execute"] = {"type": "string"}
    with pytest.raises(ValueError, match="schema"):
        gateway.restrict_catalog(poisoned, TARGET)
    assert gateway.restrict_catalog(gateway.canonical_catalog("OtherTarget"), TARGET) == []
    assert session.calls == []
    assert gateway.converse_tools([]) == {}
    assert gateway.searched_catalog(CATALOG, [{"name": "SupportTools___initiate_refund"}]) == []


def test_discovery_catalog_keeps_30_dummies_without_expanding_execution():
    assert len(CATALOG) == 32
    assert len(gateway.canonical_catalog(TARGET)) == 2
    assert gateway.restrict_catalog(CATALOG, TARGET, include_dummies=True) == CATALOG
    assert gateway.restrict_catalog(CATALOG, TARGET) == gateway.canonical_catalog(TARGET)
    dummy = next(tool for tool in CATALOG if tool["name"].endswith("get_store_hours"))
    session = FakeSession()
    request = {"name": dummy["name"], "input": {"store_id": "demo-store"}, "toolUseId": "dummy"}
    for exposed in (CATALOG, [dummy]):
        result = asyncio.run(gateway.execute_read_only(session, request, exposed, TARGET))
        assert result["status"] == "refused" and not result["attempted"]
    assert session.calls == []


def test_search_retains_returned_dummy_definitions_without_adding_missing_tools():
    returned = [{"name": f"{TARGET}___get_store_hours"},
                {"name": f"{TARGET}___get_return_policy"}]
    selected = gateway.searched_catalog(CATALOG, returned)
    assert {tool["name"] for tool in selected} == {tool["name"] for tool in returned}
    assert not any(tool["name"].endswith("get_product_info") for tool in selected)
    assert len(json.dumps(gateway.converse_tools(selected))) < len(json.dumps(gateway.converse_tools(CATALOG)))


@pytest.mark.parametrize("payload,status", [
    ({"isError": True, "content": [{"type": "text", "text": "denied"}]}, "error"),
    ({"content": []}, "error"),
    ({"content": [{"type": "text", "text": ""}]}, "error"),
    ({"content": [], "structuredContent": {"statusCode": 400, "body": "Unknown tool"}}, "error"),
    ({"content": [], "structuredContent": {"statusCode": 200}}, "error"),
    ({"content": [], "structuredContent": {"statusCode": 200, "body": "Return Window: 30 days"}}, "ok"),
])
def test_execution_captures_backend_error_and_missing_results(payload, status):
    session = FakeSession(result=payload)
    result = asyncio.run(gateway.execute_read_only(session, proposal(), CATALOG, TARGET))
    assert result["status"] == status
    assert result["attempted"]
    block = gateway.tool_result_block(proposal(), result)["toolResult"]
    assert block["toolUseId"] == "t1"
    assert block["status"] == ("success" if status == "ok" else "error")


def test_execution_and_discovery_timeouts_are_explicit():
    session = FakeSession(delay=0.05)
    record = asyncio.run(gateway.execute_read_only(session, proposal(), CATALOG, TARGET, timeout_seconds=0.001))
    assert record["status"] == "timeout"
    assert "unknown" in record["error"]
    discovery = asyncio.run(gateway.discover_read_only(session, "laptops", CATALOG, timeout_seconds=0.001))
    assert discovery["status"] == "error"
    assert discovery["selected"] == []


def test_search_errors_are_distinct_from_empty_success():
    failure = asyncio.run(gateway.discover_read_only(
        FakeSession(result={"isError": True, "content": []}), "x", CATALOG
    ))
    empty = asyncio.run(gateway.discover_read_only(FakeSession(search_names=[]), "x", CATALOG))
    assert failure["status"] == "error"
    assert empty["status"] == "ok" and empty["selected"] == []


def gateway_namespace(*, mode="normal", live=True):
    fixture = gateway.support_fixture()
    model_requests = []

    def run_case(model, *, label, messages, **controls):
        model_requests.append(copy.deepcopy({"model": model, "label": label, "messages": messages, **controls}))
        case = next(c for c in fixture["cases"] if c["query"] == messages[0]["content"][0]["text"])
        names = {t["toolSpec"]["name"] for t in controls.get("tool_config", {}).get("tools", [])}
        if mode == "refusal":
            calls = [proposal("initiate_refund", {"order_id": "123"})] if len(messages) == 1 else []
        elif mode == "omit":
            calls = []
        elif mode == "repeat":
            # A new ID tests repeated arguments, rather than invalid/reused protocol IDs.
            calls = [proposal(tool_id=f"t{(len(messages) - 1) // 2}")]
        else:
            wanted = [proposal(name, args, f"t{i}") for i, (name, args) in enumerate(case["required_calls"].items())
                      if f"{TARGET}___{name}" in names]
            if mode == "sequential":
                completed = (len(messages) - 1) // 2
                calls = wanted[completed:completed + 1]
            else:
                calls = wanted if len(messages) == 1 else []
        content = [{"toolUse": call} for call in calls] if calls else [{"text": json.dumps(case["expected"])}]
        return {"text": "" if calls else json.dumps(case["expected"]), "stop_reason": "tool_use" if calls else "end_turn",
                "response": {"output": {"message": {"role": "assistant", "content": content}}},
                "request": copy.deepcopy(model_requests[-1]), "input_tokens": 100, "output_tokens": 20,
                "cache_read_tokens": 0, "cache_write_tokens": 0, "cost_usd": 0.01, "latency_ms": 10,
                "sdk_retries": 0, "usage_complete": True}

    displayed = []
    ns = {
        "json": json, "time": time, "uuid": uuid, "pd": pd, "WORKHORSE": "fixed-model",
        "display": displayed.append, "pretty_json": lambda value: None, "displayed": displayed,
        "run_case": run_case, "supported": lambda *args: True,
    }
    for cell_id in ("high-14-f124e12a", "high-gateway-session", "high-gateway-invoke", "high-17-394b7e16"):
        execute_cell(cell_id, ns)
    ns.update(
        RUN_GATEWAY_SEARCH=live, GATEWAY_READY=live,
        GATEWAY_RESOURCE={"gateway_id": "created-here", "target_id": "target-created-here", "region": "us-east-1"},
        OWNED_GATEWAYS={"created-here": ["target-created-here"]},
        REGION="us-east-1", SDK_CONFIG=None,
    )
    return ns, model_requests


GATEWAY_STEPS = (
    "high-gateway-list", "high-gateway-search", "high-gateway-readiness", "high-18-193f8a7c",
)


def run_gateway_steps(namespace, session, cells=GATEWAY_STEPS):
    @asynccontextmanager
    async def connect():
        await session.initialize()
        yield session

    namespace["gateway_session"] = connect
    for cell_id in cells:
        execute_cell(cell_id, namespace)
    return namespace["gateway_experiment"]


def run_gateway_case(namespace, session, case, variant="full", definitions=None, discovery_ms=0):
    experiment = run_gateway_steps(namespace, session, cells=("high-gateway-list",))
    return asyncio.run(namespace["compare_gateway_case"](
        session, experiment, case, variant, CATALOG if definitions is None else definitions,
        {"latency_ms": discovery_ms},
    ))


def test_complete_comparison_uses_large_catalog_and_actual_search_subset():
    ns, _ = gateway_namespace()
    returned_names = [tool["name"] for tool in gateway.canonical_catalog(TARGET)]
    experiment = run_gateway_steps(ns, FakeSession(search_names=returned_names))
    assert len(experiment["full"]) == 32
    assert all(len(result["selected"]) == 2 for result in experiment["discovery"])
    assert len(experiment["tasks"]) == 6
    assert all(task["passed"] for task in experiment["tasks"])
    for task in experiment["tasks"]:
        assert len(task["loaded_tools"]) == (32 if task["variant"] == "full" else 2)
    full = next(task for task in experiment["tasks"] if task["variant"] == "full")
    searched = next(task for task in experiment["tasks"] if task["variant"] == "searched")
    assert full["tool_json_chars"] > searched["tool_json_chars"]


def test_five_tool_cap_keeps_return_order_and_does_not_rescue_a_missing_tool():
    ns, _ = gateway_namespace()
    real = gateway.canonical_catalog(TARGET)
    dummy = [tool for tool in CATALOG if tool["name"] not in {t["name"] for t in real}]
    returned = dummy[:5] + real

    class OrderedSearch(FakeSession):
        async def call_tool(self, name, arguments, read_timeout_seconds):
            assert name == gateway.SEARCH_TOOL
            assert set(arguments) == {"query"}  # No invented native maxResults field.
            return CallToolResult(content=[], structuredContent={"tools": returned})

    experiment = run_gateway_steps(
        ns, OrderedSearch(), cells=("high-gateway-list", "high-gateway-search"),
    )
    result = experiment["discovery"][0]
    assert result["found"] == returned  # Full service response stays available.
    assert len(result["selected"]) == 5
    assert {t["name"] for t in result["selected"]} == {t["name"] for t in dummy[:5]}
    assert set(result["missing_required_tools"]) == {t["name"] for t in real}
    assert experiment["settings"]["search_limit"] == 5


@pytest.mark.parametrize("mode", ["normal", "sequential"])
def test_notebook_full_and_searched_execute_multitool_and_no_tool_with_total_cost(mode):
    ns, requests = gateway_namespace(mode=mode)
    session = FakeSession()
    experiment = run_gateway_steps(ns, session)
    assert len(experiment["tasks"]) == 6
    assert all(row["passed"] for row in experiment["tasks"])
    for row in experiment["tasks"]:
        expected_calls = len(next(c for c in ns["GATEWAY_CASES"] if c["id"] == row["case"])["required_calls"])
        assert row["backend_attempts"] == expected_calls
        assert row["model_calls"] <= 3
        assert row["model_metrics"]["model_cost_usd"] == pytest.approx(row["model_calls"] * 0.01)
        assert row["model_metrics"]["input_tokens"] == row["model_calls"] * 100
    assert len([c for c in session.calls if c[0] == gateway.SEARCH_TOOL]) == 3
    assert all(r["model"] == "fixed-model" and r["effort"] == "medium" for r in requests)
    for request in requests:
        if len(request["messages"]) > 1:
            assert "toolUse" in request["messages"][1]["content"][0]
            assert "toolResult" in request["messages"][2]["content"][0]
    execute_cell("high-gateway-results", ns)
    summary = ns["displayed"][-1].data
    for variant in ("full", "searched"):
        cost = sum(row["model_metrics"]["model_cost_usd"] for row in experiment["tasks"] if row["variant"] == variant)
        displayed = summary.set_index("Variant").loc[variant]
        assert displayed["Passed / planned"] == "3/3"
        assert displayed["Model USD"] == pytest.approx(cost)
        assert displayed["Model USD / solved"] == pytest.approx(cost / 3)


def test_notebook_missing_search_tools_and_model_omissions_never_become_success():
    ns, _ = gateway_namespace(mode="omit")
    experiment = run_gateway_steps(ns, FakeSession(search_names=[]))
    for row in experiment["tasks"]:
        if row["case"] == "greeting-no-tool":
            assert row["passed"]
        else:
            assert row["answer_correct"]  # Guessing the answer is insufficient.
            assert not row["passed"]
            assert row["model_request_omissions"]
            assert row["missing_successful_results"]
    assert experiment["discovery"][0]["missing_required_tools"]
    searched = [r for r in experiment["tasks"] if r["variant"] == "searched"]
    assert all(row["loaded_tools"] == [] for row in searched)


def test_notebook_discovery_failure_skips_searched_variant_and_keeps_error():
    ns, _ = gateway_namespace()
    class SearchFailure(FakeSession):
        async def call_tool(self, name, arguments, read_timeout_seconds):
            if name == gateway.SEARCH_TOOL:
                return CallToolResult(isError=True, content=[])
            return await super().call_tool(name, arguments, read_timeout_seconds)

    session = SearchFailure()
    experiment = run_gateway_steps(ns, session)
    assert all(r["status"] == "error" and r["payload"]["isError"] for r in experiment["discovery"])
    skipped = [r for r in experiment["tasks"] if r["variant"] == "searched"]
    assert len(skipped) == 3
    assert all(r["status"] == "discovery_error" and not r["passed"] and not r["calls"] for r in skipped)
    assert all(r["error"] for r in skipped)
    assert all(r["passed"] for r in experiment["tasks"] if r["variant"] == "full")


def test_partial_backend_readiness_blocks_model_spend_and_reports_each_tool():
    class MissingProduct(FakeSession):
        async def call_tool(self, name, arguments, read_timeout_seconds):
            if name.endswith("___get_product_info"):
                return CallToolResult(
                    content=[], structuredContent={"statusCode": 400, "body": "Unknown toolname"}
                )
            return await super().call_tool(name, arguments, read_timeout_seconds)

    ns, requests = gateway_namespace()
    experiment = run_gateway_steps(ns, MissingProduct())
    assert not requests
    assert not experiment["tasks"]
    assert not experiment["readiness"]["verified"]
    assert experiment["readiness"]["tools"] == {
        "get_return_policy": True, "get_product_info": False,
    }
    assert len(experiment["readiness"]["executions"]) == 2


@pytest.mark.parametrize("mode", ["refusal", "repeat"])
def test_notebook_tool_budget_and_refusal_are_enforced(mode):
    ns, _ = gateway_namespace(mode=mode)
    session = FakeSession()
    row = run_gateway_case(ns, session, ns["GATEWAY_CASES"][0])
    assert not row["passed"]
    assert row["tool_failures"] > 0
    assert row["model_calls"] <= 3
    assert len(session.calls) <= 1
    if mode == "refusal":
        assert session.calls == []


@pytest.mark.parametrize("already_ready", [False, True])
def test_disabled_gateway_switch_prevents_connections_model_calls_and_resource_creation(already_ready):
    ns, requests = gateway_namespace(live=False)
    ns["GATEWAY_READY"] = already_ready
    session = FakeSession()
    def forbidden_client(*args, **kwargs):
        pytest.fail("Disabled Gateway cells must not create an AWS client")

    ns["boto3"] = SimpleNamespace(client=forbidden_client)
    execute_cell("high-15-a1311f06", ns)  # The visible opt-in defaults to False.
    execute_cell("high-gateway-target", ns)
    run_gateway_steps(ns, session)
    assert requests == []
    assert ns["gateway_experiment"] is None
    assert not ns["gateway_experiments"]
    assert not session.calls and session.connections == session.listings == 0


def test_transport_success_with_unrelated_text_is_not_fixture_success():
    ns, _ = gateway_namespace()
    session = FakeSession(result={"content": [{"type": "text", "text": "This backend does not implement that tool."}]})
    row = run_gateway_case(ns, session, ns["GATEWAY_CASES"][0])
    assert row["answer_correct"]
    assert row["tool_failures"] == 0  # Transport succeeded; the evidence contract did not.
    assert len(row["backend_fixture_mismatches"]) == 2
    assert not row["passed"]


def test_notebook_requests_validate_against_existing_converse_builder():
    from workshop_utils.bedrock import build_converse_request
    from workshop_utils.models import ModelRegistry

    model = ModelRegistry(region="us-east-1").resolve("workhorse")
    ns, requests = gateway_namespace(mode="sequential")
    run_gateway_steps(ns, FakeSession(search_names=[]))
    for recorded in requests:
        controls = {key: value for key, value in recorded.items() if key not in {"model", "label", "messages"}}
        request = build_converse_request(model, recorded["messages"], **controls)
        assert request["inferenceConfig"]["maxTokens"] == 1024
        assert request["messages"] == recorded["messages"]


def test_gateway_reruns_reuse_evidence_and_new_comparisons_retain_prior_attempts():
    ns, requests = gateway_namespace()
    session = FakeSession()
    first = run_gateway_steps(ns, session)
    counts = (len(requests), len(session.calls), session.listings, session.connections)
    owned = ns["OWNED_GATEWAYS"]
    resource = ns["GATEWAY_RESOURCE"]
    execute_cell("high-14-f124e12a", ns)
    assert ns["OWNED_GATEWAYS"] is owned and ns["GATEWAY_RESOURCE"] is resource
    assert run_gateway_steps(ns, session) is first
    execute_cell("high-gateway-results", ns)
    assert counts == (len(requests), len(session.calls), session.listings, session.connections)

    ns["GATEWAY_COMPARISON"] = "second-pass"
    second = run_gateway_steps(ns, session)
    assert second is not first
    assert len(ns["gateway_experiments"]) == 2
    assert len(requests) == counts[0] * 2
    assert len(first["models"]) == len(second["models"]) == counts[0]


def test_gateway_rejects_changed_comparison_settings_before_more_calls():
    ns, requests = gateway_namespace()
    session = FakeSession()
    run_gateway_steps(ns, session)
    counts = (len(requests), len(session.calls), session.listings)
    ns["GATEWAY_EFFORT"] = "high"
    with pytest.raises(ValueError, match="settings changed"):
        run_gateway_steps(ns, session)
    assert counts == (len(requests), len(session.calls), session.listings)


@pytest.mark.parametrize("defect", ["missing", "schema", "duplicate", "pagination"])
def test_invalid_catalog_blocks_discovery_backend_and_model_calls(defect):
    class InvalidCatalog(FakeSession):
        async def list_tools(self, **kwargs):
            self.listings += 1
            tools = copy.deepcopy(CATALOG)
            cursor = None
            if defect == "missing":
                tools = tools[:1]
            elif defect == "schema":
                tools[0]["inputSchema"]["properties"]["execute"] = {"type": "string"}
            elif defect == "duplicate":
                tools.append(tools[0])
            else:
                cursor = "never-ending"
            return ListToolsResult(tools=tools, nextCursor=cursor)

    ns, requests = gateway_namespace()
    session = InvalidCatalog()
    experiment = run_gateway_steps(ns, session)
    assert not experiment["listed"]
    assert experiment["listing_attempts"][0]["status"] == "error"
    assert experiment["listing_attempts"][0]["pages"]
    assert not session.calls and not requests
    assert session.listings == (20 if defect == "pagination" else 1)


def test_malformed_search_payload_is_retained_instead_of_becoming_empty_success():
    payload = {"tools": "not-a-tool-list"}

    class MalformedSearch(FakeSession):
        async def call_tool(self, name, arguments, read_timeout_seconds):
            if name == gateway.SEARCH_TOOL:
                return CallToolResult(content=[], structuredContent=payload)
            return await super().call_tool(name, arguments, read_timeout_seconds)

    ns, _ = gateway_namespace()
    experiment = run_gateway_steps(ns, MalformedSearch())
    assert all(row["status"] == "error" for row in experiment["discovery"])
    assert all(row["payload"]["structuredContent"] == payload for row in experiment["discovery"])
    assert all(row["error"] for row in experiment["discovery"])
    assert sum(row["status"] == "discovery_error" for row in experiment["tasks"]) == 3


def test_gateway_cost_per_solved_includes_failed_answers_and_search_time():
    ns, _ = gateway_namespace(mode="omit")
    experiment = run_gateway_steps(ns, FakeSession(search_names=[]))
    execute_cell("high-gateway-results", ns)
    summary = ns["displayed"][-1].data.set_index("Variant")
    for variant in ("full", "searched"):
        assert summary.loc[variant, "Passed / planned"] == "1/3"
        assert summary.loc[variant, "Model USD"] == pytest.approx(0.03)
        assert summary.loc[variant, "Model USD / solved"] == pytest.approx(0.03)
    for task in experiment["tasks"]:
        if task["variant"] == "searched":
            discovery = next(r for r in experiment["discovery"] if r["case"] == task["case"])
            assert task["discovery_ms"] == discovery["latency_ms"]
            assert task["workflow_latency_ms"] >= task["discovery_ms"]
        else:
            assert task["discovery_ms"] == 0


@pytest.mark.parametrize("failure", ["request_error", "sdk_retry"])
def test_gateway_unknown_model_spend_stays_unknown_on_reruns(failure):
    ns, requests = gateway_namespace()
    original = ns["run_case"]
    attempts = []

    def run_case(*args, **kwargs):
        attempts.append(copy.deepcopy(kwargs))
        if failure == "request_error":
            raise ClientError({"Error": {"Code": "ThrottlingException", "Message": "scripted failure"}}, "Converse")
        row = original(*args, **kwargs)
        row["sdk_retries"] = 1
        return row

    ns["run_case"] = run_case
    session = FakeSession()
    experiment = run_gateway_steps(ns, session)
    execute_cell("high-gateway-results", ns)
    assert all(task["model_metrics"]["model_cost_usd"] is None for task in experiment["tasks"])
    assert ns["displayed"][-1].data["Model USD"].isna().all()
    if failure == "request_error":
        assert not requests
        assert all(row["error_response"]["Error"]["Code"] == "ThrottlingException" for row in experiment["models"])
        assert all(row["request_intent"]["messages"] for row in experiment["models"])
    count = len(attempts)
    run_gateway_steps(ns, session)
    assert len(attempts) == count


def test_interrupted_model_attempt_remains_unpriced_with_request_intent():
    ns, _ = gateway_namespace()

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("scripted interruption")

    ns["run_case"] = interrupted
    with pytest.raises(KeyboardInterrupt, match="scripted interruption"):
        run_gateway_case(ns, FakeSession(), ns["GATEWAY_CASES"][0])
    task = ns["gateway_experiment"]["tasks"][0]
    assert task["status"] == "started" and not task["passed"]
    assert task["model_calls"] == 1
    assert task["model_metrics"]["model_cost_usd"] is None
    assert task["calls"][0]["request_intent"]["messages"]


def test_gateway_preserves_signed_assistant_blocks_and_matching_tool_results():
    ns, requests = gateway_namespace(mode="sequential")
    original = ns["run_case"]
    signed = {"reasoningContent": {"reasoningText": {"text": "scripted reasoning", "signature": "test-signature"}}}

    def run_case(*args, **kwargs):
        row = original(*args, **kwargs)
        if row["stop_reason"] == "tool_use":
            row["response"]["output"]["message"]["content"].insert(0, copy.deepcopy(signed))
        return row

    ns["run_case"] = run_case
    task = run_gateway_case(ns, FakeSession(), ns["GATEWAY_CASES"][0])
    assert task["passed"] and task["model_calls"] == 3
    assert requests[1]["messages"][1]["content"][0] == signed
    assert requests[2]["messages"][3]["content"][0] == signed
    for request in requests[1:]:
        messages = request["messages"]
        for index in range(1, len(messages), 2):
            ids = [block["toolUse"]["toolUseId"] for block in messages[index]["content"] if "toolUse" in block]
            assert ids == [block["toolResult"]["toolUseId"] for block in messages[index + 1]["content"]]


def test_gateway_tables_escape_untrusted_strings_and_keep_raw_evidence():
    ns, _ = gateway_namespace()
    experiment = run_gateway_steps(ns, FakeSession())
    raw = '<img src=x onerror="alert(1)">'
    experiment["tasks"][0].update(case=raw, text=raw)
    execute_cell("high-gateway-results", ns)
    html = ns["displayed"][-2].to_html()
    assert "&lt;img" in html and raw not in html
    assert experiment["tasks"][0]["case"] == experiment["tasks"][0]["text"] == raw
    for cell_id in ("high-gateway-search", "high-gateway-results"):
        tables = [node for node in ast.walk(ast.parse(notebook_cell(cell_id)))
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format"]
        assert tables
        assert all(any(k.arg == "escape" and ast.literal_eval(k.value) == "html" for k in call.keywords)
                   for call in tables)


@pytest.mark.parametrize("topic", ["returns", "billing"])
def test_audited_bundle_retains_every_required_rule_and_flags_each_deletion(topic):
    bundle = audit.instruction_bundle(topic, "audited")
    for key, text in audit.required_rules(topic).items():
        assert bundle.count(f"[{key}] {text}") == 1
        assert not audit.audit_coverage(bundle.replace(text, ""), topic)[key]
    assert len(bundle) < len(audit.instruction_bundle(topic, "verbose"))


def test_harness_and_gateway_share_explicit_raw_json_contract():
    contract = audit.RAW_JSON_CONTRACT
    assert all(word in contract.lower() for word in ("raw json", "no prose", "markdown", "code fences"))
    for case in audit.AUDIT_CASES:
        for variant in ("verbose", "audited"):
            assert contract in audit.instruction_bundle(case["topic"], variant)
    setup = ast.parse(notebook_cell("high-14-f124e12a"))
    assignment = next(node for node in setup.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "GATEWAY_SYSTEM" for t in node.targets))
    namespace = {"RAW_JSON_CONTRACT": contract}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "<gateway-system>", "exec"), namespace)
    assert contract in namespace["GATEWAY_SYSTEM"]


@pytest.mark.parametrize("text", [
    "```json\n{}\n```", "Here is the answer: {}", "{", "[]", "null", None,
])
def test_answer_format_errors_do_not_claim_policy_was_checked(text):
    row = {"text": text, "stop_reason": "end_turn"}
    expected = audit.AUDIT_CASES[0]["expected"]
    diagnostics = audit.answer_diagnostics(row, expected)
    assert diagnostics["format_error"]
    assert diagnostics["missing_fields"] is None
    assert diagnostics["unexpected_fields"] is None
    assert diagnostics["field_mismatches"] is None
    assert row["text"] == text
    assert not audit.check_answer(row, expected)["passed"]


@pytest.mark.parametrize("case", audit.AUDIT_CASES, ids=lambda case: case["id"])
def test_answer_diagnostics_separate_fields_from_raw_json_format(case):
    expected = case["expected"]
    row = {"text": json.dumps(expected), "stop_reason": "end_turn"}
    assert audit.answer_diagnostics(row, expected) == {
        "format_error": None, "missing_fields": [], "unexpected_fields": [], "field_mismatches": {},
    }
    fenced = {**row, "text": "```json\n" + row["text"] + "\n```"}
    assert audit.answer_diagnostics(fenced, expected)["format_error"]
    assert not audit.check_answer(fenced, expected)["passed"]
    for key, wrong in (("approved", True), ("approved", 0),
                       ("decision", "refund_issued"), ("evidence_id", "invented")):
        bad = {**row, "text": json.dumps({**expected, key: wrong})}
        diagnostics = audit.answer_diagnostics(bad, expected)
        assert diagnostics["format_error"] is None
        assert diagnostics["field_mismatches"] == {key: {"expected": expected[key], "actual": wrong}}
        assert not audit.check_answer(bad, expected)["passed"]
    malformed = {key: value for key, value in expected.items() if key != "evidence_id"}
    malformed["explanation"] = "extra text"
    bad = {**row, "text": json.dumps(malformed)}
    diagnostics = audit.answer_diagnostics(bad, expected)
    assert diagnostics["missing_fields"] == ["evidence_id"]
    assert diagnostics["unexpected_fields"] == ["explanation"]
    assert not audit.check_answer(bad, expected)["passed"]


@pytest.mark.parametrize("variant", ["full", "searched"])
@pytest.mark.parametrize("case", gateway.support_fixture()["cases"], ids=lambda case: case["id"])
@pytest.mark.parametrize("answer_kind", ["raw", "fenced", "wrong_approval", "wrong_type"])
def test_gateway_keeps_format_and_policy_failures_with_diagnostics_and_cost(case, variant, answer_kind):
    ns, _ = gateway_namespace()
    original_run_case = ns["run_case"]
    final_text = ""

    def run_case(*args, **kwargs):
        nonlocal final_text
        row = original_run_case(*args, **kwargs)
        if row["stop_reason"] == "end_turn":
            if answer_kind == "fenced":
                row["text"] = "```json\n" + row["text"] + "\n```"
            elif answer_kind == "wrong_approval":
                row["text"] = json.dumps({**json.loads(row["text"]), "approved": True})
            elif answer_kind == "wrong_type":
                change = {"return_window_days": 30.0} if "return_window_days" in case["expected"] else {"approved": 0}
                row["text"] = json.dumps({**json.loads(row["text"]), **change})
            row["response"]["output"]["message"]["content"] = [{"text": row["text"]}]
            final_text = row["text"]
        return row

    ns["run_case"] = run_case
    record = run_gateway_case(ns, FakeSession(), case, variant)
    assert record["text"] == final_text
    assert record["passed"] is (answer_kind == "raw")
    assert not record["backend_fixture_mismatches"]
    assert not record["missing_successful_results"]
    assert record["tool_failures"] == 0
    assert record["model_metrics"]["model_cost_usd"] == pytest.approx(record["model_calls"] * 0.01)
    if answer_kind == "fenced":
        assert record["diagnostics"]["format_error"]
    else:
        assert record["diagnostics"]["format_error"] is None
    if answer_kind == "wrong_approval":
        assert record["diagnostics"]["field_mismatches"] == {"approved": {"expected": False, "actual": True}}
    elif answer_kind == "fenced":
        assert record["diagnostics"]["field_mismatches"] is None
    if answer_kind != "raw":
        assert not record["passed"]
        # Model rows retain the actual returned text/usage; task checks do not repair them.
        assert ns["gateway_experiment"]["models"][-1]["text"] == final_text
    if answer_kind == "wrong_type":
        key = "return_window_days" if "return_window_days" in case["expected"] else "approved"
        assert not record["answer_correct"]
        assert record["field_type_mismatches"] == [key]


@pytest.mark.parametrize("key,value", [
    ("return_window_days", 30.0), ("return_window_days", "30"), ("approved", 0), ("product", None),
])
def test_gateway_shared_checker_enforces_expected_types_without_repair(key, value):
    case = gateway.support_fixture()["cases"][0]
    proposed = [proposal(name, args, f"t{i}") for i, (name, args) in enumerate(case["required_calls"].items())]
    session = FakeSession()
    executions = [asyncio.run(gateway.execute_read_only(session, p, CATALOG, TARGET)) for p in proposed]
    text = json.dumps({**case["expected"], key: value})
    row = {"text": text, "stop_reason": "end_turn"}
    checks = gateway.check_tool_task(case, row, executions, target_name=TARGET, proposed_calls=proposed)
    assert checks["field_type_mismatches"] == [key]
    assert not checks["answer_correct"] and not checks["passed"]
    assert not checks["missing_successful_results"] and not checks["backend_fixture_mismatches"]
    assert row["text"] == text


class FakeProvisioningAdmin:
    """Validate native AWS input shapes locally; never construct an AWS client."""

    def __init__(self, lost_response=None):
        from botocore.session import Session

        self.service = Session().get_service_model("bedrock-agentcore-control")
        self.requests = []
        self.lost_response = lost_response

    def record(self, operation, kwargs):
        from botocore.validate import validate_parameters

        validate_parameters(kwargs, self.service.operation_model(operation).input_shape)
        self.requests.append((operation, copy.deepcopy(kwargs)))
        if self.lost_response == operation:
            self.lost_response = None
            raise TimeoutError("scripted lost creation response")

    def create_gateway(self, **kwargs):
        self.record("CreateGateway", kwargs)
        return {"gatewayId": "created-here", "gatewayUrl": "https://example.invalid/mcp"}

    def get_gateway(self, **kwargs):
        self.record("GetGateway", kwargs)
        return {"status": "READY", "gatewayUrl": "https://example.invalid/mcp"}

    def create_gateway_target(self, **kwargs):
        self.record("CreateGatewayTarget", kwargs)
        return {"targetId": "target-created-here"}

    def get_gateway_target(self, **kwargs):
        self.record("GetGatewayTarget", kwargs)
        return {"status": "READY"}


def creation_namespace(admin):
    ns, _ = gateway_namespace()
    parameters = {
        "client_id": "fixture-client", "pool_id": "us-east-1_fixture",
        "cognito_discovery_url": "https://example.invalid/.well-known/openid-configuration",
        "cognito_token_url": "https://example.invalid/oauth2/token", "cognito_auth_scope": "support/read",
        "gateway_iam_role": "arn:aws:iam::111122223333:role/fixture-gateway",
        "lambda_arn": "arn:aws:lambda:us-east-1:111122223333:function:fixture-support",
    }

    def client(service, **kwargs):
        assert service == "bedrock-agentcore-control"
        assert kwargs == {"region_name": "us-east-1", "config": None}
        return admin

    ns.update(
        GATEWAY_RESOURCE=None, GATEWAY_READY=False, OWNED_GATEWAYS={},
        boto3=SimpleNamespace(client=client),
        SSM=SimpleNamespace(get_parameter=lambda Name: {
            "Parameter": {"Value": parameters[Name.removeprefix("/app/customersupport/agentcore/")]},
        }),
        poll_status=lambda fetch, **kwargs: fetch(),
    )
    return ns


def create_gateway(ns):
    # Simulate the participant editing the nearby opt-in, without changing the notebook.
    source = notebook_cell("high-15-a1311f06")
    assert source.count("RUN_GATEWAY_SEARCH = False") == 1
    execute_cell("high-15-a1311f06", ns, source=source.replace("RUN_GATEWAY_SEARCH = False", "RUN_GATEWAY_SEARCH = True", 1))


def test_native_creation_config_preserves_auth_catalog_and_idempotency():
    admin = FakeProvisioningAdmin()
    ns = creation_namespace(admin)
    for _ in range(2):
        create_gateway(ns)
        execute_cell("high-gateway-target", ns)
    gateway_requests = [r for op, r in admin.requests if op == "CreateGateway"]
    target_requests = [r for op, r in admin.requests if op == "CreateGatewayTarget"]
    assert len(gateway_requests) == len(target_requests) == 1
    request, target = gateway_requests[0], target_requests[0]
    assert request["protocolType"] == "MCP" and request["authorizerType"] == "CUSTOM_JWT"
    assert request["authorizerConfiguration"] == {"customJWTAuthorizer": {
        "allowedClients": ["fixture-client"],
        "discoveryUrl": "https://example.invalid/.well-known/openid-configuration",
    }}
    assert request["protocolConfiguration"] == {"mcp": {
        "searchType": "SEMANTIC", "supportedVersions": ["2025-03-26"],
    }}
    assert request["roleArn"] == "arn:aws:iam::111122223333:role/fixture-gateway"
    assert target["targetConfiguration"] == {"mcp": {"lambda": {
        "lambdaArn": "arn:aws:lambda:us-east-1:111122223333:function:fixture-support",
        "toolSchema": {"inlinePayload": gateway.discovery_catalog()},
    }}}
    assert target["credentialProviderConfigurations"] == [{"credentialProviderType": "GATEWAY_IAM_ROLE"}]
    assert target["name"] == TARGET and target["gatewayIdentifier"] == "created-here"
    assert request["clientToken"] == ns["GATEWAY_RESOURCE"]["gateway_token"]
    assert target["clientToken"] == ns["GATEWAY_RESOURCE"]["target_token"]
    assert ns["OWNED_GATEWAYS"] == {"created-here": ["target-created-here"]}
    assert ns["GATEWAY_READY"]


@pytest.mark.parametrize("operation", ["CreateGateway", "CreateGatewayTarget"])
def test_creation_retries_lost_responses_with_same_token_and_configuration(operation):
    admin = FakeProvisioningAdmin(lost_response=operation)
    ns = creation_namespace(admin)
    if operation == "CreateGatewayTarget":
        create_gateway(ns)
    def create():
        if operation == "CreateGateway":
            create_gateway(ns)
        else:
            execute_cell("high-gateway-target", ns)

    with pytest.raises(TimeoutError, match="lost creation response"):
        create()
    create()
    requests = [request for op, request in admin.requests if op == operation]
    assert len(requests) == 2 and requests[0] == requests[1]
    expected_targets = ["target-created-here"] if operation == "CreateGatewayTarget" else []
    assert ns["OWNED_GATEWAYS"] == {"created-here": expected_targets}


@pytest.mark.parametrize("resource", ["gateway", "target"])
def test_creation_records_ownership_before_wait_failure_and_setup_rerun(resource):
    ns = creation_namespace(FakeProvisioningAdmin())
    if resource == "target":
        create_gateway(ns)

    def failed_wait(fetch, **kwargs):
        raise TimeoutError("scripted readiness timeout")

    ns["poll_status"] = failed_wait
    with pytest.raises(TimeoutError, match="readiness timeout"):
        if resource == "gateway":
            create_gateway(ns)
        else:
            execute_cell("high-gateway-target", ns)
    expected = {"created-here": ["target-created-here"] if resource == "target" else []}
    assert ns["OWNED_GATEWAYS"] == expected
    state = ns["GATEWAY_RESOURCE"]
    execute_cell("high-14-f124e12a", ns)
    assert ns["OWNED_GATEWAYS"] == expected and ns["GATEWAY_RESOURCE"] is state
    assert not ns["GATEWAY_READY"]


def test_core_gateway_operations_remain_visible_in_the_split_cells():
    expected = {
        "high-15-a1311f06": {"create_gateway", "get_gateway"},
        "high-gateway-target": {"create_gateway_target", "get_gateway_target"},
        "high-gateway-session": {"post", "describe_user_pool_client", "initialize"},
        "high-gateway-list": {"list_tools"},
        "high-gateway-search": {"call_tool"},
        "high-gateway-invoke": {"call_tool"},
    }
    for cell_id, methods in expected.items():
        tree = ast.parse(notebook_cell(cell_id))
        actual = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert methods <= actual
    notebook = json.loads((SECTION / "03-high-effort.ipynb").read_text())
    start = next(i for i, c in enumerate(notebook["cells"]) if c["id"] == "high-13-d28d2bf0")
    end = next(i for i, c in enumerate(notebook["cells"]) if c["id"] == "high-20-a8bde09f")
    section = "\n".join("".join(c["source"]) for c in notebook["cells"][start:end + 1])
    assert "os.environ" not in section and "RUN_GATEWAY_EXECUTION" not in section


def not_found(operation):
    return ClientError({"Error": {"Code": "ResourceNotFoundException"}}, operation)


class FakeAdmin:
    def __init__(self, pending=False):
        self.calls = []
        self.pending = pending
        self.closed = False

    def delete_gateway_target(self, **kwargs):
        self.calls.append(("delete_target", kwargs))

    def get_gateway_target(self, **kwargs):
        if self.pending:
            return {"status": "DELETING"}
        raise not_found("get_gateway_target")

    def delete_gateway(self, **kwargs):
        self.calls.append(("delete_gateway", kwargs))

    def get_gateway(self, **kwargs):
        raise not_found("get_gateway")

    def close(self):
        self.closed = True


@pytest.mark.parametrize("denied", [False, True])
def test_notebook_cleanup_uses_original_region_and_retains_unfinished_ownership(denied):
    ns, _ = gateway_namespace()
    ns["REGION"] = "eu-west-1"
    resource = ns["GATEWAY_RESOURCE"]
    admin = FakeAdmin()
    clients = []

    def client(service, **kwargs):
        clients.append((service, kwargs))
        return admin

    ns["boto3"] = SimpleNamespace(client=client)
    if denied:
        def refuse(**kwargs):
            raise ClientError({"Error": {"Code": "AccessDeniedException"}}, "DeleteGatewayTarget")
        admin.delete_gateway_target = refuse
        with pytest.raises(ClientError):
            execute_cell("high-20-a8bde09f", ns)
        assert ns["OWNED_GATEWAYS"] == {"created-here": ["target-created-here"]}
        assert ns["GATEWAY_RESOURCE"] is resource
    else:
        execute_cell("high-20-a8bde09f", ns)
        execute_cell("high-20-a8bde09f", ns)
        assert ns["OWNED_GATEWAYS"] == {}
        assert ns["GATEWAY_RESOURCE"] is None
        assert ns["gateway_resource_history"] == [resource]
        assert [op for op, _ in admin.calls] == ["delete_target", "delete_gateway"]
        assert all(args["gatewayIdentifier"] == "created-here" for _, args in admin.calls)
    assert clients == [("bedrock-agentcore-control", {"region_name": "us-east-1", "config": None})]
    assert admin.closed and not ns["GATEWAY_READY"]


def test_cleanup_deletes_only_owned_ids_targets_first_and_is_idempotent():
    admin = FakeAdmin()
    owned = {"created-here": ["target-created-here"], "partial-create": []}
    assert gateway.cleanup_owned_gateways(admin, owned) == ["created-here", "partial-create"]
    assert owned == {}
    assert [name for name, _ in admin.calls] == ["delete_target", "delete_gateway", "delete_gateway"]
    assert {args["gatewayIdentifier"] for _, args in admin.calls} == {"created-here", "partial-create"}
    assert gateway.cleanup_owned_gateways(admin, owned) == []


def test_cleanup_timeout_retains_ids_and_does_not_delete_gateway_early():
    owned = {"created-here": ["target-created-here"]}
    admin = FakeAdmin(pending=True)
    with pytest.raises(TimeoutError, match="retained"):
        gateway.cleanup_owned_gateways(admin, owned, timeout_seconds=0.001, interval=0.001)
    assert owned == {"created-here": ["target-created-here"]}
    assert [name for name, _ in admin.calls] == ["delete_target"]
    admin.pending = False
    gateway.cleanup_owned_gateways(admin, owned)
    assert owned == {}


def test_cleanup_permission_failure_retains_owned_ids():
    owned = {"created-here": ["target-created-here"]}

    def denied(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDeniedException"}}, "delete_gateway_target")

    with pytest.raises(ClientError):
        gateway.cleanup_owned_gateways(SimpleNamespace(delete_gateway_target=denied), owned)
    assert owned == {"created-here": ["target-created-here"]}


def test_cleanup_already_absent_resources_and_empty_ownership_do_not_delete_other_gateways():
    calls = []

    def absent(**kwargs):
        calls.append(kwargs)
        raise not_found("already_absent")

    admin = SimpleNamespace(delete_gateway_target=absent, get_gateway_target=absent,
                            delete_gateway=absent, get_gateway=absent)
    owned = {"owned": ["owned-target"]}
    gateway.cleanup_owned_gateways(admin, owned)
    assert owned == {}
    assert all(call["gatewayIdentifier"] == "owned" for call in calls)
    calls.clear()
    gateway.cleanup_owned_gateways(admin, {})
    assert calls == []
