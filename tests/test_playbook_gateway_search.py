from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from mcp.types import CallToolResult

spec = importlib.util.spec_from_file_location(
    "playbook_gateway_search_test",
    Path(__file__).resolve().parents[1] / "02-optimization-playbook/utils/gateway.py",
)
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)
search_tools = gateway.search_tools


@pytest.mark.parametrize("structured", [True, False])
def test_search_accepts_both_mcp_representations(structured):
    tools = [{"name": "SupportTools___check_warranty_status", "inputSchema": {"type": "object"}}]
    payload = {"isError": False}
    if structured:
        payload["structuredContent"] = {"tools": tools}
    else:
        payload.update(structuredContent=None, content=[{"type": "text", "text": json.dumps({"tools": tools})}])
    assert search_tools(payload) == tools


@pytest.mark.parametrize("payload", [
    {"isError": True, "structuredContent": {"tools": []}},
    {"content": [{"type": "text", "text": "invalid json"}]},
    {"structuredContent": {"tools": "not a list"}},
    {"structuredContent": {"tools": [{"name": "missing_schema"}]}},
    {"content": [{"type": "image", "data": "not text"}]},
    {"content": []},
])
def test_errors_and_malformed_payloads_do_not_become_empty_success(payload):
    with pytest.raises((ValueError, RuntimeError)):
        search_tools(payload)


def test_empty_results_are_distinct_from_invalid_results():
    assert search_tools({"structuredContent": {"tools": []}}) == []


def test_duplicate_tools_are_rejected():
    tool = {"name": "duplicate", "inputSchema": {"type": "object"}}
    with pytest.raises(ValueError, match="duplicate"):
        search_tools({"structuredContent": {"tools": [tool, tool]}})


class NumericTimeoutSession:
    """Model MCP 2.1.1's numeric timeout contract without opening a transport."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def call_tool(self, name, *, arguments, read_timeout_seconds):
        self.calls.append((name, arguments, read_timeout_seconds))
        if not isinstance(read_timeout_seconds, (int, float)):
            raise TypeError("read_timeout_seconds must be numeric seconds")
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def read_only_call():
    target = "SupportTools"
    definitions = gateway.canonical_catalog(target)
    proposed = {
        "name": f"{target}___get_return_policy",
        "input": {"product_category": "laptops"},
        "toolUseId": "read-only-test",
    }
    return proposed, definitions, target


@pytest.mark.parametrize("options", [{}, {"timeout_seconds": 2.5}], ids=["default", "fractional"])
def test_execute_passes_numeric_timeout_and_preserves_result(read_only_call, options):
    proposed, definitions, target = read_only_call
    payload = {"statusCode": 200, "body": "Return Window: 30 days"}
    session = NumericTimeoutSession(CallToolResult(content=[], structuredContent=payload))

    record = asyncio.run(gateway.execute_read_only(session, proposed, definitions, target, **options))

    assert record["status"] == "ok", record
    assert record["attempted"] is True
    assert record["text"] == payload["body"]
    assert record["payload"]["structuredContent"] == payload
    assert session.calls == [(proposed["name"], proposed["input"], options.get("timeout_seconds", 30))]


@pytest.mark.parametrize("options", [{}, {"timeout_seconds": 2.5}], ids=["default", "fractional"])
def test_discovery_passes_numeric_timeout_and_preserves_selection(read_only_call, options):
    _, definitions, _ = read_only_call
    found = [
        definitions[0],
        {"name": "Unowned___get_return_policy", "inputSchema": {"type": "object"}},
    ]
    session = NumericTimeoutSession(CallToolResult(content=[], structuredContent={"tools": found}))

    record = asyncio.run(gateway.discover_read_only(session, "laptop returns", definitions, **options))

    assert record["status"] == "ok", record
    assert record["found"] == found
    assert record["selected"] == [definitions[0]]
    assert session.calls == [
        (gateway.SEARCH_TOOL, {"query": "laptop returns"}, options.get("timeout_seconds", 30)),
    ]


def test_refused_execution_never_calls_transport(read_only_call):
    proposed, definitions, target = read_only_call
    proposed["input"] = {"product_category": "outside-allowlist"}
    session = NumericTimeoutSession()

    record = asyncio.run(gateway.execute_read_only(session, proposed, definitions, target))

    assert record["status"] == "refused"
    assert record["attempted"] is False
    assert session.calls == []


def test_numeric_timeouts_keep_execution_and_discovery_failures_visible(read_only_call):
    proposed, definitions, target = read_only_call
    session = NumericTimeoutSession(error=TimeoutError("MCP read timed out"))

    execution = asyncio.run(gateway.execute_read_only(session, proposed, definitions, target, timeout_seconds=7))
    discovery = asyncio.run(gateway.discover_read_only(session, "laptops", definitions, timeout_seconds=7))

    assert execution["status"] == "timeout"
    assert execution["attempted"] is True
    assert "unknown" in execution["error"]
    assert discovery["status"] == "error"
    assert discovery["selected"] == []
    assert discovery["error"] == "TimeoutError: MCP read timed out"
    assert len(session.calls) == 2


def test_cleanup_deletes_owned_target_before_gateway_and_is_idempotent():
    absent = RuntimeError("already deleted")
    absent.response = {"Error": {"Code": "ResourceNotFoundException"}}
    admin = Mock()
    admin.get_gateway_target.side_effect = absent
    admin.get_gateway.side_effect = absent
    owned = {"owned-gateway": ["owned-target"]}

    assert gateway.cleanup_owned_gateways(admin, owned) == ["owned-gateway"]
    assert owned == {}
    assert admin.mock_calls == [
        call.delete_gateway_target(gatewayIdentifier="owned-gateway", targetId="owned-target"),
        call.get_gateway_target(gatewayIdentifier="owned-gateway", targetId="owned-target"),
        call.delete_gateway(gatewayIdentifier="owned-gateway"),
        call.get_gateway(gatewayIdentifier="owned-gateway"),
    ]
    admin.reset_mock()
    assert gateway.cleanup_owned_gateways(admin, owned) == []
    assert admin.mock_calls == []


@pytest.mark.parametrize("pending", ["target", "gateway"])
def test_cleanup_timeout_retains_unfinished_owned_ids(monkeypatch, pending):
    admin = Mock()
    admin.get_gateway_target.return_value = {"status": "DELETING"}
    admin.get_gateway.return_value = {"status": "DELETING"}
    owned = {"owned-gateway": ["owned-target"] if pending == "target" else []}
    expected_owned = {name: list(targets) for name, targets in owned.items()}
    clock = SimpleNamespace(monotonic=Mock(side_effect=[0, 2]), sleep=Mock())
    monkeypatch.setattr(gateway, "time", clock)

    with pytest.raises(TimeoutError, match="owned IDs retained"):
        gateway.cleanup_owned_gateways(admin, owned, timeout_seconds=1, interval=0.1)

    assert owned == expected_owned
    clock.sleep.assert_not_called()
    if pending == "target":
        admin.delete_gateway_target.assert_called_once_with(
            gatewayIdentifier="owned-gateway", targetId="owned-target",
        )
        admin.delete_gateway.assert_not_called()
    else:
        admin.delete_gateway_target.assert_not_called()
        admin.delete_gateway.assert_called_once_with(gatewayIdentifier="owned-gateway")
