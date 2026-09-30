from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

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
