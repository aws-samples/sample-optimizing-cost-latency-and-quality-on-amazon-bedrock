from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "stable_gateway_response", Path(__file__).resolve().parents[1] / "03-developer-journey/utils/gateway_response.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TOOL = {
    "name": "customer-support-tools___get_return_policy",
    "description": "Return policy",
    "inputSchema": {"type": "object", "properties": {}},
}


@pytest.mark.parametrize(
    "response",
    [
        {"structuredContent": {"tools": [TOOL]}},
        {"status": "success", "isError": False, "content": [{"text": json.dumps({"tools": [TOOL]})}]},
    ],
)
def test_both_mcp_output_formats_preserve_the_same_tools(response):
    assert module.gateway_search_tools(response) == [TOOL]


def test_empty_result_remains_empty():
    assert module.gateway_search_tools({"structuredContent": {"tools": []}}) == []


def test_service_error_not_treated_as_tools():
    with pytest.raises(RuntimeError, match="Access denied"):
        module.gateway_search_tools(
            {"isError": True, "structuredContent": {"tools": [TOOL]}, "content": [{"text": "Access denied"}]}
        )


@pytest.mark.parametrize(
    "response",
    [
        {"content": [{"text": "not JSON"}]},
        {"structuredContent": {}},
        {"structuredContent": {"tools": [{"name": "missing-schema"}]}},
    ],
)
def test_invalid_result_cannot_be_used_for_selection(response):
    with pytest.raises(ValueError):
        module.gateway_search_tools(response)
