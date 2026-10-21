"""Decode Gateway search results without changing which tools are selected."""

from __future__ import annotations

import json


def gateway_search_tools(result):
    """Accept structured MCP output or the equivalent JSON text output."""
    if result.get("isError") or result.get("status") == "error":
        details = "\n".join(b.get("text", "") for b in result.get("content", []))
        raise RuntimeError(f"Gateway search failed: {details}")
    payload = result.get("structuredContent")
    if payload is None:
        text = "\n".join(b["text"] for b in result.get("content", []) if "text" in b)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("Gateway search did not return a JSON tools object") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
        raise ValueError("Gateway search did not return a tools list")
    tools = payload["tools"]
    if any(
        not isinstance(t, dict)
        or not isinstance(t.get("name"), str)
        or not t["name"]
        or not isinstance(t.get("inputSchema"), dict)
        for t in tools
    ):
        raise ValueError("Gateway search returned an invalid tool definition")
    return tools
