"""HIGH playbook: bounded discovery, read-only execution, and owned-ID cleanup."""

from __future__ import annotations

import asyncio
import copy
import json
import time
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path

from jsonschema import ValidationError, validate
from mcp.types import Tool

READ_ONLY_ARGUMENTS = {"get_return_policy": "product_category", "get_product_info": "product_type"}
ALLOWED_CATEGORIES = frozenset({"laptops", "smartphones", "tablets", "audio", "accessories"})
SEARCH_TOOL = "x_amz_bedrock_agentcore_search"


def support_fixture() -> dict:
    """Snapshot of the two deterministic Journey Lambda contracts, not MEDIUM's tools."""
    return json.loads(Path(__file__).with_name("high_support_fixture.json").read_text())


def canonical_catalog(target_name: str = "") -> list[dict]:
    tools = support_fixture()["tools"]
    for tool in tools:
        if target_name:
            tool["name"] = f"{target_name}___{tool['name']}"
    return tools


def restrict_catalog(definitions: list[dict], target_name: str) -> list[dict]:
    """Exact owned-target names AND schemas; never authorize by suffix or search rank."""
    trusted = {t["name"]: t for t in canonical_catalog(target_name)}
    result = []
    seen = set()
    for definition in definitions:
        name = definition["name"]
        if name not in trusted:
            continue
        if name in seen:
            raise ValueError(f"Duplicate catalog name: {name}")
        if definition["inputSchema"] != trusted[name]["inputSchema"]:
            raise ValueError(f"Backend schema differs from the reviewed contract: {name}")
        seen.add(name)
        # Use the reviewed description; discovery cannot inject new instructions.
        result.append(copy.deepcopy(trusted[name]))
    return result


def searched_catalog(full: list[dict], found: list[dict]) -> list[dict]:
    """Discovery changes exposure only. Ignore unknown definitions; retain the full-list schema."""
    found_names = {tool["name"] for tool in found}
    return [copy.deepcopy(tool) for tool in full if tool["name"] in found_names]


def converse_tools(definitions: list[dict]) -> dict:
    """An empty search result means no toolConfig, not an invalid empty tools array."""
    if not definitions:
        return {}
    return {"tool_config": {"tools": [
        {"toolSpec": {"name": t["name"], "description": t["description"], "inputSchema": {"json": t["inputSchema"]}}}
        for t in definitions
    ]}}


def validate_call(proposed: dict, definitions: list[dict], target_name: str) -> None:
    """Exposure, allowlist, exact argument keys, JSON schema, and value bounds all apply."""
    trusted = {tool["name"]: tool for tool in restrict_catalog(definitions, target_name)}
    name = proposed.get("name")
    if name not in trusted:
        raise ValueError("Tool is not in the reviewed, exposed read-only catalog")
    base = name.removeprefix(target_name + "___")
    argument = READ_ONLY_ARGUMENTS[base]
    args = proposed.get("input")
    if not isinstance(args, dict) or set(args) != {argument}:
        raise ValueError("Exactly the canonical argument is required")
    validate(args, trusted[name]["inputSchema"])
    if args[argument] not in ALLOWED_CATEGORIES:
        raise ValueError("Category is outside the bounded fixture allowlist")


def result_text(payload: dict) -> str:
    """Require usable MCP content and unwrap the actual Lambda statusCode/body envelope."""
    if payload.get("isError"):
        raise ValueError("MCP isError")
    value = payload.get("structuredContent")
    if value is None:
        content = payload.get("content")
        if not isinstance(content, list) or not content or any(
            not isinstance(block, dict) or block.get("type") != "text" or not isinstance(block.get("text"), str)
            for block in content
        ):
            raise ValueError("Missing or unsupported tool result content")
        text = "\n".join(block["text"] for block in content)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
    if isinstance(value, dict):
        if value.get("statusCode") != 200:
            raise ValueError("Missing or unsuccessful Lambda statusCode")
        value = value.get("body")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing tool result text")
    return value


async def execute_read_only(session, proposed: dict, definitions: list[dict], target_name: str,
                            *, timeout_seconds: float = 30) -> dict:
    """One bounded invocation; refusals never reach the transport and failures stay visible."""
    started = time.perf_counter()
    record = {"name": proposed.get("name"), "arguments": proposed.get("input"), "attempted": False}
    try:
        validate_call(proposed, definitions, target_name)
    except (ValueError, TypeError, KeyError, ValidationError) as exc:
        record.update(status="refused", error=str(exc))
    else:
        record["attempted"] = True
        try:
            result = await asyncio.wait_for(
                session.call_tool(proposed["name"], arguments=proposed["input"],
                                  read_timeout_seconds=timedelta(seconds=timeout_seconds)),
                timeout=timeout_seconds,
            )
            payload = result.model_dump(by_alias=True, exclude_none=True)
            record["payload"] = payload
            record.update(text=result_text(payload), status="ok")
        except TimeoutError:
            record.update(status="timeout", error="Invocation timed out; backend outcome is unknown.")
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
    record["latency_ms"] = (time.perf_counter() - started) * 1000
    return record


def tool_result_block(proposed: dict, record: dict) -> dict:
    ok = record["status"] == "ok"
    return {"toolResult": {
        "toolUseId": proposed["toolUseId"], "status": "success" if ok else "error",
        "content": [{"text": record["text"] if ok else record["error"]}],
    }}


def check_tool_task(case: dict, final_row: dict, executions: list[dict], *,
                    target_name: str, proposed_calls: list[dict]) -> dict:
    """Separate model omissions, backend failures, unnecessary calls, and final-answer quality."""
    expected = {f"{target_name}___{name}": args for name, args in case["required_calls"].items()}
    requested = {
        name for name, args in expected.items()
        if any(p.get("name") == name and p.get("input") == args for p in proposed_calls)
    }
    succeeded = {
        name for name, args in expected.items()
        if any(r["name"] == name and r["arguments"] == args and r["status"] == "ok" for r in executions)
    }
    try:
        answer = json.loads(final_row.get("text", ""))
    except (json.JSONDecodeError, TypeError):
        answer = None
    correct = answer == case["expected"] and isinstance(answer, dict) and answer.get("approved") is False
    omissions = sorted(set(expected) - requested)
    missing_results = sorted(set(expected) - succeeded)
    # These cases ask only about the deterministic laptop fixtures. A status=ok
    # response with unrelated/error text is not evidence for the expected answer.
    fixture = support_fixture()["laptop_results"]
    mismatches = sorted(
        name for name in succeeded
        if not any(
            r["name"] == name and r["arguments"] == expected[name] and r["status"] == "ok"
            and r.get("text", "").strip() == fixture[name.removeprefix(target_name + "___")].strip()
            for r in executions
        )
    )
    unexpected = [p.get("name") for p in proposed_calls if p.get("name") not in expected]
    failures = [r for r in executions if r["status"] != "ok"]
    return {
        "model_request_omissions": omissions, "missing_successful_results": missing_results,
        "backend_fixture_mismatches": mismatches,
        "unexpected_requests": unexpected, "tool_failures": len(failures),
        "answer_correct": correct, "complete": final_row.get("stop_reason") == "end_turn",
        "passed": correct and final_row.get("stop_reason") == "end_turn"
        and not (omissions or missing_results or mismatches or unexpected or failures),
    }


def search_tools(payload: Mapping) -> list[dict]:
    """Accept structured content or one JSON text block, retaining schema errors."""
    if payload.get("isError"):
        raise RuntimeError("Gateway semantic search returned an error")
    structured = payload.get("structuredContent")
    if structured is None:
        content = payload.get("content", [])
        if (
            not isinstance(content, list)
            or len(content) != 1
            or not isinstance(content[0], dict)
            or content[0].get("type") != "text"
            or not isinstance(content[0].get("text"), str)
        ):
            raise ValueError("Expected structured search results or one JSON text block")
        structured = json.loads(content[0]["text"])
    if not isinstance(structured, dict) or not isinstance(structured.get("tools"), list):
        raise ValueError("Gateway search result must contain a tools list")
    tools = [Tool.model_validate(item).model_dump(by_alias=True, exclude_none=True) for item in structured["tools"]]
    if len({item["name"] for item in tools}) != len(tools):
        raise ValueError("Gateway search returned duplicate tool names")
    return tools


async def discover_read_only(session, query: str, full: list[dict], *, timeout_seconds: float = 30) -> dict:
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            session.call_tool(SEARCH_TOOL, arguments={"query": query},
                              read_timeout_seconds=timedelta(seconds=timeout_seconds)),
            timeout=timeout_seconds,
        )
        payload = result.model_dump(by_alias=True, exclude_none=True)
        found = search_tools(payload)
        return {"status": "ok", "found": found, "selected": searched_catalog(full, found),
                "payload": payload, "latency_ms": (time.perf_counter() - started) * 1000}
    except Exception as exc:
        return {"status": "error", "selected": [], "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": (time.perf_counter() - started) * 1000}


async def verify_read_only_backend(session, definitions: list[dict], target_name: str) -> dict:
    """Check both deployed read-only contracts before spending on model comparisons."""
    executions, checks = [], {}
    for name, argument in READ_ONLY_ARGUMENTS.items():
        execution = await execute_read_only(
            session,
            {"name": f"{target_name}___{name}", "input": {argument: "laptops"},
             "toolUseId": f"readiness-{name}"},
            definitions, target_name,
        )
        executions.append(execution)
        checks[name] = (
            execution["status"] == "ok"
            and execution["text"].strip() == support_fixture()["laptop_results"][name].strip()
        )
    return {"verified": all(checks.values()), "tools": checks, "executions": executions,
            "latency_ms": sum(result["latency_ms"] for result in executions)}


def cleanup_owned_gateways(admin, owned: dict, *, timeout_seconds: float = 180, interval: float = 5) -> list[str]:
    """Delete only IDs recorded from this notebook's create responses; retain unfinished IDs."""
    if timeout_seconds <= 0 or interval <= 0:
        raise ValueError("Cleanup limits must be positive")
    deadline = time.monotonic() + timeout_seconds

    def absent_or_call(method, **kwargs):
        try:
            return method(**kwargs)
        except Exception as exc:
            if getattr(exc, "response", {}).get("Error", {}).get("Code") == "ResourceNotFoundException":
                return None
            raise

    def wait_deleted(fetch, **kwargs):
        while absent_or_call(fetch, **kwargs) is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Deletion still pending; owned IDs retained. Retry cleanup.")
            time.sleep(min(interval, remaining))

    deleted = []
    for gateway_id, targets in list(owned.items()):
        for target_id in list(targets):
            kwargs = {"gatewayIdentifier": gateway_id, "targetId": target_id}
            absent_or_call(admin.delete_gateway_target, **kwargs)
            wait_deleted(admin.get_gateway_target, **kwargs)
            targets.remove(target_id)
        kwargs = {"gatewayIdentifier": gateway_id}
        absent_or_call(admin.delete_gateway, **kwargs)
        wait_deleted(admin.get_gateway, **kwargs)
        del owned[gateway_id]
        deleted.append(gateway_id)
    return deleted
