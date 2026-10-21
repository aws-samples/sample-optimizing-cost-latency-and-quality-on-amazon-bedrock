"""Security version floors and compatibility of the workshop's existing APIs."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def test_security_versions_match_every_export():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    versions = {entry["name"]: entry["version"] for entry in lock["package"]}
    assert Version(versions["certifi"]) >= Version("2026.10.19")
    assert Version(versions["aiohttp"]) >= Version("3.14.4")
    # Strands supports this maintained line; the affected server code is in MCP 2.0/2.1.
    assert Version("1.30") <= Version(versions["mcp"]) < Version("2")
    for name in ("notebook", "langfuse", "runtime", "runtime-langfuse"):
        exported = dict(re.findall(r"(?m)^([\w-]+)==([^\s\\]+)", (ROOT / f"requirements-{name}.lock").read_text()))
        for package in ("certifi", "aiohttp", "mcp", "rich"):
            assert exported[package] == versions[package]
    journey = dict(
        re.findall(
            r"(?m)^([\w-]+)==([^\s\\]+)",
            (ROOT / "03-developer-journey/requirements-for-agentcore.txt").read_text(),
        )
    )
    for package in ("certifi", "aiohttp", "mcp"):
        assert journey[package] == versions[package]


def test_mcp_keeps_the_tool_shapes_used_by_gateway():
    from mcp.types import CallToolResult, TextContent, Tool

    tool = Tool(
        name="customer-support-tools___get_return_policy",
        inputSchema={"type": "object", "properties": {"category": {"type": "string"}}},
    )
    response = CallToolResult(
        content=[TextContent(type="text", text='{"return_window_days":30}')],
        structuredContent={"return_window_days": 30},
        isError=False,
    )
    assert tool.model_dump()["inputSchema"]["properties"]["category"]["type"] == "string"
    assert response.content[0].text == '{"return_window_days":30}'
    assert response.structuredContent == {"return_window_days": 30}
    assert response.isError is False


def test_patched_evaluation_stack_keeps_opus_controls_and_legacy_metrics(monkeypatch):
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    pytest.importorskip("ragas")
    from langchain_core.prompt_values import StringPromptValue

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "offline-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "offline-test")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    from utils import ragas_eval

    judge, embeddings = ragas_eval.build_bedrock_evaluator(region="us-east-1")
    assert judge.cache is None
    assert embeddings.cache is None
    requests = []

    def converse(**request):
        requests.append(request)
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": '{"claims":[]}'}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 4, "totalTokens": 14},
            "metrics": {"latencyMs": 1},
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    monkeypatch.setattr(judge.langchain_llm.client, "converse", converse)
    # The workshop metrics use RAGAS's async generate path, which honors bypass_temperature.
    result = ragas_eval._run_async(judge.generate(StringPromptValue(text="Extract claims."), temperature=0.0))
    assert result.generations[0][0].text == '{"claims":[]}'
    assert len(requests) == 1
    assert requests[0]["modelId"] == "global.anthropic.claude-opus-5"
    assert requests[0]["additionalModelRequestFields"] == {"output_config": {"effort": "low"}}
    assert "temperature" not in requests[0].get("inferenceConfig", {})
    assert "topP" not in requests[0].get("inferenceConfig", {})
    for metric in (
        ragas_eval.FactualCorrectness,
        ragas_eval.LLMContextPrecisionWithReference,
        ragas_eval.ToolCallAccuracy,
    ):
        assert metric.__module__.startswith("ragas.metrics._")
        assert ".collections." not in metric.__module__
    assert ragas_eval.score_tool_call_accuracy("policy", ["get_return_policy"], ["get_return_policy"]) == 1.0
