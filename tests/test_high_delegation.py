"""Offline delegation behavior: isolated workers, reused evidence and complete accounting."""
from __future__ import annotations

import ast
import asyncio
import copy
import importlib.util
import json
import re
import sys
import time
import uuid
from html import escape
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from botocore.config import Config

from workshop_utils.metrics import metrics_records
from workshop_utils.models import Support

ROOT = Path(__file__).resolve().parents[1]
SECTION = ROOT / "02-optimization-playbook"
NOTEBOOK = SECTION / "03-high-effort.ipynb"
spec = importlib.util.spec_from_file_location("delegation_test_helper", SECTION / "utils/delegation_experiment.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


async def run_cell(cell_id, namespace):
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    source = "".join(next(cell["source"] for cell in cells if cell["id"] == cell_id))
    tree = ast.parse(source)
    # Retain the pure lesson helper import; inject scripted Strands/SDK objects.
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
                 or (isinstance(node, ast.ImportFrom) and node.module == "utils.delegation_experiment")]
    with patch.dict(sys.modules, {"utils": ModuleType("utils"), "utils.delegation_experiment": helper}):
        result = eval(compile(tree, cell_id, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), namespace)
        if result is not None:
            await result


def scenario(*, worker_mode="normal", price_missing=False, no_spans=False):
    documents, requests, agents, printed = [], [], [], []
    arrived, gate = [], asyncio.Event()

    def record_call(config, inputs):
        if no_spans:
            return
        trace_id = uuid.uuid4().hex
        root = {
            "traceId": trace_id, "spanId": "1" * 16,
            "scope": {"name": "strands.telemetry.tracer"},
            "startTimeUnixNano": 1_000_000, "endTimeUnixNano": 2_000_000,
            "attributes": {"session.id": config["trace_attributes"]["session.id"],
                           "gen_ai.operation.name": "invoke_agent"},
        }
        chat = {
            **root, "spanId": "2" * 16, "parentSpanId": root["spanId"],
            "attributes": {"gen_ai.operation.name": "chat", "gen_ai.request.model": config["model"]["model_id"],
                           "gen_ai.usage.input_tokens": inputs, "gen_ai.usage.output_tokens": 20},
        }
        sdk = {**chat, "spanId": "3" * 16, "parentSpanId": chat["spanId"],
               "scope": {"name": "opentelemetry.instrumentation.botocore.bedrock-runtime"}}
        documents.extend([root, chat, sdk])

    class Result(str):
        def __new__(cls, text, stop="end_turn"):
            result = super().__new__(cls, text)
            result.stop_reason = stop
            return result

    class ScriptedAgent:
        def __init__(self, **config):
            self.config = config
            agents.append(config)

        async def invoke_async(self, question):
            requests.append((self.config["name"], question))
            if self.config["name"].endswith("-reader"):
                arrived.append(self.config["name"])
                if len(arrived) == 2:
                    gate.set()
                # This fails if production code accidentally runs the workers sequentially.
                await asyncio.wait_for(gate.wait(), timeout=1)
                is_policy = self.config["name"] == "policy-reader"
                ids = ["P02", "P03"] if is_policy else ["C02"]
                if worker_mode == "omit_purchase_age" and not is_policy:
                    ids = ["C03"]
                if worker_mode == "bad_id":
                    ids = ["FAKE"]
                record_call(self.config, 1000)
                reply = json.dumps({"section_ids": ids})
                if worker_mode == "fenced":
                    reply = "```json\n" + reply + "\n```"
                return Result(reply, "max_tokens" if worker_mode == "incomplete" else "end_turn")
            ids = re.findall(r"^\[([PC]\d{2})\]", self.config["system_prompt"], re.MULTILINE)
            record_call(self.config, 1000 if len(ids) == 24 else 100)
            q = helper.QUESTION
            assert helper.question_prompt(q) == question
            # Even an oracle's correct answer must fail if it cites evidence absent from its context.
            return Result(json.dumps({"result": q["expected"], "source_ids": sorted(q["sources"])}))

    namespace = {
        "asyncio": asyncio, "time": time, "uuid": uuid, "pd": pd, "Config": Config,
        "Agent": ScriptedAgent, "BedrockModel": lambda **kwargs: kwargs,
        "paced_boto3_session": lambda **kwargs: object(),
        "OBS": SimpleNamespace(collector=SimpleNamespace(span_documents=lambda: list(documents))),
        "REGION": "us-east-1", "SMALL": "small-fixture", "WORKHORSE": "large-fixture",
        "REPO_ROOT": ROOT, "SECTION_ROOT": SECTION, "HTML": str, "escape": escape,
        "SDK_CONFIG": Config(retries={"total_max_attempts": 3}), "Support": Support,
        "caps": lambda model: SimpleNamespace(support=lambda feature: Support.SUPPORTED),
        "resolve_model": lambda *args, **kwargs: None,
        "build_converse_request": lambda *args, **kwargs: {"additionalModelRequestFields": {"output_config": {"effort": "low"}}},
        "get_model": lambda model: SimpleNamespace(name=model), "metrics_records": metrics_records,
        "display": lambda *args: None, "print": lambda *args: printed.append(args), "pretty_json": lambda value: None,
        "price_observed_call": lambda **kwargs: (
            None if price_missing else 0.014 if kwargs["model_id"] == "small-fixture" else kwargs["input_tokens"] * 0.00001
        ),
    }
    return namespace, agents, requests, printed


STEPS = (
    "high-delegation-task", "high-delegation-call", "high-delegation-single", "high-05-7fc0a0ee",
    "high-delegation-workers-run", "high-delegation-first-brief", "high-delegation-results",
)


def test_workers_are_parallel_isolated_and_reused_without_duplicate_billing():
    ns, agents, requests, _ = scenario()

    async def run():
        for step in STEPS:
            await run_cell(step, ns)
        await run_cell("high-delegation-workers-run", ns)
        await run_cell("high-delegation-first-brief", ns)

    asyncio.run(run())
    assert len(requests) == 4
    assert len(ns["delegation_trial"]["workers"]) == 2
    assert len(ns["delegation_trial"]["answers"]) == 2
    assert all(row["passed"] for row in ns["delegation_trial"]["answers"])
    for config in agents:
        assert config["retry_strategy"] is None
        assert config["model"]["boto_client_config"].retries["total_max_attempts"] == 1
        if config["name"] == "policy-reader":
            assert "[P02]" in config["system_prompt"] and "[C02]" not in config["system_prompt"]
        elif config["name"] == "case-reader":
            assert "[C03]" in config["system_prompt"] and "[P03]" not in config["system_prompt"]
        else:
            assert config["model"]["max_tokens"] == 512
            assert config["model"]["additional_request_fields"] == ns["WORKHORSE_FIELDS"]
            assert config["system_prompt"].startswith(helper.ANSWER_INSTRUCTIONS)
    for key, text in ns["BRIEF_SECTIONS"].items():
        assert ns["ALL_SECTIONS"][key] == text
    # The initial full-document call is reused by the paired comparison.
    assert len([name for name, _ in requests if name == "support-answer"]) == 2
    # Session filtering prevents one parallel worker from capturing the other's spans.
    assert all(len(record["model_records"]) == 1 for record in ns["delegation_trial"]["workers"])


def test_single_question_table_separates_lead_context_from_whole_workflow_cost():
    ns, _, _, _ = scenario()

    async def run():
        for step in STEPS:
            await run_cell(step, ns)

    asyncio.run(run())
    full = ns["delegation_summary"]["Full documents"]
    delegated = ns["delegation_summary"]["Delegated brief"]
    assert full["model_requests"] == 1 and delegated["model_requests"] == 3
    assert full["cost_usd"] == pytest.approx(0.01)
    assert delegated["cost_usd"] == pytest.approx(0.029)
    assert full["lead_input_tokens"] == 1000 and delegated["lead_input_tokens"] == 100
    assert full["paragraphs"] == 24 and delegated["paragraphs"] == 3
    lead = next(r for r in ns["delegation_trial"]["answers"] if r["approach"] == "Delegated brief")
    assert delegated["seconds"] == pytest.approx(ns["delegation_trial"]["prep_seconds"] + lead["seconds"])
    assert "cumulative_rows" not in ns


@pytest.mark.parametrize("kwargs", [{"price_missing": True}, {"no_spans": True}])
def test_missing_measurement_never_becomes_free_work(kwargs):
    ns, _, _, _ = scenario(**kwargs)

    async def run():
        for step in STEPS:
            await run_cell(step, ns)

    asyncio.run(run())
    assert all(row["cost_usd"] is None for row in ns["delegation_summary"].values())


@pytest.mark.parametrize("mode", ["bad_id", "incomplete"])
def test_unusable_worker_output_is_retained_and_prevents_lead_comparison(mode):
    ns, _, requests, _ = scenario(worker_mode=mode)

    async def run():
        for step in STEPS[:5]:
            await run_cell(step, ns)
        with pytest.raises(ValueError, match="Inspect the worker replies"):
            await run_cell("high-delegation-first-brief", ns)

    asyncio.run(run())
    assert len(requests) == 3
    assert all(record["error"] and record["result"] is not None for record in ns["delegation_trial"]["workers"])
    assert helper.total_cost(ns["delegation_trial"]["workers"]) == pytest.approx(0.028)


def test_missing_evidence_is_not_added_back_to_make_the_lead_pass():
    ns, _, _, _ = scenario(worker_mode="omit_purchase_age")

    async def run():
        for step in STEPS:
            await run_cell(step, ns)

    asyncio.run(run())
    assert "C02" not in ns["BRIEF_SECTIONS"]
    row = next(row for row in ns["delegation_trial"]["answers"]
               if row["approach"] == "Delegated brief" and row["question_id"] == "return-route")
    assert row["checks"]["facts_correct"]
    assert not row["checks"]["evidence_cited"] and not row["passed"]
    assert not ns["delegation_summary"]["Delegated brief"]["passed"]


def test_fenced_worker_replies_are_retained_and_accepted_without_another_model_call():
    ns, _, requests, _ = scenario(worker_mode="fenced")

    async def run():
        for step in STEPS:
            await run_cell(step, ns)

    asyncio.run(run())
    assert len(requests) == 4
    assert ns["workers_ready"]
    assert all(r["text"].startswith("```json") for r in ns["delegation_trial"]["workers"])
    assert all(r["passed"] for r in ns["delegation_trial"]["answers"])


def test_saved_parse_failures_can_be_rechecked_without_rebilling_or_losing_the_reply():
    ns, _, requests, printed = scenario(worker_mode="fenced")

    async def run():
        for step in STEPS[:5]:
            await run_cell(step, ns)
        original = [(r["text"], r["cost_usd"], r["seconds"]) for r in ns["delegation_trial"]["workers"]]
        preparation_time = ns["delegation_trial"]["prep_seconds"]
        for record in ns["delegation_trial"]["workers"]:
            record.update(ready=False, selected={}, error="Expecting value: line 1 column 1 (char 0)")
        await run_cell("high-delegation-workers-run", ns)
        assert original == [(r["text"], r["cost_usd"], r["seconds"]) for r in ns["delegation_trial"]["workers"]]
        assert preparation_time == ns["delegation_trial"]["prep_seconds"]

    asyncio.run(run())
    assert len(requests) == 3  # One baseline and the original two workers only.
    assert ns["workers_ready"] and len(ns["BRIEF_SECTIONS"]) == 3
    assert all(r["initial_error"].startswith("Expecting value") for r in ns["delegation_trial"]["workers"])
    assert any("no new worker calls" in str(line) for line in printed)


@pytest.mark.parametrize("text", [
    '{"section_ids": ["P02"]}',
    ' \n```json\n{"section_ids": ["P02"]}\n```\n',
    '```JSON\r\n{"section_ids": ["P02"]}\r\n```',
    '```\n{"section_ids": ["P02"]}\n```',
])
def test_only_the_enclosing_code_fence_is_removed(text):
    assert helper.selected_sections(text, "end_turn", {"P02": "Unchanged source"}) == {"P02": "Unchanged source"}


@pytest.mark.parametrize("text", [
    "",
    'Here is the result:\n```json\n{"section_ids":["P02"]}\n```',
    '```json\n{"section_ids":["P02"]}\n```\nExtra commentary',
    '```python\n{"section_ids":["P02"]}\n```',
    '```json\n{"section_ids":["P02"]}\n```\n```json\n{"section_ids":["P03"]}\n```',
    '```json\n{"section_ids":["C02"]}\n```',
])
def test_malformed_or_out_of_scope_fenced_output_still_fails(text):
    with pytest.raises(ValueError):
        helper.selected_sections(text, "end_turn", {"P02": "Source"})


@pytest.mark.parametrize("payload", [
    {"section_ids": []}, {"section_ids": ["P02", "P02"]}, {"section_ids": ["C02"]},
    {"section_ids": [None]}, {"section_ids": "P02"}, {"section_ids": ["P02"], "extra": "x"},
])
def test_worker_contract_does_not_repair_invalid_outputs(payload):
    with pytest.raises(ValueError):
        helper.selected_sections(json.dumps(payload), "end_turn", {"P02": "Source"})


@pytest.mark.parametrize("question", [helper.QUESTION], ids=lambda q: q["id"])
def test_answer_checks_require_typed_facts_and_real_sources(question):
    available = dict.fromkeys(question["sources"], "source")
    good = {"result": copy.deepcopy(question["expected"]), "source_ids": sorted(available)}
    assert helper.check_answer(json.dumps(good), "end_turn", question, available)["passed"]
    assert not helper.check_answer(json.dumps(good), "max_tokens", question, available)["passed"]
    assert not helper.check_answer(json.dumps(good), "end_turn", question, {})["passed"]
    assert not helper.check_answer(json.dumps({**good, "result": None}), "end_turn", question, available)["passed"]
    if isinstance(next(iter(good["result"].values())), bool):
        good["result"][next(iter(good["result"]))] = 0
        assert not helper.check_answer(json.dumps(good), "end_turn", question, available)["passed"]
