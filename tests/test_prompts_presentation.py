"""Run the Prompts101 setup and local validation with injected offline clients."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from workshop_utils.bedrock import build_converse_request
from workshop_utils.models import Support, UnsupportedFeatureError, resolve_model

NOTEBOOK = Path(__file__).parents[1] / "01-fundamentals/01-prompts-101.ipynb"
CELLS = {cell["id"]: "".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text())["cells"]}


@pytest.mark.parametrize("fallback", [False, True])
def test_preflight_summary_distinguishes_fresh_cached_and_fallback_evidence(
    tmp_path, monkeypatch, capsys, fallback,
):
    runtime, sts, display = Mock(), Mock(), Mock()
    runtime.meta.region_name = "us-east-1"
    runtime.meta.endpoint_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
    denial = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "fixture"}}, "Converse")
    runtime.converse.side_effect = [denial, {}, {}] if fallback else [{}, {}]
    sts.get_caller_identity.return_value = {
        "Account": "111111111111", "Arn": "arn:aws:sts::111111111111:assumed-role/notebook/fixture",
    }
    client = Mock(side_effect=lambda service, **kwargs: sts if service == "sts" else runtime)
    monkeypatch.setattr("boto3.client", client)
    monkeypatch.setattr("dotenv.load_dotenv", Mock())
    monkeypatch.setattr("workshop_utils.pacing.attach_boto3", Mock())
    monkeypatch.setattr("workshop_utils.pacing.get_workshop_pacer", Mock())
    monkeypatch.setattr("IPython.display.display", display)
    monkeypatch.delenv("WORKSHOP_MODEL_ID", raising=False)
    namespace = {"REPO": tmp_path}
    attempts = 3 if fallback else 2
    model_id = "global.anthropic.claude-sonnet-4-6" if fallback else "global.anthropic.claude-sonnet-5"
    for cached in (False, True):
        exec(compile(CELLS["cell-004"], str(NOTEBOOK), "exec"), namespace)
        output = capsys.readouterr().out
        assert f"Workhorse role OK: {model_id}" in output
        assert "Small role OK: global.anthropic.claude-haiku-4-5-20251001-v1:0" in output
        assert (f"{attempts} cached; 0 new attempts" if cached else f"0 cached; {attempts} new attempts") in output
        assert ("Fallback:" in output) == fallback
        if fallback:
            assert f"unavailable (AccessDeniedException); selected {model_id}" in output
        assert "111111111111" not in output
        report = namespace["PREFLIGHT_REPORT"]
        assert report == namespace["PREFLIGHT"].as_dict()
        assert namespace["EXPERIMENT"]["setup_preflight"] is report
        assert namespace["MODEL_ID"] == model_id
        assert all(probe["cache_hit"] == cached for probe in report["probes"])
        assert runtime.converse.call_count == attempts  # Cached setup sends no new probes.
        display.assert_not_called()  # Full records are retained, not dumped.
    assert sts.get_caller_identity.call_count == 2
    assert runtime.close.call_count == sts.close.call_count == 2


def test_parameter_exercise_builds_gpt_request_without_invoking_it(capsys):
    selected = resolve_model("global.anthropic.claude-sonnet-5", region="us-east-1")
    namespace = {
        "SELECTED": selected, "REGION": "us-east-1",
        "request": {"messages": [{"role": "user", "content": [{"text": "Fixture."}]}]},
        "Support": Support, "UnsupportedFeatureError": UnsupportedFeatureError,
        "build_converse_request": build_converse_request, "resolve_model": resolve_model,
        "display": Mock(), "runtime": object(),  # Any attempted inference fails.
    }
    exec(compile(CELLS["cell-027"], str(NOTEBOOK), "exec"), namespace)
    assert "Expected validation failure:" in capsys.readouterr().out
    request = namespace["gpt_request"]
    assert request["modelId"] == "global.openai.gpt-5.6-luna"
    assert request["additionalModelRequestFields"] == {"reasoning": {"effort": "low"}}
    assert request["inferenceConfig"] == {"maxTokens": 512}
    namespace["display"].assert_called_once_with(request)
