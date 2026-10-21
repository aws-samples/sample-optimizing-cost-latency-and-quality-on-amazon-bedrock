from __future__ import annotations

import ast
import json
from pathlib import Path


def test_credit_card_masks_and_other_configured_checks_still_block():
    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / "03-developer-journey/05-guardrails.ipynb").read_text()
    )
    cell = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "guardrail_config = {" in "".join(c["source"])
    )
    assignments = [n for n in ast.parse(cell).body if isinstance(n, ast.Assign)]
    namespace = {}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "guardrail_config", "exec"), namespace)
    config = namespace["guardrail_config"]
    assert {p["type"]: p["action"] for p in config["sensitiveInformationPolicyConfig"]["piiEntitiesConfig"]} == {
        "CREDIT_DEBIT_CARD_NUMBER": "ANONYMIZE",
        "US_SOCIAL_SECURITY_NUMBER": "BLOCK",
    }
    card = config["sensitiveInformationPolicyConfig"]["piiEntitiesConfig"][0]
    assert card["inputAction"] == card["outputAction"] == "ANONYMIZE"
    assert card["inputEnabled"] is True and card["outputEnabled"] is True
    assert {t["name"] for t in config["topicPolicyConfig"]["topicsConfig"]} == {
        "competitor-products",
        "investment-advice",
        "medical-advice",
    }
    assert all(t["type"] == "DENY" for t in config["topicPolicyConfig"]["topicsConfig"])
    assert {
        t["type"]: (t["inputStrength"], t["outputStrength"]) for t in config["contentPolicyConfig"]["filtersConfig"]
    } == {
        "VIOLENCE": ("HIGH", "HIGH"),
        "HATE": ("HIGH", "HIGH"),
        "INSULTS": ("MEDIUM", "MEDIUM"),
    }


def test_masked_text_is_forwarded_and_missing_mask_output_stops_processing():
    from types import SimpleNamespace

    import pytest

    source = (Path(__file__).resolve().parents[1] / "03-developer-journey/agents/v5_guardrails.py").read_text()
    fn = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "guardrail_blocks_input"
    )
    response = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [{"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "ANONYMIZED"}]
                }
            }
        ],
        "outputs": [{"text": "pay with {CREDIT_DEBIT_CARD_NUMBER}"}],
    }
    namespace = {
        "GUARDRAIL_ID": "test",
        "GUARDRAIL_VERSION": "DRAFT",
        "_bedrock_runtime": SimpleNamespace(apply_guardrail=lambda **kwargs: response),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "guardrail_gate", "exec"), namespace)
    blocked, policies, model_input = namespace["guardrail_blocks_input"]("pay with 4111-1111-1111-1111")
    assert not blocked and policies == ["pii:CREDIT_DEBIT_CARD_NUMBER:ANONYMIZED"]
    assert model_input == "pay with {CREDIT_DEBIT_CARD_NUMBER}"
    response["outputs"] = []
    with pytest.raises(ValueError, match="without returning"):
        namespace["guardrail_blocks_input"]("pay with 4111-1111-1111-1111")
