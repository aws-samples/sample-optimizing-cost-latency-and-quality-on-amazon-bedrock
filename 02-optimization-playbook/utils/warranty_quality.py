"""Local evidence and result checks for the warranty lesson; no AWS calls."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def save_state(path, state):
    """Replace the saved file only after the complete JSON has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def native_trace(documents, session_id):
    """Select one real Strands request and its native children."""
    roots = [
        doc for doc in documents
        if doc.get("scope", {}).get("name") == "strands.telemetry.tracer"
        and doc.get("attributes", {}).get("gen_ai.operation.name") == "invoke_agent"
        and doc.get("attributes", {}).get("session.id") == session_id
    ]
    if len(roots) != 1 or not roots[0].get("events"):
        raise ValueError("Expected one Strands agent span with conversation events.")
    trace_id = roots[0]["traceId"]
    spans = [
        doc for doc in documents
        if doc.get("traceId") == trace_id
        and doc.get("scope", {}).get("name") == "strands.telemetry.tracer"
    ]
    return trace_id, json.loads(json.dumps(spans, allow_nan=False))


def prepare_evaluation_retry(row, evaluator_id):
    """Retry service errors on cell rerun, retaining evidence and limiting attempts."""
    response = row.get("evaluation")
    if not response or not any(
        result.get("errorCode") or result.get("errorMessage")
        for result in response.get("evaluationResults", [])
        if isinstance(result, dict)
    ):
        return False
    if row.get("evaluated_by") != evaluator_id:
        raise ValueError("Saved evaluation belongs to a different evaluator.")
    attempts = row.setdefault("evaluation_attempts", [])
    if len(attempts) >= 2:
        raise ValueError("Evaluator failed three attempts. Inspect the saved responses before retrying.")
    attempts.append({
        "response": copy.deepcopy(response),
        "seconds": row.get("evaluation_seconds", 0),
        "evaluator_id": evaluator_id,
    })
    row["evaluation"] = None
    row["evaluation_validated"] = False
    row["evaluation_cost_usd"] = None
    row.pop("score", None)
    return True


def checked_score(response, row, evaluator_id):
    """Reject missing, ambiguous or unrelated scores; keep the raw response outside."""
    results = response.get("evaluationResults", [])
    if len(results) != 1:
        raise ValueError("Expected one evaluator result for this answer.")
    result = results[0]
    context = result.get("context", {}).get("spanContext", {})
    labels = {0: "Incorrect", 1: "Incomplete", 2: "Meets policy"}
    value = result.get("value")
    if (
        result.get("errorCode") or result.get("errorMessage")
        or result.get("ignoredReferenceInputFields")
        or result.get("evaluatorId") != evaluator_id
        or context.get("traceId") != row["trace_id"]
        or context.get("sessionId") != row["session_id"]
        or isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value not in labels
        or result.get("label") != labels[value]
        or not isinstance(result.get("explanation"), str)
        or not result["explanation"].strip()
    ):
        raise ValueError("No usable policy score for this answer; inspect its saved evaluation.")
    return result


def complete_cost(rows, key="cost_usd"):
    values = [row.get(key) for row in rows]
    if any(value is None for value in values):
        return None
    return sum(values)


def comparison_tables(rows, cases):
    """Require the complete paired comparison and preserve evaluator failures."""
    expected = {(variant, case["id"]) for variant in ("baseline", "recommended") for case in cases}
    actual = [(row["variant"], row["case_id"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Finish both prompts on every held-out case before comparing totals.")
    if any(not row.get("evaluation_validated") for row in rows):
        raise ValueError("Every held-out answer needs a validated policy evaluation.")
    summary, quality = [], []
    for variant in ("baseline", "recommended"):
        attempts = [row for row in rows if row["variant"] == variant]
        passed = sum(row["score"]["value"] == 2 and row["stop_reason"] == "end_turn" for row in attempts)
        cost = complete_cost(attempts)
        summary.append({
            "Prompt": variant, "Meets policy": f"{passed}/{len(attempts)}",
            "Complete": sum(row["stop_reason"] == "end_turn" for row in attempts),
            "Answer cost (USD)": cost,
            "USD / passing answer": cost / passed if cost is not None and passed else None,
            "Mean answer time (s)": sum(row["seconds"] for row in attempts) / len(attempts),
        })
    for case in cases:
        pair = {row["variant"]: row for row in rows if row["case_id"] == case["id"]}
        quality.append({
            "Case": case["id"], "Question": case["question"],
            "Baseline": pair["baseline"]["score"]["label"],
            "Recommended": pair["recommended"]["score"]["label"],
            "New regression": (
                pair["baseline"]["score"]["value"] == 2
                and (pair["recommended"]["score"]["value"] < 2
                     or pair["recommended"]["stop_reason"] != "end_turn")
            ),
        })
    return summary, quality
