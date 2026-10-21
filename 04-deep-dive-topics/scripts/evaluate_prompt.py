"""
Evaluation script for CI/CD pipeline integration.

Fetches a dataset from Langfuse, runs each item through an agent via
dataset.run_experiment() with keyword + LLM-as-Judge evaluators. Results
are logged as a Langfuse dataset run and written to a local JSON file
for the quality gate.

Usage:
    python scripts/evaluate_prompt.py

Environment variables required:
    LANGFUSE_PUBLIC_KEY  - Langfuse public key
    LANGFUSE_SECRET_KEY  - Langfuse secret key
    LANGFUSE_BASE_URL    - Langfuse host URL (default: https://cloud.langfuse.com)
    AWS_DEFAULT_REGION   - AWS region for Bedrock (default: us-east-1)

Optional CI/CD environment variables (auto-detected):
    CODEBUILD_BUILD_NUMBER - Build number (used in experiment run name)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import boto3
import httpx

try:
    from langfuse_text_transport import TextDatasetTransport
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.langfuse_text_transport import TextDatasetTransport

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_NAME = os.environ.get("EVAL_DATASET_NAME", "customer-support-eval")
SCORE_THRESHOLD = float(os.environ.get("EVAL_SCORE_THRESHOLD", "0.7"))
PASS_RATE_THRESHOLD = float(os.environ.get("EVAL_PASS_RATE_THRESHOLD", "0.8"))
MODEL_ID = os.environ.get("EVAL_MODEL_ID", "global.anthropic.claude-sonnet-5")
JUDGE_MODEL_ID = os.environ.get("EVAL_JUDGE_MODEL_ID", "global.anthropic.claude-opus-5")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
PROMPT_NAME = os.environ.get("EVAL_PROMPT_NAME", "customer-support-system")
PROMPT_LABEL = os.environ.get("EVAL_PROMPT_LABEL", "draft")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_langfuse():
    """Return a Langfuse client or None if credentials are missing."""
    try:
        from langfuse import Langfuse

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        host = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            print("[ERROR] LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set.")
            return None

        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            httpx_client=httpx.Client(transport=TextDatasetTransport(host), timeout=30),
        )
    except ImportError:
        print("[ERROR] langfuse package not installed.")
        return None


def _inference_config(model, max_tokens):
    config = {"maxTokens": max_tokens}
    if "claude-sonnet-5" not in model and "claude-opus-5" not in model:
        config["temperature"] = 0.0
    return config


def _reasoning_config(model):
    if "claude-sonnet-5" in model or "claude-opus-5" in model:
        return {"additionalModelRequestFields": {"output_config": {"effort": "low"}}}
    return {}


def _call_agent(query: str, system_prompt: str, bedrock_client) -> dict:
    """Call the agent under test via Bedrock Converse API."""
    response = bedrock_client.converse(
        modelId=MODEL_ID,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": query}]}],
        inferenceConfig=_inference_config(MODEL_ID, 500),
        **_reasoning_config(MODEL_ID),
    )
    if response["stopReason"] != "end_turn":
        raise RuntimeError(f"Incomplete model answer: {response['stopReason']}")
    usage = response["usage"]
    return {
        "text": "".join(b["text"] for b in response["output"]["message"]["content"] if "text" in b),
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
    }


def _calculate_keyword_score(response: str, expected: dict) -> float:
    """Calculate accuracy score based on expected output keywords."""
    score = 0.0
    checks = 0

    if "should_contain" in expected:
        for keyword in expected["should_contain"]:
            checks += 1
            if keyword.lower() in response.lower():
                score += 1

    if "should_use_tool" in expected:
        checks += 1
        if expected["should_use_tool"].replace("_", " ").lower() in response.lower():
            score += 1

    return score / checks if checks > 0 else 1.0


def _evaluate_with_llm_judge(query: str, response: str, bedrock_client) -> dict:
    """Use an LLM judge to score helpfulness on a 0-10 scale."""
    eval_prompt = f"""Rate the helpfulness of this customer support response on a scale of 0 to 10.

Customer Query: {query}

Agent Response: {response}

Scoring criteria:
- 0-3: Unhelpful, incorrect, or confusing
- 4-6: Partially helpful, missing key information
- 7-9: Helpful, accurate, addresses the query well
- 10: Exceptional, exceeds expectations

Return ONLY a JSON object: {{"score": <number>, "reason": "<one sentence>"}}"""

    try:
        judge_response = bedrock_client.converse(
            modelId=JUDGE_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": eval_prompt}]}],
            inferenceConfig=_inference_config(JUDGE_MODEL_ID, 300),
            **_reasoning_config(JUDGE_MODEL_ID),
        )
        if judge_response["stopReason"] != "end_turn":
            raise RuntimeError(f"Incomplete judge answer: {judge_response['stopReason']}")
        raw = "".join(b["text"] for b in judge_response["output"]["message"]["content"] if "text" in b).strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}', raw)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                return {"score": 0.0, "reason": f"No JSON found in: {raw[:80]}"}

        return {
            "score": float(parsed.get("score", 0)) / 10.0,
            "reason": parsed.get("reason", ""),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": 0.0, "reason": "Failed to parse judge response"}


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------


def run_evaluation() -> dict:
    """Run the full evaluation pipeline using dataset.run_experiment()."""
    from langfuse import Evaluation

    bedrock = boto3.client("bedrock-runtime", region_name=REGION)
    langfuse = _init_langfuse()

    if langfuse is None:
        return {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "avg_score": 0.0,
            "pass_rate": 0.0,
        }

    # Fetch the prompt under test from Langfuse (default: "draft" label)
    prompt_obj = langfuse.get_prompt(PROMPT_NAME, label=PROMPT_LABEL)
    system_prompt = prompt_obj.compile()
    print(f"Evaluating prompt: {PROMPT_NAME} (v{prompt_obj.version}, label={PROMPT_LABEL})")
    print(f"  Text: {system_prompt[:80]}...")

    langfuse_dataset = langfuse.get_dataset(name=DATASET_NAME)
    print(f"Loaded dataset '{DATASET_NAME}' from Langfuse ({len(langfuse_dataset.items)} items)")

    # Collect local results via closure
    results: list[dict] = []
    judge_cache: dict[str, dict] = {}

    build_number = os.environ.get("CODEBUILD_BUILD_NUMBER", "local")
    run_name = f"ci-eval-{build_number}"

    def task(*, item, **kwargs):
        query = item.input.get("query", "")
        expected = item.expected_output or {}
        item_id = (item.metadata or {}).get("id", f"item-{len(results)}")

        # Report model usage to Langfuse for cost tracking
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="bedrock-converse",
            model=MODEL_ID,
            input=query,
        ) as gen:
            agent_result = _call_agent(query, system_prompt, bedrock)
            agent_response = agent_result["text"]
            gen.update(
                output=agent_response,
                usage_details={
                    "input": agent_result["input_tokens"],
                    "output": agent_result["output_tokens"],
                },
            )

        # Pre-compute scores for local results
        keyword_score = _calculate_keyword_score(agent_response, expected)
        judge_result = _evaluate_with_llm_judge(query, agent_response, bedrock)
        judge_score = judge_result["score"]
        combined_score = (keyword_score + judge_score) / 2.0
        judge_cache[(query, agent_response)] = judge_result

        results.append(
            {
                "item_id": item_id,
                "keyword_score": keyword_score,
                "judge_score": judge_score,
                "combined_score": combined_score,
                "success": True,
            }
        )
        print(f"  [{len(results)}/{len(langfuse_dataset.items)}] {query[:60]}... score={combined_score:.2f}")
        return agent_response

    def keyword_evaluator(*, output, expected_output, **kwargs):
        return Evaluation(
            name="keyword_accuracy",
            value=_calculate_keyword_score(output, expected_output or {}),
        )

    def judge_evaluator(*, input, output, **kwargs):
        query = input.get("query", "")
        cached = judge_cache.get((query, output))
        if cached:
            return Evaluation(
                name="helpfulness_llm",
                value=cached["score"],
                comment=cached.get("reason", ""),
            )
        result = _evaluate_with_llm_judge(query, output, bedrock)
        return Evaluation(
            name="helpfulness_llm",
            value=result["score"],
            comment=result.get("reason", ""),
        )

    # Run experiment — traces are auto-linked to dataset items
    langfuse_dataset.run_experiment(
        name=run_name,
        task=task,
        evaluators=[keyword_evaluator, judge_evaluator],
        max_concurrency=1,
    )
    langfuse.flush()
    time.sleep(1)

    if not langfuse_dataset.items or len(results) != len(langfuse_dataset.items):
        raise RuntimeError("Incomplete dataset experiment; do not run the quality gate")

    # Calculate summary
    successful = [r for r in results if r["success"]]
    avg_score = sum(r["combined_score"] for r in successful) / len(successful) if successful else 0.0
    # Match the notebook: a completed API call is not automatically a quality pass.
    pass_rate = sum(r["combined_score"] >= 0.5 for r in results) / len(results) if results else 0.0

    # Fetch production baseline score from SSM Parameter Store for regression check
    prod_avg_score = None
    ssm_param = "/prompt-pipeline/production-baseline-score"
    try:
        ssm = boto3.client("ssm", region_name=REGION)
        param = ssm.get_parameter(Name=ssm_param)
        prod_avg_score = float(param["Parameter"]["Value"])
        print(f"\n  Production baseline avg score: {prod_avg_score:.3f} (from SSM: {ssm_param})")
    except ssm.exceptions.ParameterNotFound:
        print(f"\n  [INFO] No production baseline in SSM ({ssm_param}). Skipping regression check.")
    except Exception as e:
        print(f"\n  [WARN] Could not fetch production baseline: {e}")

    return {
        "prompt_name": PROMPT_NAME,
        "prompt_version": prompt_obj.version,
        "total": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "avg_score": avg_score,
        "pass_rate": pass_rate,
        "prod_avg_score": prod_avg_score,
        "details": results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Prompt Evaluation Pipeline")
    print("=" * 60)
    print(f"  Prompt:           {PROMPT_NAME} (label={PROMPT_LABEL})")
    print(f"  Dataset:          {DATASET_NAME}")
    print(f"  Model:            {MODEL_ID}")
    print(f"  Judge Model:      {JUDGE_MODEL_ID}")
    print(f"  Score Threshold:  {SCORE_THRESHOLD}")
    print(f"  Pass Rate Target: {PASS_RATE_THRESHOLD}")
    print("=" * 60)

    evaluation_results = run_evaluation()

    # Write results to JSON for CI artifact collection
    output_path = Path("evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\nResults written to {output_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("  EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total items:    {evaluation_results['total']}")
    print(f"  Successful:     {evaluation_results['successful']}")
    print(f"  Failed:         {evaluation_results['failed']}")
    print(f"  Average score:  {evaluation_results['avg_score']:.3f}")
    print(f"  Pass rate:      {evaluation_results['pass_rate']:.1%}")
    print(f"{'=' * 60}")

    # Quality gate check:
    # 1. Average score must meet threshold
    # 2. Pass rate must meet threshold
    # 3. Must not regress from production baseline (if available)
    avg = evaluation_results["avg_score"]
    pr = evaluation_results["pass_rate"]
    prod_avg = evaluation_results.get("prod_avg_score")

    reasons = []
    if avg < SCORE_THRESHOLD:
        reasons.append(f"avg_score={avg:.3f} < threshold {SCORE_THRESHOLD}")
    if pr < PASS_RATE_THRESHOLD:
        reasons.append(f"pass_rate={pr:.1%} < threshold {PASS_RATE_THRESHOLD:.1%}")
    if prod_avg is not None and avg < prod_avg:
        reasons.append(f"avg_score={avg:.3f} < production baseline {prod_avg:.3f} (regression)")

    if not reasons:
        msg = f"avg_score={avg:.3f} >= {SCORE_THRESHOLD}"
        if prod_avg is not None:
            msg += f", >= production baseline {prod_avg:.3f}"
        msg += f", pass_rate={pr:.1%} >= {PASS_RATE_THRESHOLD:.1%}"
        print(f"\n  QUALITY GATE PASSED: {msg}")
        sys.exit(0)
    else:
        print("\n  QUALITY GATE FAILED:")
        for reason in reasons:
            print(f"    - {reason}")
        sys.exit(1)
