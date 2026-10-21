"""Provisioning-time checks for the API paths used by the workshop.

These are small billable setup requests, not application measurements. Run as
the Code Editor user after installing the notebook lock. No credentials or
model responses are written to the readiness report.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from .bedrock import build_converse_request
from .models import DEFAULT_ALIASES, Support, resolve_model

_ACTIVATION_MESSAGES = (
    "subscription to the model is being set up",
    "subscription is being set up",
    "agreement is being created",
    "agreement is being processed",
)
_RETRYABLE_CODES = {
    "ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException",
    "InternalServerException", "ModelNotReadyException", "ModelTimeoutException",
}


class ActivationPending(RuntimeError):
    """Invocation succeeded but the model agreement is still propagating."""


def error_state(error):
    """Never treat arbitrary permission errors as pending activation."""
    response = getattr(error, "response", {})
    detail = response.get("Error", {}) if isinstance(response, dict) else {}
    code = detail.get("Code", type(error).__name__)
    message = detail.get("Message", str(error)).lower()
    pending = isinstance(error, ActivationPending) or any(text in message for text in _ACTIVATION_MESSAGES)
    transient_http = getattr(error, "status_code", None) in {429, 500, 502, 503, 504}
    return ("pending" if pending or code in _RETRYABLE_CODES or transient_http else "failed"), code


def wait_for_readiness(checks, *, timeout_s=600, interval_s=30, clock=time.monotonic,
                       sleep=time.sleep, report=None):
    """Retry only unfinished transient checks; preserve hard failures in the report."""
    if not all(math.isfinite(value) and value > 0 for value in (timeout_s, interval_s)):
        raise ValueError("Timeout and polling interval must be positive")
    deadline = clock() + timeout_s
    states = {name: {"status": "pending", "attempts": 0} for name in checks}
    while True:
        for name, check in checks.items():
            state = states[name]
            if state["status"] != "pending":
                continue
            if clock() >= deadline:
                break
            state["attempts"] += 1
            try:
                check()
            except Exception as error:
                state["status"], state["error_code"] = error_state(error)
            else:
                state.update(status="ready")
                state.pop("error_code", None)
        if report:
            report(states)
        pending = any(item["status"] == "pending" for item in states.values())
        if not pending or clock() >= deadline:
            break
        sleep(min(interval_s, max(0, deadline - clock())))
    for state in states.values():
        if state["status"] == "pending":
            state["status"] = "timed_out"
    if report:
        report(states)
    return states


def workshop_checks(runtime, bedrock, mantle, *, region):
    """Use the same profiles and request formats as Fundamentals and Playbook."""
    checks = {}
    for role in ("workhorse", "small", "deep"):
        selected = resolve_model(DEFAULT_ALIASES[role], region=region)
        controls = {"max_tokens": 128}
        if selected.capabilities.support("effort") is Support.SUPPORTED:
            controls["effort"] = "low"
        request = build_converse_request(
            selected, [{"role": "user", "content": [{"text": "Reply with OK."}]}], **controls,
        )

        def converse_check(request=request):
            response = runtime.converse(**request)
            if "usage" not in response or not response.get("output", {}).get("message", {}).get("content"):
                raise ValueError("Model probe returned no message/usage")

        checks[f"converse:{selected.model_id}"] = converse_check

    mantle_invoked = False

    def mantle_check():
        nonlocal mantle_invoked
        args = {"model": "anthropic.claude-sonnet-5",
                "messages": [{"role": "user", "content": "Reply with OK."}]}
        if not mantle_invoked:
            mantle.messages.create(**args, max_tokens=128, output_config={"effort": "low"})
            mantle_invoked = True
        counted = mantle.messages.count_tokens(**args)
        if counted.input_tokens <= 0:
            raise ValueError("Mantle returned no input-token count")

    cohere_invoked = False

    def cohere_check():
        nonlocal cohere_invoked
        model_id = "cohere.rerank-v3-5:0"
        if not cohere_invoked:
            response = runtime.invoke_model(
                modelId=model_id, contentType="application/json", accept="application/json",
                body=json.dumps({"api_version": 2, "query": "Return policy",
                                 "documents": ["Returns require a receipt.", "Store opening hours."], "top_n": 1}),
            )
            body = response["body"]
            try:
                result = json.loads(body.read())
            finally:
                body.close()
            if not result.get("results"):
                raise ValueError("Cohere returned no ranking")
            cohere_invoked = True
        available = bedrock.get_foundation_model_availability(modelId=model_id)
        if available.get("authorizationStatus") != "AUTHORIZED":
            raise PermissionError("Cohere model access is not authorized")
        if available.get("regionAvailability") != "AVAILABLE":
            raise ValueError("Cohere model is unavailable in this Region")
        if (available.get("agreementAvailability", {}).get("status") != "AVAILABLE"
                or available.get("entitlementAvailability") != "AVAILABLE"):
            raise ActivationPending("Cohere agreement/entitlement activation is pending")

    checks["mantle:sonnet-5:messages-and-count"] = mantle_check
    checks["rerank:cohere-v3-5:invocation-and-agreement"] = cohere_check
    return checks


def main():
    import boto3
    from anthropic import AnthropicBedrockMantle
    from botocore.config import Config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--report", type=Path, default=Path(".workshop-readiness.json"))
    args = parser.parse_args()
    config = Config(connect_timeout=5, read_timeout=30, retries={"total_max_attempts": 1})
    session = boto3.Session(region_name=args.region)
    with closing(session.client("sts", config=config)) as sts:
        identity = sts.get_caller_identity()
    summary = {"region": args.region, "account": identity["Account"],
               "principal_arn": identity["Arn"], "checks": {}}

    def save(states):
        summary.update(checked_at=datetime.now(UTC).isoformat(), checks=states)
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
        print(" | ".join(f"{name}: {item['status']}" for name, item in states.items()), flush=True)

    with (closing(session.client("bedrock-runtime", config=config)) as runtime,
          closing(session.client("bedrock", config=config)) as bedrock,
          AnthropicBedrockMantle(aws_region=args.region, timeout=30, max_retries=0) as mantle):
        states = wait_for_readiness(
            workshop_checks(runtime, bedrock, mantle, region=args.region),
            timeout_s=args.timeout, interval_s=args.interval, report=save,
        )
    return 0 if all(item["status"] == "ready" for item in states.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
