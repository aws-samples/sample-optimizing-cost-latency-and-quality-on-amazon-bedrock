"""Exact-manifest cleanup for runtime-created workshop resources (SDK 1.43.101).

No resource discovery by name, bucket deletion, IAM mutation, log deletion, or
account-settings mutation. Plans are local. Execution verifies caller/region,
server ownership and targeted CloudFormation membership before the first write.
A manifest is a trusted creation record, not an authorization mechanism: restrict
its writers. Clients must have bounded connect/read timeouts and retries.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError
from botocore.session import get_session
from botocore.validate import validate_parameters

WORKSHOP = "bedrock-prompt-optimization"
KINDS = (
    "ab_test",
    "online_evaluation",
    "gateway_target",
    "gateway",
    "bundle",
    "runtime",
    "memory",
    "guardrail",
    "artifact",
)
SERVICE = dict.fromkeys(KINDS, "bedrock-agentcore-control")
SERVICE.update(ab_test="bedrock-agentcore", guardrail="bedrock", artifact="s3")
GET = {
    "runtime": ("get_agent_runtime", "agentRuntimeId", "agentRuntimeArn"),
    "ab_test": ("get_ab_test", "abTestId", "abTestArn"),
    "online_evaluation": ("get_online_evaluation_config", "onlineEvaluationConfigId", "onlineEvaluationConfigArn"),
    "gateway": ("get_gateway", "gatewayIdentifier", "gatewayArn"),
    "bundle": ("get_configuration_bundle", "bundleId", "bundleArn"),
    "memory": ("get_memory", "memoryId", "arn"),
    "guardrail": ("get_guardrail", "guardrailIdentifier", "guardrailArn"),
}
DELETE = {
    "runtime": "delete_agent_runtime",
    "ab_test": "delete_ab_test",
    "online_evaluation": "delete_online_evaluation_config",
    "gateway": "delete_gateway",
    "gateway_target": "delete_gateway_target",
    "bundle": "delete_configuration_bundle",
    "memory": "delete_memory",
    "guardrail": "delete_guardrail",
}
NOT_FOUND = {
    "ResourceNotFoundException",
    "ResourceNotFound",
    "NotFoundException",
    "NoSuchKey",
    "NoSuchVersion",
    "NoSuchBucket",
}


class OwnershipError(RuntimeError):
    """The exact resource needs additional creation/ownership evidence."""


class CleanupError(RuntimeError):
    """Cleanup was blocked or partially failed; report preserves every step."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__(f"Cleanup {report['status']}: {report['action_needed']}")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _arn(value: str, manifest: Mapping, service: str | None = None) -> list[str]:
    parts = value.split(":", 5) if isinstance(value, str) else []
    if len(parts) != 6 or parts[0] != "arn" or parts[1] not in {"aws", "aws-cn", "aws-us-gov"}:
        raise ValueError(f"Invalid recorded ARN: {value!r}")
    if parts[4] != manifest["account_id"] or parts[3] != manifest["region"]:
        raise ValueError(f"ARN account/region does not match the manifest: {value}")
    if service and parts[2] != service:
        raise ValueError(f"Unexpected ARN service: {value}")
    return parts


def _key(resource: Mapping) -> str:
    if resource["kind"] == "artifact":
        return f"artifact:s3://{resource['bucket']}/{resource['key']}?versionId={resource.get('version_id', '<resolve-current>')}"
    if resource["kind"] == "gateway_target":
        return f"gateway_target:{resource['gateway_id']}/{resource['id']}"
    return f"{resource['kind']}:{resource['arn']}"


def _validate(manifest: Mapping) -> None:
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
        raise ValueError("schema_version=1 and the recorded run_id are required")
    if not re.fullmatch(r"\d{12}", manifest.get("account_id", "")) or not re.fullmatch(
        r"[a-z]{2}(?:-gov)?-[a-z]+-\d", manifest.get("region", "")
    ):
        raise ValueError("Explicit account_id and region are required")
    if not isinstance(manifest.get("resources"), list):
        raise ValueError("resources must be an explicit list")
    seen = set()
    for resource in manifest["resources"]:
        kind = resource.get("kind")
        if kind not in KINDS or resource.get("origin") not in {"runtime-created", "cloudformation"}:
            raise ValueError(
                "Only declared resource kinds and explicit runtime-created/cloudformation origins are allowed"
            )
        if not isinstance(resource.get("recorded_in"), str) or not resource["recorded_in"]:
            raise ValueError("Every resource needs its creation-record path in recorded_in")
        if kind == "artifact":
            if not resource.get("bucket") or not resource.get("key") or resource["key"].endswith("/"):
                raise ValueError("Artifact records must identify an exact bucket and object key, never a prefix")
        else:
            if not isinstance(resource.get("id"), str) or not resource["id"]:
                raise ValueError("An exact resource ID is required")
            if kind == "gateway_target":
                _arn(resource["gateway_arn"], manifest, "bedrock-agentcore")
                if resource["gateway_arn"].rsplit("/", 1)[-1] != resource["gateway_id"]:
                    raise ValueError("Gateway ID/ARN mismatch")
            else:
                _arn(resource["arn"], manifest, "bedrock" if kind == "guardrail" else "bedrock-agentcore")
                if resource["arn"].rsplit("/", 1)[-1] != resource["id"]:
                    raise ValueError("Recorded ID/ARN mismatch")
        identity = _key(resource)
        if identity in seen:
            raise ValueError(f"Duplicate manifest resource: {identity}")
        seen.add(identity)


def plan_cleanup(manifest: Mapping) -> dict:
    """Return exact identities and dependency order without creating clients or reading AWS.

    Stop every A/B, disable every online evaluator, then delete tests/evaluators,
    targets, gateways, bundles, runtimes, memories, guardrails and exact S3 objects.
    Native CloudFormation entries are retained. A failure halts subsequent writes.
    """
    _validate(manifest)
    resources = [r for r in manifest["resources"] if r["origin"] == "runtime-created"]
    steps = []
    for kind, action in [("ab_test", "stop"), ("online_evaluation", "disable"), *[(k, "delete") for k in KINDS]]:
        for resource in resources:
            if resource["kind"] == kind:
                steps.append({"resource": _key(resource), "action": action, "status": "planned"})
    return {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "account_id": manifest["account_id"],
        "region": manifest["region"],
        "manifest_sha256": _digest(manifest),
        "status": "planned",
        "mutated": False,
        "steps": steps,
        "resolved": {},
        "action_needed": list(manifest.get("action_needed", [])),
        "retained": [_key(r) for r in manifest["resources"] if r["origin"] == "cloudformation"],
    }


@lru_cache(maxsize=80)
def _shape(service: str, method: str):
    operation = "".join(p.title() for p in method.split("_")).replace("AbTest", "ABTest")
    return get_session().get_service_model(service).operation_model(operation).input_shape


class _Runner:
    def __init__(self, manifest, clients, timeout_s, poll_interval_s, sleep, monotonic):
        self.manifest, self.clients = manifest, clients
        self.sleep, self.monotonic, self.interval = sleep, monotonic, poll_interval_s
        self.deadline = monotonic() + timeout_s
        self.resources = {_key(r): r for r in manifest["resources"] if r["origin"] == "runtime-created"}
        self.snapshots = {}
        self.artifact_versions = {}
        self.mutation_attempts = 0

    def remaining(self):
        remaining = self.deadline - self.monotonic()
        if remaining <= 0:
            raise TimeoutError("Cleanup deadline exceeded; inspect the report and rerun the same manifest")
        return remaining

    def call(self, service, method, **request):
        self.remaining()
        validate_parameters(request, _shape(service, method))
        if method.startswith(("delete_", "update_")):
            self.mutation_attempts += 1
        return getattr(self.clients[service], method)(**request)

    def maybe(self, service, method, **request):
        try:
            return self.call(service, method, **request)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in NOT_FOUND or (service == "s3" and method == "head_object" and code == "404"):
                return None
            raise  # AccessDenied, throttling and validation failures are never absence.

    def get(self, r):
        kind = r["kind"]
        if kind == "gateway_target":
            return self.maybe(SERVICE[kind], "get_gateway_target", gatewayIdentifier=r["gateway_id"], targetId=r["id"])
        method, field, _ = GET[kind]
        request = {field: r["id"]}
        if kind == "memory":
            request["view"] = "without_decryption"
        result = self.maybe(SERVICE[kind], method, **request)
        return result.get("memory") if result is not None and kind == "memory" else result

    def tags(self, service, arn):
        keyword = "resourceARN" if service == "bedrock" else "resourceArn"
        tags = self.call(service, "list_tags_for_resource", **{keyword: arn}).get("tags", {})
        return {t["key"]: t["value"] for t in tags} if isinstance(tags, list) else tags

    def not_cfn(self, identities, tags=None):
        if any(str(key).startswith("aws:cloudformation:") for key in (tags or {})):
            raise OwnershipError("Native CloudFormation resource: retain it for stack cleanup")
        for identity in dict.fromkeys(identities):
            try:
                response = self.call("cloudformation", "describe_stack_resources", PhysicalResourceId=identity)
            except ClientError as exc:
                error = exc.response.get("Error", {})
                # DescribeStackResources uses ValidationError for an exact lookup
                # miss. Do not classify arbitrary validation/access errors as absent.
                if error.get("Code") == "ValidationError" and error.get("Message") in {
                    f"Stack for {identity} does not exist",
                    f"Stack with id {identity} does not exist",
                }:
                    continue
                raise
            if response.get("StackResources"):
                raise OwnershipError(f"CloudFormation owns {identity}; use stack cleanup")
            # An empty successful targeted lookup is also not a managed resource.

    def role(self, arn, purpose):
        partition = (
            "aws-cn"
            if self.manifest["region"].startswith("cn-")
            else "aws-us-gov"
            if self.manifest["region"].startswith("us-gov-")
            else "aws"
        )
        expected = (
            f"arn:{partition}:iam::{self.manifest['account_id']}:role/techmart-{purpose}-{self.manifest['region']}"
        )
        if arn != expected:
            raise OwnershipError(f"Server role does not match the known workshop role {expected}")

    def run_tags(self, tags, *, require_run=False):
        if tags.get("Workshop") != WORKSHOP:
            raise OwnershipError(f"Server Workshop tag must equal {WORKSHOP}")
        run = tags.get("WorkshopRunId")
        if (require_run and run != self.manifest["run_id"]) or (run is not None and run != self.manifest["run_id"]):
            raise OwnershipError("Server WorkshopRunId tag does not match this recorded run")

    def owned_runtime(self, arn):
        _arn(arn, self.manifest, "bedrock-agentcore")
        records = [r for r in self.resources.values() if r["kind"] == "runtime" and r["arn"] == arn]
        if len(records) != 1:
            raise OwnershipError("Bundle requires its Runtime's explicit runtime-created manifest entry")
        resource = records[0]
        server = self.get(resource)
        if server is None:
            raise OwnershipError("Bound Runtime is absent; bundle ownership needs an independent recorded attestation")
        if not self.verify(resource, server):
            raise OwnershipError("Bound Runtime disappeared during ownership verification")
        return server

    def pages(self, method, result_key, **request):
        rows, token = [], None
        seen = set()
        while True:
            result = self.call(
                "bedrock-agentcore-control", method, **request, **({"nextToken": token} if token else {})
            )
            rows.extend(result.get(result_key, []))
            token = result.get("nextToken")
            if not token:
                return rows
            if token in seen:
                raise RuntimeError("Pagination token repeated")
            seen.add(token)

    def verify(self, r, server):
        try:
            self.verify_present(r, server)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in NOT_FOUND and self.get(r) is None:
                return False
            raise
        return True

    def verify_present(self, r, server):
        kind = r["kind"]
        if kind == "gateway_target":
            if server.get("gatewayArn") != r["gateway_arn"] or server.get("targetId") != r["id"]:
                raise OwnershipError("Gateway target identity mismatch")
            gateway = self.maybe("bedrock-agentcore-control", "get_gateway", gatewayIdentifier=r["gateway_id"])
            if not gateway or gateway.get("gatewayArn") != r["gateway_arn"]:
                raise OwnershipError("Target parent Gateway could not be verified")
            self.role(gateway.get("roleArn"), "gateway")
            binding = r.get("target_binding", {})
            config = server.get("targetConfiguration", {})
            if binding.get("runtime_arn"):
                actual = config.get("http", {}).get("agentcoreRuntime", {}).get("arn")
                expected = binding["runtime_arn"]
            elif binding.get("lambda_arn"):
                actual = config.get("mcp", {}).get("lambda", {}).get("lambdaArn")
                expected = binding["lambda_arn"]
            else:
                raise OwnershipError("Record target_binding.runtime_arn or target_binding.lambda_arn from creation")
            _arn(expected, self.manifest)
            if actual != expected:
                raise OwnershipError("Gateway target binding changed")
            self.not_cfn([r["id"], f"{r['gateway_id']}|{r['id']}", f"{r['gateway_arn']}/target/{r['id']}"])
            return
        if server.get(GET[kind][2]) != r["arn"]:
            raise OwnershipError("Returned ARN differs from the recorded resource")
        tags = {} if kind == "ab_test" else self.tags(SERVICE[kind], r["arn"])
        self.not_cfn([r["id"], r["arn"]], tags)
        if kind == "runtime":
            self.role(server.get("roleArn"), "runtime")
            self.run_tags(tags)
            run = server.get("environmentVariables", {}).get("WORKSHOP_RUN_ID")
            if run is not None and run != self.manifest["run_id"]:
                raise OwnershipError("Runtime WORKSHOP_RUN_ID belongs to another run")
        elif kind == "ab_test":
            self.role(server.get("roleArn"), "abtest")
            if not r.get("gateway_arn") or server.get("gatewayArn") != r["gateway_arn"]:
                raise OwnershipError("Record and verify the A/B Gateway binding")
        elif kind == "online_evaluation":
            self.role(server.get("evaluationExecutionRoleArn"), "evaluation")
            if not r.get("data_source_sha256") or _digest(server.get("dataSourceConfig")) != r["data_source_sha256"]:
                raise OwnershipError("Record the exact online evaluation dataSourceConfig SHA-256")
        elif kind == "gateway":
            self.role(server.get("roleArn"), "gateway")
            recorded = {
                item["id"]
                for item in self.resources.values()
                if item["kind"] == "gateway_target" and item["gateway_id"] == r["id"]
            }
            current = self.pages("list_gateway_targets", "items", gatewayIdentifier=r["id"])
            if not {target["targetId"] for target in current} <= recorded:
                raise OwnershipError("Gateway has unrecorded targets; record their origin/ownership before cleanup")
        elif kind == "bundle":
            versions = r.get("versions", [])
            runtime_arns = r.get("runtime_arns", [])
            if not versions or not runtime_arns:
                raise OwnershipError("Bundle needs immutable version identities and explicit runtime_arns")
            for arn in runtime_arns:
                self.owned_runtime(arn)
            recorded = {v.get("version_id") for v in versions}
            current = self.pages("list_configuration_bundle_versions", "versions", bundleId=r["id"])
            if {v["versionId"] for v in current} != recorded or len(recorded) != len(versions):
                raise OwnershipError(
                    "Bundle has missing/unrecorded versions; capture all branch history before deletion"
                )
            if server.get("versionId") not in recorded:
                raise OwnershipError("Bundle head changed")
            for identity in versions:
                version = self.call(
                    "bedrock-agentcore-control",
                    "get_configuration_bundle_version",
                    bundleId=r["id"],
                    versionId=identity["version_id"],
                )
                expected = {
                    "bundle_id": r["id"],
                    "bundle_arn": r["arn"],
                    "version_id": version["versionId"],
                    "content_sha256": _digest(version["components"]),
                }
                if expected != identity or set(version["components"]) != set(runtime_arns):
                    raise OwnershipError(
                        "Bundle version content or Runtime binding differs from the recorded candidate"
                    )
        elif kind in {"memory", "guardrail"}:
            self.run_tags(tags, require_run=True)
            if server.get("managedByResourceArn"):
                raise OwnershipError("Memory is service-managed; clean up through its owning resource")

    def artifact(self, r):
        account, region = self.manifest["account_id"], self.manifest["region"]
        bucket, key = r["bucket"], r["key"]
        allowed = {f"techmart-agentcore-code-{account}-{region}", f"techmart-prompt-gate-{account}-{region}"}
        if bucket not in allowed:
            raise OwnershipError("Artifact bucket is not one of the known workshop artifact buckets")
        tags_response = self.maybe("s3", "get_bucket_tagging", Bucket=bucket, ExpectedBucketOwner=account)
        if tags_response is None:
            return None
        tags = {t["Key"]: t["Value"] for t in tags_response.get("TagSet", [])}
        self.run_tags(tags)  # The bucket itself may be native CFN; never delete it.
        location = self.call("s3", "get_bucket_location", Bucket=bucket, ExpectedBucketOwner=account).get(
            "LocationConstraint"
        )
        if ({None: "us-east-1", "EU": "eu-west-1"}.get(location, location)) != region:
            raise OwnershipError("Artifact bucket is in a different region")
        if not re.fullmatch(r"[a-f0-9]{64}", r.get("sha256", "")):
            raise OwnershipError("Record the uploaded object's SHA-256; a key prefix is not ownership proof")
        if not key.startswith(("workshop-refresh/", "source/", "requests/", "results/")):
            raise OwnershipError("Object is outside the workshop artifact paths")
        request = {"Bucket": bucket, "Key": key, "ExpectedBucketOwner": account}
        if r.get("version_id") is not None:
            request["VersionId"] = r["version_id"]
        head = self.maybe("s3", "head_object", **request)
        if head is None:
            return None
        if head.get("DeleteMarker"):
            raise OwnershipError(
                "A delete marker needs a separately recorded marker identity; it is not a content artifact"
            )
        if head.get("VersionId") == "null" or r.get("version_id") == "null":
            raise OwnershipError("Mutable S3 null versions require separate reviewed bucket cleanup")
        if r.get("version_id") is not None and head.get("VersionId") != r["version_id"]:
            raise OwnershipError("S3 returned a different version from the recorded upload")
        if head.get("VersionId") not in (None, "null") and r.get("version_id") is None:
            raise OwnershipError(
                "Versioned artifact needs the original upload's recorded version_id; current is not creation proof"
            )
        if head.get("VersionId") is not None:
            request["VersionId"] = head["VersionId"]
        if not head.get("ETag"):
            raise OwnershipError("Missing artifact ETag for conditional deletion")
        request["IfMatch"] = head["ETag"]
        obj = self.maybe("s3", "get_object", **request)
        if obj is None:
            return None
        sha = hashlib.sha256()
        try:
            while chunk := obj["Body"].read(1024 * 1024):
                self.remaining()
                sha.update(chunk)
        finally:
            obj["Body"].close()
        if sha.hexdigest() != r["sha256"]:
            raise OwnershipError("Artifact bytes do not match the recorded upload")
        self.artifact_versions[_key(r)] = request
        return {"version_id": request.get("VersionId"), "etag": head["ETag"], "sha256": sha.hexdigest()}

    def poll(self, fetch, done, label):
        while True:
            self.remaining()
            result = fetch()
            if done(result):
                return result
            if result and (
                result.get("status", "").endswith("FAILED")
                or result.get("status") in {"ERROR", "FAILED", "UPDATE_UNSUCCESSFUL"}
            ):
                raise RuntimeError(f"{label} failed: {result.get('status')}")
            self.sleep(min(self.interval, self.remaining()))

    def execute(self, r, action):
        if r["kind"] == "artifact":
            snapshot = self.artifact(r)  # Rehash/re-resolve immediately before the conditional write.
            if snapshot is None:
                return "already_absent"
            request = dict(self.artifact_versions[_key(r)])
            if "VersionId" in request:
                # Exact non-null S3 versions are immutable. Combining VersionId
                # with If-Match is rejected by the general-purpose bucket API.
                request.pop("IfMatch")
            self.maybe("s3", "delete_object", **request)
            # For an unversioned object, HEAD must now be absent. For a versioned
            # key, verify exactly the deleted version, not the new current version.
            check = {k: v for k, v in request.items() if k != "IfMatch"}
            self.poll(lambda: self.maybe("s3", "head_object", **check), lambda value: value is None, _key(r))
            return "deleted"
        server = self.get(r)
        if server is None:
            return "already_absent"
        if not self.verify(r, server):  # Fresh owner/CFN check before every resource mutation.
            return "already_absent"
        kind = r["kind"]
        if server.get("status") == "DELETING":
            self.poll(lambda: self.get(r), lambda value: value is None, _key(r))
            return "already_absent"
        if action == "stop":
            if server.get("executionStatus") == "STOPPED":
                return "already_stopped"
            self.maybe(SERVICE[kind], "update_ab_test", abTestId=r["id"], executionStatus="STOPPED")
            self.poll(
                lambda: self.get(r),
                lambda value: (
                    value is None or (value.get("executionStatus") == "STOPPED" and value.get("status") == "ACTIVE")
                ),
                _key(r),
            )
            return "stopped"
        if action == "disable":
            if server.get("executionStatus") == "DISABLED":
                return "already_disabled"
            self.maybe(
                SERVICE[kind],
                "update_online_evaluation_config",
                onlineEvaluationConfigId=r["id"],
                executionStatus="DISABLED",
            )
            self.poll(
                lambda: self.get(r),
                lambda value: (
                    value is None or (value.get("executionStatus") == "DISABLED" and value.get("status") == "ACTIVE")
                ),
                _key(r),
            )
            return "disabled"
        if kind == "ab_test" and server.get("executionStatus") != "STOPPED":
            raise OwnershipError("A/B resumed since the stop phase; stop it before deletion")
        if kind == "online_evaluation" and server.get("executionStatus") != "DISABLED":
            raise OwnershipError("Online evaluation re-enabled since the disable phase")
        if kind == "gateway_target":
            request = {"gatewayIdentifier": r["gateway_id"], "targetId": r["id"]}
        else:
            request = {GET[kind][1]: r["id"]}
        self.maybe(SERVICE[kind], DELETE[kind], **request)
        self.poll(lambda: self.get(r), lambda value: value is None, _key(r))
        return "deleted"


def cleanup(
    manifest: Mapping,
    clients: Mapping[str, Any] | None = None,
    *,
    execute: bool = False,
    verify: bool = False,
    timeout_s: float = 300,
    poll_interval_s: float = 2,
    sleep: Callable = time.sleep,
    monotonic: Callable = time.monotonic,
) -> dict:
    """Plan by default; verify=True reads only; execute=True verifies then mutates.

    Client keys are boto3 service names plus sts and cloudformation. All regional
    clients must target manifest.region. Total timeout bounds preflight/polling;
    configure each SDK call's timeout separately. CleanupError.report retains
    partial progress; no AccessDenied or ownership failure is reported as success.
    """
    report = plan_cleanup(manifest)
    if type(execute) is not bool or type(verify) is not bool:
        raise ValueError("execute and verify must be explicit booleans")
    if not execute and not verify:
        return report
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not 0 < timeout_s <= 3600
        or isinstance(poll_interval_s, bool)
        or not isinstance(poll_interval_s, (int, float))
        or not 0 < poll_interval_s <= 60
    ):
        raise ValueError("Use a positive finite timeout <=3600 and polling interval <=60")
    manifest = copy.deepcopy(dict(manifest))
    runner = _Runner(manifest, clients or {}, timeout_s, poll_interval_s, sleep, monotonic)
    checking = "manifest/account/region"
    try:
        if report["action_needed"]:
            raise OwnershipError("Resolve the manifest's recorded action_needed items before execution")
        services = {SERVICE[r["kind"]] for r in runner.resources.values()} | {"sts", "cloudformation"}
        if any(r["kind"] in {"ab_test", "bundle", "gateway_target"} for r in runner.resources.values()):
            services.add("bedrock-agentcore-control")
        for service in services:
            if service not in runner.clients or runner.clients[service].meta.region_name != manifest["region"]:
                raise OwnershipError(f"Supply a {service} client in {manifest['region']}")
        if runner.call("sts", "get_caller_identity").get("Account") != manifest["account_id"]:
            raise OwnershipError("Caller account does not match the recorded workshop account")
        # Verify all resources before stopping/deleting any. An unprovable artifact
        # must not be discovered only after its Runtime has already been removed.
        for identity, resource in runner.resources.items():
            checking = identity
            server = runner.artifact(resource) if resource["kind"] == "artifact" else runner.get(resource)
            if server is not None and resource["kind"] != "artifact" and not runner.verify(resource, server):
                server = None
            runner.snapshots[identity] = server
            report["resolved"][identity] = {"ownership": "verified" if server is not None else "already_absent"}
            if resource["kind"] == "artifact" and server is not None:
                report["resolved"][identity].update(server)
    except Exception as exc:
        report["status"] = "blocked"
        report["action_needed"].append(f"{checking}: {type(exc).__name__}: {exc}")
        raise CleanupError(report) from exc
    if not execute:
        report["status"] = "verified_plan"
        return report
    for step in report["steps"]:
        try:
            step["status"] = "in_progress"
            if runner.snapshots[step["resource"]] is None:
                step["status"] = "already_absent"
            else:
                step["status"] = runner.execute(runner.resources[step["resource"]], step["action"])
            report["mutated"] = runner.mutation_attempts > 0
        except Exception as exc:
            report["mutated"] = runner.mutation_attempts > 0
            step["status"] = "failed"
            step["error"] = f"{type(exc).__name__}: {exc}"
            report["status"] = "partial_failure"
            report["action_needed"].append(step["error"])
            for pending in report["steps"]:
                if pending["status"] == "planned":
                    pending["status"] = "not_attempted"
            raise CleanupError(report) from exc
    report["status"] = "complete"
    return report


def manifest_from_records(paths: Sequence[str | Path], *, run_id: str, account_id: str, region: str) -> dict:
    """Read explicitly selected helper records; never discover AWS resources.

    Supported: .deployment-*.json, .gateway-*.json and schema_version=1 cleanup
    manifests persisted by a caller. Other lifecycle JSON files are returned as
    action_needed until creation snapshots/complete bundle history are supplied.
    Missing fields are not invented. The resulting manifest remains plan-only.
    """
    if isinstance(paths, (str, bytes)):
        raise ValueError("paths must be an explicit list of creation-record files")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "account_id": account_id,
        "region": region,
        "resources": [],
        "action_needed": [],
    }
    for path in paths:
        path = Path(path).resolve(strict=True)
        record = json.loads(path.read_text())
        source = str(path)
        if record.get("schema_version") == 1 and "resources" in record:
            _validate(record)
            if any(record[k] != manifest[k] for k in ("run_id", "account_id", "region")):
                raise ValueError(f"Different run/account/region in {path}")
            manifest["resources"].extend(record["resources"])
            manifest["action_needed"].extend(record.get("action_needed", []))
        elif "agentRuntimeId" in record and "agentRuntimeArn" in record:
            if record.get("run_id") != run_id:
                raise ValueError(f"Missing/different deployment run_id in {path}")
            if record.get("origin") != "runtime-created":
                manifest["action_needed"].append(
                    f"{source}: Runtime {record['agentRuntimeArn']} lacks explicit runtime-created origin; "
                    "confirm the original create response, not an update or evaluated reference"
                )
                continue
            manifest["resources"].append(
                {
                    "kind": "runtime",
                    "id": record["agentRuntimeId"],
                    "arn": record["agentRuntimeArn"],
                    "origin": "runtime-created",
                    "recorded_in": source,
                }
            )
            if all(record.get(k) for k in ("artifactBucket", "artifactKey", "artifactSha256")):
                artifact = {
                    "kind": "artifact",
                    "bucket": record["artifactBucket"],
                    "key": record["artifactKey"],
                    "sha256": record["artifactSha256"],
                    "origin": "runtime-created",
                    "recorded_in": source,
                }
                if record.get("artifactVersionId"):
                    artifact["version_id"] = record["artifactVersionId"]
                manifest["resources"].append(artifact)
        elif "gateway_id" in record and "target_id" in record:
            if record.get("run_id") != run_id:
                raise ValueError(f"Missing/different Gateway run_id in {path}")
            if record.get("origin") != "runtime-created":
                manifest["action_needed"].append(
                    f"{source}: Gateway {record['gateway_id']} / target {record['target_id']} need explicit creation origin"
                )
                continue
            partition = (
                "aws-cn" if region.startswith("cn-") else "aws-us-gov" if region.startswith("us-gov-") else "aws"
            )
            arn = f"arn:{partition}:bedrock-agentcore:{region}:{account_id}:gateway/{record['gateway_id']}"
            manifest["resources"].extend(
                [
                    {
                        "kind": "gateway",
                        "id": record["gateway_id"],
                        "arn": arn,
                        "origin": "runtime-created",
                        "recorded_in": source,
                    },
                    {
                        "kind": "gateway_target",
                        "id": record["target_id"],
                        "gateway_id": record["gateway_id"],
                        "gateway_arn": arn,
                        "target_binding": {"lambda_arn": record.get("lambda_arn")},
                        "origin": "runtime-created",
                        "recorded_in": source,
                    },
                ]
            )
        else:
            manifest["action_needed"].append(
                f"{source}: persist an explicit runtime-created resource manifest; this record lacks creation/ownership fields"
            )
    unique = {}
    for resource in manifest["resources"]:
        identity = _key(resource)
        if identity in unique and unique[identity] != resource:
            raise ValueError(f"Conflicting creation records for {identity}")
        unique[identity] = resource
    manifest["resources"] = list(unique.values())
    _validate(manifest)
    return manifest
