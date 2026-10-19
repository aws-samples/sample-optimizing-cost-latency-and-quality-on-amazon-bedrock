"""Offline cleanup tests using stateful fake AWS clients and real SDK schemas."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

spec = importlib.util.spec_from_file_location(
    "journey_cleanup_test",
    Path(__file__).resolve().parents[1] / "03-developer-journey/utils/cleanup.py",
)
cleanup_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup_module)
CleanupError = cleanup_module.CleanupError
cleanup = cleanup_module.cleanup
manifest_from_records = cleanup_module.manifest_from_records
plan_cleanup = cleanup_module.plan_cleanup

ACCOUNT, REGION, RUN = "123456789012", "us-east-1", "actual-recorded-run"
BASE = f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}"
RUNTIME = f"{BASE}:runtime/WorkshopAgent-0123456789"
GATEWAY = f"{BASE}:gateway/workshop-0123456789"
BUNDLE = f"{BASE}:configuration-bundle/WorkshopPrompt-0123456789"
AB = f"{BASE}:ab-test/11111111-1111-4111-8111-111111111111"
EVAL = f"{BASE}:online-evaluation-config/WorkshopEval-0123456789"
MEMORY = f"{BASE}:memory/WorkshopMemory-0123456789"
GUARDRAIL = f"arn:aws:bedrock:{REGION}:{ACCOUNT}:guardrail/abc123456789"
BUCKET = f"techmart-agentcore-code-{ACCOUNT}-{REGION}"
WORKSHOP = "bedrock-prompt-optimization"
VERSIONS = ["00000000-0000-4000-8000-000000000001", "00000000-0000-4000-8000-000000000002"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def error(code="ResourceNotFoundException", message="resource absent"):
    return ClientError({"Error": {"Code": code, "Message": message}}, "FakeOperation")


def role(purpose):
    return f"arn:aws:iam::{ACCOUNT}:role/techmart-{purpose}-{REGION}"


def record(kind, arn, **extra):
    return {
        "kind": kind,
        "id": arn.rsplit("/", 1)[-1],
        "arn": arn,
        "origin": "runtime-created",
        "recorded_in": "/recorded/created-resources.json",
        **extra,
    }


def fixture_manifest():
    config = {RUNTIME: {"configuration": {"system_prompt": "Real recorded test prompt", "model_id": "example"}}}
    versions = [
        {
            "bundle_id": BUNDLE.rsplit("/", 1)[-1],
            "bundle_arn": BUNDLE,
            "version_id": version,
            "content_sha256": digest(config),
        }
        for version in VERSIONS
    ]
    source = {
        "cloudWatchLogs": {
            "logGroupNames": ["/aws/bedrock-agentcore/workshop/notebook-agents"],
            "serviceNames": ["workshop"],
        }
    }
    manifest = {
        "schema_version": 1,
        "run_id": RUN,
        "account_id": ACCOUNT,
        "region": REGION,
        "resources": [
            record("runtime", RUNTIME),
            record("gateway", GATEWAY),
            record("ab_test", AB, gateway_arn=GATEWAY),
            record("online_evaluation", EVAL, data_source_sha256=digest(source)),
            {
                "kind": "gateway_target",
                "id": "target123456",
                "gateway_id": GATEWAY.rsplit("/", 1)[-1],
                "gateway_arn": GATEWAY,
                "target_binding": {"runtime_arn": RUNTIME},
                "origin": "runtime-created",
                "recorded_in": "/recorded/created-target.json",
            },
            record("bundle", BUNDLE, versions=versions, runtime_arns=[RUNTIME]),
            record("memory", MEMORY),
            record("guardrail", GUARDRAIL),
            {
                "kind": "artifact",
                "bucket": BUCKET,
                "key": "workshop-refresh/agent/hash.zip",
                "version_id": "upload-v1",
                "sha256": hashlib.sha256(b"actual uploaded bytes").hexdigest(),
                "origin": "runtime-created",
                "recorded_in": "/recorded/upload.json",
            },
        ],
    }
    return manifest, config, source


class FakeClient:
    def __init__(self, world, service):
        self.world, self.service = world, service
        self.meta = SimpleNamespace(region_name=REGION)

    def __getattr__(self, method):
        def call(**kwargs):
            return self.world.call(self.service, method, kwargs)

        return call


class World:
    def __init__(self):
        self.manifest, self.config, source = fixture_manifest()
        self.events = []
        self.denied = None
        self.not_found_delete = None
        self.stuck_delete = None
        self.native_ids = set()
        self.account = ACCOUNT
        self.extra_targets = []
        self.states = {
            "runtime": {
                "agentRuntimeArn": RUNTIME,
                "roleArn": role("runtime"),
                "status": "READY",
                "environmentVariables": {"WORKSHOP_RUN_ID": RUN},
            },
            "ab_test": {
                "abTestArn": AB,
                "roleArn": role("abtest"),
                "gatewayArn": GATEWAY,
                "status": "ACTIVE",
                "executionStatus": "RUNNING",
            },
            "online_evaluation": {
                "onlineEvaluationConfigArn": EVAL,
                "evaluationExecutionRoleArn": role("evaluation"),
                "status": "ACTIVE",
                "executionStatus": "ENABLED",
                "dataSourceConfig": source,
            },
            "gateway": {"gatewayArn": GATEWAY, "roleArn": role("gateway"), "status": "READY"},
            "gateway_target": {
                "gatewayArn": GATEWAY,
                "targetId": "target123456",
                "status": "READY",
                "targetConfiguration": {"http": {"agentcoreRuntime": {"arn": RUNTIME}}},
            },
            "bundle": {"bundleArn": BUNDLE, "versionId": VERSIONS[1], "status": "ACTIVE"},
            "memory": {"arn": MEMORY, "status": "ACTIVE"},
            "guardrail": {"guardrailArn": GUARDRAIL, "status": "READY"},
        }
        self.tags = {
            arn: {"Workshop": WORKSHOP, "WorkshopRunId": RUN}
            for arn in (RUNTIME, EVAL, GATEWAY, BUNDLE, MEMORY, GUARDRAIL)
        }
        self.objects = {"upload-v1": b"actual uploaded bytes", "someone-else-v2": b"other run bytes"}
        self.object_version = "upload-v1"
        self.listed_versions = list(VERSIONS)
        self.clients = {
            service: FakeClient(self, service)
            for service in ["sts", "cloudformation", "bedrock-agentcore-control", "bedrock-agentcore", "bedrock", "s3"]
        }

    def call(self, service, method, kwargs):
        self.events.append((method, copy.deepcopy(kwargs)))
        if method == self.denied:
            raise error("AccessDeniedException", "deliberate permission denial")
        if method == "get_caller_identity":
            return {"Account": self.account}
        if method == "describe_stack_resources":
            physical = kwargs["PhysicalResourceId"]
            if physical in self.native_ids:
                return {"StackResources": [{"PhysicalResourceId": physical, "StackName": "native-stack"}]}
            raise error("ValidationError", f"Stack for {physical} does not exist")
        if method == "list_tags_for_resource":
            arn = kwargs.get("resourceArn", kwargs.get("resourceARN"))
            tags = self.tags[arn]
            return {"tags": [{"key": k, "value": v} for k, v in tags.items()] if service == "bedrock" else tags}
        if method == "list_gateway_targets":
            return {
                "items": ([{"targetId": "target123456"}] if "gateway_target" in self.states else [])
                + self.extra_targets
            }
        if method == "list_configuration_bundle_versions":
            return {"versions": [{"versionId": version} for version in self.listed_versions]}
        if method == "get_configuration_bundle_version":
            return {"versionId": kwargs["versionId"], "components": self.config}
        if service == "s3":
            assert kwargs.get("ExpectedBucketOwner") == ACCOUNT
            if method == "get_bucket_tagging":
                return {"TagSet": [{"Key": "Workshop", "Value": WORKSHOP}]}
            if method == "get_bucket_location":
                return {"LocationConstraint": None}
            version = kwargs.get("VersionId", self.object_version)
            if method in {"head_object", "get_object", "delete_object"}:
                if version not in self.objects:
                    raise error("NoSuchVersion")
                if method == "head_object":
                    return {"VersionId": version, "ETag": '"etag"'}
                if method == "get_object":
                    return {"Body": io.BytesIO(self.objects[version])}
                if version is None:
                    assert kwargs["IfMatch"] == '"etag"'
                else:
                    assert "IfMatch" not in kwargs
                del self.objects[version]
                return {}
        mapping = {
            "agent_runtime": "runtime",
            "ab_test": "ab_test",
            "online_evaluation_config": "online_evaluation",
            "gateway_target": "gateway_target",
            "gateway": "gateway",
            "configuration_bundle": "bundle",
            "memory": "memory",
            "guardrail": "guardrail",
        }
        for suffix, kind in mapping.items():
            if method not in {f"get_{suffix}", f"update_{suffix}", f"delete_{suffix}"}:
                continue
            if kind not in self.states:
                raise error()
            if method.startswith("get_"):
                state = copy.deepcopy(self.states[kind])
                return {"memory": state} if kind == "memory" else state
            if method.startswith("update_"):
                self.states[kind]["executionStatus"] = kwargs["executionStatus"]
                return {}
            if method == self.stuck_delete:
                self.states[kind]["status"] = "DELETING"
                return {}
            del self.states[kind]
            if method == self.not_found_delete:
                raise error()
            return {}
        raise AssertionError(f"Unexpected call: {service}.{method}({kwargs})")

    def writes(self):
        return [method for method, _ in self.events if method.startswith(("delete_", "update_"))]


@pytest.fixture
def world():
    return World()


def test_default_plan_is_local_and_dependency_ordered(world):
    result = cleanup(world.manifest)
    assert result["status"] == "planned" and result["mutated"] is False
    assert world.events == []
    assert [(s["action"], s["resource"].split(":")[0]) for s in result["steps"]] == [
        ("stop", "ab_test"),
        ("disable", "online_evaluation"),
        ("delete", "ab_test"),
        ("delete", "online_evaluation"),
        ("delete", "gateway_target"),
        ("delete", "gateway"),
        ("delete", "bundle"),
        ("delete", "runtime"),
        ("delete", "memory"),
        ("delete", "guardrail"),
        ("delete", "artifact"),
    ]
    assert "versionId=upload-v1" in result["steps"][-1]["resource"]


def test_verified_plan_performs_no_mutations(world):
    report = cleanup(world.manifest, world.clients, verify=True)
    assert report["status"] == "verified_plan" and not report["mutated"]
    assert not world.writes()
    assert all(v["ownership"] == "verified" for v in report["resolved"].values())


def test_execution_waits_for_absence_and_deletes_only_recorded_version(world):
    report = cleanup(world.manifest, world.clients, execute=True)
    assert report["status"] == "complete" and report["mutated"]
    assert world.writes() == [
        "update_ab_test",
        "update_online_evaluation_config",
        "delete_ab_test",
        "delete_online_evaluation_config",
        "delete_gateway_target",
        "delete_gateway",
        "delete_configuration_bundle",
        "delete_agent_runtime",
        "delete_memory",
        "delete_guardrail",
        "delete_object",
    ]
    assert world.objects == {"someone-else-v2": b"other run bytes"}
    assert not world.states
    again = cleanup(world.manifest, world.clients, execute=True)
    assert again["status"] == "complete" and not again["mutated"]
    assert all(s["status"] == "already_absent" for s in again["steps"])


@pytest.mark.parametrize(
    "kind,purpose",
    [("runtime", "runtime"), ("ab_test", "abtest"), ("online_evaluation", "evaluation"), ("gateway", "gateway")],
)
def test_foreign_role_blocks_before_any_write(world, kind, purpose):
    key = "evaluationExecutionRoleArn" if kind == "online_evaluation" else "roleArn"
    world.states[kind][key] = role(purpose) + "-not-the-known-role"
    with pytest.raises(CleanupError) as caught:
        cleanup(world.manifest, world.clients, execute=True)
    assert caught.value.report["status"] == "blocked" and not world.writes()
    assert "known workshop role" in str(caught.value)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda w: w.tags[RUNTIME].pop("Workshop"),
        lambda w: w.states["runtime"]["environmentVariables"].update(WORKSHOP_RUN_ID="other-run"),
        lambda w: w.tags[MEMORY].update(WorkshopRunId="other-run"),
        lambda w: w.tags[GUARDRAIL].pop("WorkshopRunId"),
        lambda w: w.states["memory"].update(managedByResourceArn=RUNTIME),
        lambda w: w.states["online_evaluation"]["dataSourceConfig"].update(other="source"),
        lambda w: w.states["gateway_target"]["targetConfiguration"]["http"]["agentcoreRuntime"].update(
            arn="different"
        ),
        lambda w: w.extra_targets.append({"targetId": "not-recorded"}),
        lambda w: w.listed_versions.append("not-recorded"),
        lambda w: w.config[RUNTIME]["configuration"].update(system_prompt="changed"),
        lambda w: w.objects.update({"upload-v1": b"different content"}),
    ],
)
def test_unprovable_ownership_or_changed_bindings_block_every_write(world, mutation):
    mutation(world)
    with pytest.raises(CleanupError) as caught:
        cleanup(world.manifest, world.clients, execute=True)
    assert caught.value.report["status"] == "blocked" and not world.writes()


@pytest.mark.parametrize("target", ["runtime", "bundle", "gateway_target", "guardrail"])
def test_native_cfn_resource_mislabeled_runtime_created_is_not_deleted(world, target):
    item = next(r for r in world.manifest["resources"] if r["kind"] == target)
    world.native_ids.add(item["id"])
    with pytest.raises(CleanupError, match="CloudFormation owns"):
        cleanup(world.manifest, world.clients, execute=True)
    assert not world.writes()


def test_cfn_tag_and_explicit_cfn_origin_are_respected(world):
    world.tags[RUNTIME]["aws:cloudformation:stack-id"] = "native-stack"
    with pytest.raises(CleanupError, match="Native CloudFormation"):
        cleanup(world.manifest, world.clients, execute=True)
    runtime = next(r for r in world.manifest["resources"] if r["kind"] == "runtime")
    runtime["origin"] = "cloudformation"
    plan = plan_cleanup({**world.manifest, "resources": [runtime]})
    assert not plan["steps"] and plan["retained"] == [f"runtime:{RUNTIME}"]


@pytest.mark.parametrize(
    "where", ["get_agent_runtime", "list_tags_for_resource", "describe_stack_resources", "head_object"]
)
def test_access_denied_is_not_absence(world, where):
    world.denied = where
    with pytest.raises(CleanupError, match="AccessDenied") as caught:
        cleanup(world.manifest, world.clients, execute=True)
    assert caught.value.report["status"] == "blocked" and not world.writes()


def test_partial_failure_preserves_completed_steps_and_halts_dependents(world):
    world.denied = "delete_gateway_target"
    with pytest.raises(CleanupError) as caught:
        cleanup(world.manifest, world.clients, execute=True)
    report = caught.value.report
    assert report["status"] == "partial_failure"
    assert [s["status"] for s in report["steps"][:5]] == ["stopped", "disabled", "deleted", "deleted", "failed"]
    assert all(s["status"] == "not_attempted" for s in report["steps"][5:])
    assert "delete_gateway" not in world.writes() and "delete_agent_runtime" not in world.writes()
    world.denied = None
    assert cleanup(world.manifest, world.clients, execute=True)["status"] == "complete"


def test_not_found_race_on_delete_is_idempotent(world):
    world.not_found_delete = "delete_ab_test"
    assert cleanup(world.manifest, world.clients, execute=True)["status"] == "complete"


class Clock:
    def __init__(self):
        self.now = 0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        assert 0 < duration <= 2
        self.now += duration


def test_bounded_poll_reports_requested_but_not_completed_delete(world):
    world.stuck_delete = "delete_gateway_target"
    clock = Clock()
    with pytest.raises(CleanupError, match="deadline exceeded") as caught:
        cleanup(
            world.manifest,
            world.clients,
            execute=True,
            timeout_s=5,
            poll_interval_s=2,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
    assert clock.now == 5
    assert caught.value.report["steps"][4]["status"] == "failed"
    assert "delete_gateway" not in world.writes()


def test_account_region_checks_and_no_shared_settings_kinds(world):
    world.account = "999999999999"
    with pytest.raises(CleanupError, match="Caller account"):
        cleanup(world.manifest, world.clients, execute=True)
    world.account = ACCOUNT
    world.clients["s3"].meta.region_name = "us-west-2"
    with pytest.raises(CleanupError, match="client in us-east-1"):
        cleanup(world.manifest, world.clients, execute=True)
    bad = copy.deepcopy(world.manifest)
    bad["resources"][0]["kind"] = "log_group"
    with pytest.raises(ValueError):
        plan_cleanup(bad)
    assert not world.writes()


def test_versioned_artifact_requires_the_recorded_upload_version(world):
    world.manifest["resources"][-1].pop("version_id")
    with pytest.raises(CleanupError, match="original upload"):
        cleanup(world.manifest, world.clients, execute=True)
    assert not world.writes()


def test_manifest_import_requires_actual_origin_and_run(tmp_path):
    source = tmp_path / ".deployment-agent.json"
    record_data = {
        "agentRuntimeId": RUNTIME.rsplit("/", 1)[-1],
        "agentRuntimeArn": RUNTIME,
        "run_id": RUN,
        "artifactBucket": BUCKET,
        "artifactKey": "workshop-refresh/agent/hash.zip",
        "artifactSha256": hashlib.sha256(b"actual uploaded bytes").hexdigest(),
        "artifactVersionId": "upload-v1",
    }
    source.write_text(json.dumps(record_data))
    manifest = manifest_from_records([source], run_id=RUN, account_id=ACCOUNT, region=REGION)
    assert manifest["action_needed"] and not manifest["resources"]
    with pytest.raises(CleanupError, match="Resolve the manifest"):
        cleanup(manifest, {}, execute=True)
    record_data["origin"] = "runtime-created"
    source.write_text(json.dumps(record_data))
    manifest = manifest_from_records([source], run_id=RUN, account_id=ACCOUNT, region=REGION)
    assert len(manifest["resources"]) == 2 and not manifest["action_needed"]
    assert manifest["resources"][1]["version_id"] == "upload-v1"
    with pytest.raises(ValueError, match="run_id"):
        manifest_from_records([source], run_id="not-this-run", account_id=ACCOUNT, region=REGION)


def test_unknown_lifecycle_gate_is_a_reference_not_creation_authority(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"candidate": {"bundle_id": "candidate"}, "average_score": 1.0}))
    manifest = manifest_from_records([path], run_id=RUN, account_id=ACCOUNT, region=REGION)
    assert not manifest["resources"] and manifest["action_needed"]


def test_overlapping_manifests_deduplicate_only_identical_creation_entries(tmp_path, world):
    first, second = tmp_path / "deployment.json", tmp_path / "managed.json"
    first.write_text(json.dumps(world.manifest))
    second.write_text(json.dumps(world.manifest))
    merged = manifest_from_records([first, second], run_id=RUN, account_id=ACCOUNT, region=REGION)
    assert merged["resources"] == world.manifest["resources"]
    conflicting = copy.deepcopy(world.manifest)
    conflicting["resources"][0]["origin"] = "cloudformation"
    second.write_text(json.dumps(conflicting))
    with pytest.raises(ValueError, match="Conflicting creation records"):
        manifest_from_records([first, second], run_id=RUN, account_id=ACCOUNT, region=REGION)
    assert not world.events


def test_execute_requires_boolean_not_truthy_string(world):
    with pytest.raises(ValueError, match="booleans"):
        cleanup(world.manifest, world.clients, execute="false")
    assert not world.events


def test_unversioned_artifact_delete_is_conditional(world):
    artifact = world.manifest["resources"][-1]
    artifact.pop("version_id")
    world.manifest["resources"] = [artifact]
    world.object_version = None
    world.objects = {None: b"actual uploaded bytes", "another-version": b"leave intact"}
    report = cleanup(world.manifest, world.clients, execute=True)
    assert report["status"] == "complete"
    deletes = [kwargs for method, kwargs in world.events if method == "delete_object"]
    assert len(deletes) == 1 and deletes[0]["IfMatch"] == '"etag"'
    assert "VersionId" not in deletes[0]
    assert world.objects == {"another-version": b"leave intact"}


def test_mutable_s3_null_version_is_not_deleted_without_review(world):
    artifact = world.manifest["resources"][-1]
    artifact["version_id"] = "null"
    world.manifest["resources"] = [artifact]
    world.object_version = "null"
    world.objects = {"null": b"actual uploaded bytes"}
    with pytest.raises(CleanupError, match="null versions"):
        cleanup(world.manifest, world.clients, execute=True)
    assert not any(method == "delete_object" for method, _ in world.events)
    assert world.objects == {"null": b"actual uploaded bytes"}


def test_tag_read_not_found_race_requires_confirmed_resource_absence(world, monkeypatch):
    world.manifest["resources"] = [world.manifest["resources"][0]]
    original = world.call

    def call(service, method, kwargs):
        if method == "list_tags_for_resource":
            del world.states["runtime"]
            raise error()
        return original(service, method, kwargs)

    monkeypatch.setattr(world, "call", call)
    report = cleanup(world.manifest, world.clients, execute=True)
    assert report["status"] == "complete" and not report["mutated"]
    assert report["steps"][0]["status"] == "already_absent"
    assert not world.writes()


def test_changed_owner_after_preflight_is_rechecked(world, monkeypatch):
    original = world.call

    def call(service, method, kwargs):
        response = original(service, method, kwargs)
        if method == "update_ab_test":
            world.states["runtime"]["roleArn"] = role("runtime") + "-new-owner"
        return response

    monkeypatch.setattr(world, "call", call)
    with pytest.raises(CleanupError, match="known workshop role") as caught:
        cleanup(world.manifest, world.clients, execute=True)
    assert caught.value.report["status"] == "partial_failure"
    assert "delete_configuration_bundle" not in world.writes()
    assert "delete_agent_runtime" not in world.writes()


def test_import_preserves_unresolved_items_from_complete_manifest(world, tmp_path):
    world.manifest["action_needed"] = ["Missing original create response for another recorded resource"]
    path = tmp_path / "creation-manifest.json"
    path.write_text(json.dumps(world.manifest))
    imported = manifest_from_records([path], run_id=RUN, account_id=ACCOUNT, region=REGION)
    assert imported["action_needed"] == world.manifest["action_needed"]
    with pytest.raises(CleanupError, match="Resolve the manifest"):
        cleanup(imported, world.clients, execute=True)
    assert not world.events


@pytest.mark.parametrize("timeout", [True, float("nan"), float("inf"), 0])
def test_invalid_deadline_never_starts_aws_work(world, timeout):
    with pytest.raises(ValueError, match="timeout"):
        cleanup(world.manifest, world.clients, execute=True, timeout_s=timeout)
    assert not world.events


def test_native_gateway_target_composite_cfn_identity_is_blocked(world):
    # CloudFormation Ref is GatewayIdentifier|TargetId, rather than just TargetId.
    target = next(r for r in world.manifest["resources"] if r["kind"] == "gateway_target")
    world.native_ids.add(f"{target['gateway_id']}|{target['id']}")
    with pytest.raises(CleanupError, match="CloudFormation owns"):
        cleanup(world.manifest, world.clients, execute=True)
    assert not world.writes()


def test_async_stop_and_disable_finish_before_any_delete(world, monkeypatch):
    original = world.call
    pending = {}
    clock = Clock()

    def call(service, method, kwargs):
        if method.startswith("delete_"):
            assert all(count == 0 for count in pending.values())
        for kind, getter in [("ab_test", "get_ab_test"), ("online_evaluation", "get_online_evaluation_config")]:
            if method == getter and pending.get(kind, 0):
                pending[kind] -= 1
                if pending[kind] == 0:
                    world.states[kind].update(
                        status="ACTIVE", executionStatus="STOPPED" if kind == "ab_test" else "DISABLED"
                    )
        response = original(service, method, kwargs)
        if method in {"update_ab_test", "update_online_evaluation_config"}:
            kind = "ab_test" if method == "update_ab_test" else "online_evaluation"
            pending[kind] = 2
            world.states[kind].update(status="UPDATING", executionStatus="RUNNING" if kind == "ab_test" else "ENABLED")
        return response

    monkeypatch.setattr(world, "call", call)
    report = cleanup(
        world.manifest, world.clients, execute=True, timeout_s=10, sleep=clock.sleep, monotonic=clock.monotonic
    )
    assert report["status"] == "complete" and clock.now == 4


def test_unexpected_cfn_validation_error_is_not_ownership_evidence(world, monkeypatch):
    original = world.call

    def call(service, method, kwargs):
        if method == "describe_stack_resources":
            raise error("ValidationError", "Malformed lookup request")
        return original(service, method, kwargs)

    monkeypatch.setattr(world, "call", call)
    with pytest.raises(CleanupError, match="Malformed lookup"):
        cleanup(world.manifest, world.clients, execute=True)
    assert not world.writes()
