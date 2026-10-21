"""
Helper functions for deploying and invoking AgentCore Runtime agents.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import boto3
from bedrock_agentcore_starter_toolkit import Runtime


def get_clients(region: str | None = None):
    """Get boto3 clients for AgentCore."""
    region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    control_client = boto3.client("bedrock-agentcore-control", region_name=region)
    data_client = boto3.client("bedrock-agentcore", region_name=region)
    return control_client, data_client


def deploy_agent_to_runtime(
    agent_name: str,
    agent_file: str,
    requirements_file: str,
    region: str,
    env_vars: dict | None = None,
    execution_role_arn: str | None = None,
) -> str:
    """Deploy agent to AgentCore Runtime with Langfuse integration."""
    agentcore_runtime = Runtime()

    configure_kwargs = {
        "deployment_type": "container",
        "disable_otel": True,
        "entrypoint": agent_file,
        "auto_create_ecr": True,
        "requirements_file": requirements_file,
        "region": region,
        "agent_name": agent_name,
    }

    if execution_role_arn:
        configure_kwargs["execution_role"] = execution_role_arn
    else:
        configure_kwargs["auto_create_execution_role"] = True

    print(f"Configuring agent: {agent_name}")
    agentcore_runtime.configure(**configure_kwargs)

    # Modify Dockerfile to use Langfuse instead of opentelemetry-instrument
    dockerfile_path = Path("Dockerfile")
    if dockerfile_path.exists():
        content = dockerfile_path.read_text()
        agent_module = Path(agent_file).stem
        original_cmd = f'CMD ["opentelemetry-instrument", "python", "-m", "{agent_module}"]'
        new_cmd = f'CMD ["python", "-m", "{agent_module}"]'
        if original_cmd in content:
            content = content.replace(original_cmd, new_cmd)
            dockerfile_path.write_text(content)
            print("Dockerfile modified for Langfuse")

    # Set env vars
    default_env_vars = {
        "LANGFUSE_BASE_URL": os.environ.get("LANGFUSE_BASE_URL"),
        "LANGFUSE_PUBLIC_KEY": os.environ.get("LANGFUSE_PUBLIC_KEY"),
        "LANGFUSE_SECRET_KEY": os.environ.get("LANGFUSE_SECRET_KEY"),
        "PYTHONUNBUFFERED": "1",
    }
    if env_vars:
        default_env_vars.update(env_vars)

    print("Deploying to AgentCore Runtime...")
    launch_result = agentcore_runtime.launch(env_vars=default_env_vars, auto_update_on_conflict=True)
    print(f"Agent deployed: {launch_result.agent_arn}")

    return launch_result.agent_arn


def ensure_runtime_permissions(region: str | None = None) -> None:
    """Attach the shared customer-support permission set to the agent's execution role.

    The AgentCore toolkit auto-creates a separate execution role per agent, scoped to
    that agent's ECR repo and workload identity. Those roles do NOT include the
    permissions our tools need at runtime:

    - ssm:GetParameter      -- get_technical_support reads the KB ID from SSM
    - bedrock:Retrieve      -- get_technical_support queries the Bedrock Knowledge Base
    - bedrock:ApplyGuardrail -- v5+ agents attach a guardrail to model invocations

    Call this once after launch() in each runtime lab to give every agent the same
    permission set. Idempotent: re-running overwrites the same inline policy. Reads the
    execution role from .bedrock_agentcore.yaml (written by configure()).
    """
    region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    import yaml

    config = yaml.safe_load(Path(".bedrock_agentcore.yaml").read_text())
    default_agent = config["default_agent"]
    exec_role_arn = config["agents"][default_agent]["aws"]["execution_role"]
    role_name = exec_role_arn.split("/")[-1]

    account_id = boto3.client("sts").get_caller_identity()["Account"]
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "KnowledgeBaseIdLookup",
                "Effect": "Allow",
                "Action": "ssm:GetParameter",
                "Resource": [
                    f"arn:aws:ssm:{region}:{account_id}:parameter/{account_id}-{region}/kb/knowledge-base-id",
                    f"arn:aws:ssm:{region}:{account_id}:parameter/app/customersupport/kb/type",
                ],
            },
            {
                "Sid": "KnowledgeBaseRetrieve",
                "Effect": "Allow",
                "Action": "bedrock:Retrieve",
                "Resource": f"arn:aws:bedrock:{region}:{account_id}:knowledge-base/*",
            },
            {
                "Sid": "ApplyGuardrail",
                "Effect": "Allow",
                "Action": "bedrock:ApplyGuardrail",
                "Resource": f"arn:aws:bedrock:{region}:{account_id}:guardrail/*",
            },
        ],
    }
    boto3.client("iam").put_role_policy(
        RoleName=role_name,
        PolicyName="CustomerSupportRuntimeAccess",
        PolicyDocument=json.dumps(policy),
    )
    print(f"Granted customer-support runtime permissions to {role_name}")
    time.sleep(10)  # allow IAM propagation before the first invocation


def invoke_agent(data_client, agent_arn: str, prompt: str) -> dict:
    """Invoke agent via AgentCore API."""
    response = data_client.invoke_agent_runtime(
        agentRuntimeArn=agent_arn,
        runtimeSessionId=str(uuid.uuid4()),
        payload=json.dumps({"prompt": prompt}).encode(),
    )
    return json.loads(response["response"].read().decode("utf-8"))


def invoke_agent_with_timing(data_client, agent_arn: str, prompt: str) -> tuple:
    """Invoke agent and measure response time."""
    start_time = time.time()
    response = invoke_agent(data_client, agent_arn, prompt)
    latency_ms = (time.time() - start_time) * 1000
    return response, latency_ms


def find_agent_by_name(control_client, agent_name: str) -> str | None:
    """Find the intended Runtime across all pages, accepting old name separators."""
    matches = []
    next_token = None
    needle = agent_name.replace("-", "_")
    while True:
        page = control_client.list_agent_runtimes(**({"nextToken": next_token} if next_token else {}))
        matches.extend(
            agent for agent in page.get("agentRuntimes", []) if needle in agent["agentRuntimeName"].replace("-", "_")
        )
        next_token = page.get("nextToken")
        if not next_token:
            break
    if len(matches) > 1:
        raise ValueError(f"Multiple Runtimes match {agent_name}; select the intended deployment")
    return matches[0]["agentRuntimeArn"] if matches else None


def run_test_scenarios(data_client, agent_arn: str, scenarios: list) -> list:
    """Run all test scenarios against an agent."""
    results = []
    for scenario in scenarios:
        print(f"  Running: {scenario['id']}...")
        start_time = time.time()
        try:
            response = invoke_agent(data_client, agent_arn, scenario["query"])
            latency = (time.time() - start_time) * 1000
            results.append(
                {
                    "scenario_id": scenario["id"],
                    "query": scenario["query"],
                    "response": response,
                    "latency_ms": latency,
                    "success": True,
                }
            )
        except Exception as e:
            results.append(
                {
                    "scenario_id": scenario["id"],
                    "query": scenario["query"],
                    "error": str(e),
                    "latency_ms": (time.time() - start_time) * 1000,
                    "success": False,
                }
            )
    return results


def cleanup_agents(control_client, name_prefix: str, region: str | None = None):
    """Delete all agents matching name prefix and their ECR repositories."""
    region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    response = control_client.list_agent_runtimes()
    agents = response.get("agentRuntimes", [])

    matching = [a for a in agents if name_prefix in a["agentRuntimeName"]]
    print(f"Found {len(matching)} agents matching '{name_prefix}'")

    ecr_client = boto3.client("ecr", region_name=region)

    for agent in matching:
        agent_name = agent["agentRuntimeName"]
        try:
            rt = Runtime()
            rt.configure(agent_name=agent_name, region=region)
            rt.destroy(delete_ecr_repo=True)
            print(f"Deleted agent and ECR repository: {agent_name}")
        except Exception as e:
            print(f"Failed to delete {agent_name} via SDK, falling back to boto3: {e}")
            try:
                agent_id = agent["agentRuntimeArn"].split("/")[-1]
                control_client.delete_agent_runtime(agentRuntimeId=agent_id)
                print(f"Deleted agent: {agent_name}")
                ecr_repo_name = f"bedrock-agentcore-{agent_name}"
                try:
                    ecr_client.delete_repository(repositoryName=ecr_repo_name, force=True)
                    print(f"Deleted ECR repository: {ecr_repo_name}")
                except ecr_client.exceptions.RepositoryNotFoundException:
                    pass
            except Exception as e2:
                print(f"Failed to delete {agent_name}: {e2}")
