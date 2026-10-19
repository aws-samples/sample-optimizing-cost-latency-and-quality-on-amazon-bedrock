# Part 1: Fundamentals

This section introduces token economics, model APIs, observability, and a first quality evaluation. Start with AWS-native telemetry and AgentCore Evaluations. Langfuse is provisioned up front and ready as an alternative or fallback for trace inspection.

## Learning Objectives

After completing this section, you will:

- Be comfortable navigating Jupyter notebooks (kernels, cells, shortcuts; optional orientation)
- Understand token pricing and throughput limits (TPM/RPM)
- Use CountTokens where supported by the selected model/API
- Compare Converse and optional provider-specific APIs using explicit capability checks
- Capture local model spans, reconcile usage and cost, and run a first AgentCore evaluation

## Notebooks

| Notebook | Description |
| --- | --- |
| [00-jupyter-notebook-101](./00-jupyter-notebook-101.ipynb) | Optional tooling introduction: kernels, cells, shortcuts, and a first Bedrock call |
| [01-prompts-101](./01-prompts-101.ipynb) | Tokens, pricing, quotas, inference profiles, and model/API capabilities |
| [02-observability-and-evaluation](./02-observability-and-evaluation.ipynb) | AWS-native traces, usage accounting, and an initial response evaluation |

Use the [Workshop Studio agenda](https://catalog.us-east-1.prod.workshops.aws/workshops/60d21a0a-c56f-47aa-9e5d-45181cd42507/en-US) for durations and track choices.

## Prerequisites

- The provisioned Code Editor with **Bedrock Workshop (Python 3.13)** selected. For an optional local environment, run `uv sync --frozen --python 3.13 --extra notebook --extra langfuse` from the root, as described in the [repository setup](../README.md).
- AWS credentials through the configured credential chain, access to the selected Bedrock model/API, and the telemetry/evaluation prerequisites for the chosen exercise.
- A fresh notebook kernel. The default backend is `agentcore`. To use the provisioned Langfuse alternative, follow **Langfuse Project Setup** in your workshop guide to create project API keys, save the connection settings, and select `langfuse` or `both` before restarting the kernel. No additional install is needed.

## Key Metrics Covered

| Metric | Description |
|--------|-------------|
| **Quality** | Response correctness against authored references; local response checks do not prove remote tool behavior |
| **Cost** | Uncached input, cache writes/reads, and output; report evaluator and infrastructure charges separately |
| **Latency** | End-to-end duration; TTFT only where streaming is explicitly measured |
| **Throughput** | TPM, RPM |
