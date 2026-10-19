# Part 2: Optimization Playbook

Sixteen levers for cost, latency, and quality, organized into three effort tiers. Each tier combines explanations and experiments in one notebook, with theory and references in the companion walkthrough. Lever 16 is a concept-only overview.

The order is the point: **start LOW, earn HIGH effort.**

## Notebooks

| Notebook | Levers covered |
| --- | --- |
| [01-low-effort](./01-low-effort.ipynb) | 01 Model selection · 02 Prompt design, provider by provider · 03 Output and sampling controls · 04 Prompt caching, provider by provider · 05 Reasoning effort and budgets |
| [02-medium-effort](./02-medium-effort.ipynb) | 06 Routing and cascades · 07 Bedrock Guardrails · 08 RAG/indexing · 09 Context management · 10 Persistent memory · 11 Batch inference |
| [03-high-effort](./03-high-effort.ipynb) | 12 Harness engineering · 13 Sub-agent delegation · 14 Automated prompt optimization · 15 Tool search and tool descriptions · 16 Small-model customization (concept only) |

Use the [Workshop Studio agenda](https://catalog.us-east-1.prod.workshops.aws/workshops/60d21a0a-c56f-47aa-9e5d-45181cd42507/en-US) for tier durations and track choices.

## Exercise boundaries

- Context management compares full history, frequent compaction, and size-triggered compaction. Harness engineering compares overloaded and task-focused skill, tool and record loading while keeping shared instructions fixed. The notebooks identify optional service calls beside their exercises.
- Small-model customization is an optional concept overview.
- The delegation comparison uses two Strands readers in parallel, then a lead agent answers from their selected evidence.
- Lever 14 starts with a Bedrock Advanced Prompt Optimization (APO) job and a comparison on held-out tickets. A separate optional warranty exercise uses AgentCore Evaluations and Recommendations to diagnose an instruction problem, propose a fix, and compare answers on new cases.
- Managed KB, persistent memory, and Gateway exercises use the resources described in their notebook sections. Gateway compares all 32 definitions with search followed by a five-tool application cap; only the two product/policy tools execute. Inspect the returned count, loaded definitions, tool results and final answer.
- At an AWS event, run batch input preparation and leave the submission cell commented out. For an eligible own-account run, uncomment the submission cell and set `RUN_BATCH=True`.
- Keep the task, data, and acceptance rules fixed and change only the lever being tested. Model selection and routing deliberately vary the model; prompt comparisons keep the model/API fixed. Keep unsuccessful or inconclusive outcomes visible.

## Prerequisites

- Completed [Part 1: Fundamentals](../01-fundamentals/)
- The provisioned Python 3.13 Code Editor environment. For optional local setup, run `uv sync --frozen --python 3.13 --extra notebook --extra langfuse` from the root; see the [repository setup](../README.md).
- For pip setup, run `pip install --require-hashes -r requirements-langfuse.lock`, then install the shared package with `pip install --no-deps -e .`.
- AWS credentials and the Bedrock/AgentCore access required by the selected exercises.
- A fresh kernel with `OBSERVABILITY_BACKEND=agentcore` for the first lesson path. Langfuse is already provisioned and its client installed. Follow **Langfuse Project Setup** in your workshop guide to create project API keys, select `langfuse` or `both`, then restart the kernel to switch; no additional install is needed.
