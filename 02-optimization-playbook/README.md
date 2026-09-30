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

- Context management compares full history, frequent compaction, and size-triggered compaction. Harness engineering compares verbose instructions with an audited, task-scoped version. Enable their live comparisons in the relevant notebook sections.
- Small-model customization is an optional concept overview.
- The runnable delegation comparison uses Strands. Claude Agent SDK and GEPA/DSPy are optional alternatives, outside the default installed environment.
- Lever 14 starts with a dataset-based Bedrock Advanced Prompt Optimization (APO) job and a held-out comparison. A single-prompt rewrite and trace-based AgentCore Recommendations are optional extensions; AgentCore Evaluations closes the comparison by checking recorded answers.
- Managed KB, persistent memory, and Gateway exercises use the resources described in their notebook sections. The Gateway comparison searches a reviewed catalog and can execute read-only product/policy tools; inspect the tool results and final answer as well as selection.
- At an AWS event, prepare batch inputs with `RUN_BATCH=0` and leave the submission cell commented out. A live batch job is an optional own-account extension requiring both uncommenting the cell and enabling the flag.
- Keep the task, data, and acceptance rules fixed and change only the lever being tested. Model selection and routing deliberately vary the model; prompt comparisons keep the model/API fixed. Keep unsuccessful or inconclusive outcomes visible.

## Prerequisites

- Completed [Part 1: Fundamentals](../01-fundamentals/)
- The provisioned Python 3.13 Code Editor environment. For optional local setup, run `uv sync --frozen --python 3.13 --extra notebook --extra langfuse` from the root; see the [repository setup](../README.md).
- For pip setup, run `pip install --require-hashes -r requirements-langfuse.lock`, then install the shared package with `pip install --no-deps -e .`.
- AWS credentials and the Bedrock/AgentCore access required by the selected exercises.
- A fresh kernel with `OBSERVABILITY_BACKEND=agentcore` for the first lesson path. Langfuse is already provisioned and its client installed. Follow **Langfuse Project Setup** in your workshop guide to create project API keys, select `langfuse` or `both`, then restart the kernel to switch; no additional install is needed.
