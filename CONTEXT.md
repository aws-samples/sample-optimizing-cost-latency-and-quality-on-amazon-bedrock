# Workshop glossary

These terms are used across the notebooks, agent implementations and evaluations.

| Term | Meaning |
| --- | --- |
| Agent version (v1–v6) | A stage of the TechMart agent: baseline, quick wins, caching, routing, guardrails, then skills and Gateway discovery. |
| Operational metrics | Model calls, token/cache usage, model cost and elapsed time associated with an invocation. |
| Quality metrics | Scores against the task's requirements, such as answer correctness and expected tool use. |
| Test scenario | A query with declared expectations in `03-developer-journey/data/test_scenarios.json`. |
| Reference answer | The expected facts or behavior used to evaluate a scenario. Review references when changing the scenario or tool data. |
| Training split | Cases used to generate an optimization recommendation. |
| Held-out split | Separate cases used to evaluate the control and candidate. |
| Retrieved context | Knowledge Base passages, scores and source locations returned for a support query. |
| Session and trace | Identifiers that connect an invocation to its model, tool and agent spans. Local and remotely deployed invocations have separate traces. |
| Configuration bundle | Versioned runtime configuration, including the prompt and tool descriptions consumed by the agent. |
| Control and candidate | The current configuration and a proposed change evaluated on the same workload. |
| Quality gate | A check that requires complete evaluation evidence and the declared score floor before progressing. |
| A/B experiment | Fresh sessions routed between control and candidate after the offline gate. |
| Promotion and rollback | Creating a mainline configuration version from the accepted candidate or a selected previous version. |

AWS-native telemetry and AgentCore Evaluations are the primary path. The provisioned Langfuse service provides an alternative for inspecting traces, metrics and prompts. Use the notebook's selected backend and measurement scope when comparing results.
