# Part 4: Deep Dives

## Overview

Two standalone deep-dive topics that don't belong to the main Fundamentals → Playbook → Developer Journey progression — each self-contained: complex caching patterns and production prompt lifecycle management. (Advanced prompt-engineering techniques — CoT, Self-Refine, CoD, Verbalized Sampling — now live in the Optimization Playbook, Lever 05, alongside the managed Bedrock Advanced Prompt Optimization tool.)

**Duration**: ~1 hour

---

## Learning Objectives

After completing these deep dives, you will:
- Implement multi-checkpoint caching patterns with proper static/dynamic separation
- Choose appropriate cache TTL strategies (5-minute vs 1-hour) and debug low hit rates
- Version and manage prompts in Langfuse with production labels
- Build evaluation datasets and run LLM-as-Judge scoring
- Integrate prompt evaluation into a CI/CD pipeline with quality gates

## Prerequisites

- Complete [01-fundamentals](../01-fundamentals/) first
- Complete [03-developer-journey](../03-developer-journey/) for Langfuse setup (required for notebook 03)
- AWS Account with Amazon Bedrock access
- Python 3.10+

## Notebooks

| Notebook | Duration | Description |
|----------|----------|-------------|
| [01-advanced-prompt-caching](./01-advanced-prompt-caching.ipynb) | 60 min | Multi-checkpoint patterns, cache strategies |
| [02-production-prompt-lifecycle](./02-production-prompt-lifecycle.ipynb) | 75 min | Langfuse prompt management, evaluation datasets, LLM-as-Judge, CI/CD |

---

### **01-advanced-prompt-caching.ipynb** (60 min)

Multi-checkpoint caching patterns and best practices for Amazon Bedrock, including cache invalidation, TTL strategies, and performance monitoring.

**What you'll learn**:
- Cacheable content types (tools, system, messages) and assembly order
- Multi-checkpoint placement strategies (single, two, three checkpoints)
- Cumulative token counting and cache invalidation scenarios
- Static/dynamic separation to prevent cache thrashing
- 5-minute vs 1-hour TTL selection and mixed TTL strategies
- Cache performance monitoring and ROI analysis

---

### **02-production-prompt-lifecycle.ipynb** (75 min)

End-to-end production prompt lifecycle management using Langfuse for versioning, systematic evaluation with datasets, automated quality scoring with LLM-as-Judge, and CI/CD integration with AWS CodePipeline.

**What you'll learn**:
- Prompt versioning in Langfuse with production labels and trace linkage
- Creating and managing evaluation datasets (12-item customer support suite)
- Running experiments across prompt versions with side-by-side comparison
- Keyword-based and LLM-as-Judge scoring (using Opus 5 as the judge)
- Quality gate thresholds (avg score >= 0.7, pass rate >= 80%)
- CI/CD pipeline architecture: CodePipeline, CodeBuild buildspec, Lambda test action
- CloudFormation template for deploying the full pipeline
- Alternative GitHub Actions workflow reference

---

The [pipeline prerequisite template](../03-developer-journey/prerequisite/legacy-prompt-pipeline.yaml) provides the CodeCommit repository, CodeBuild project, manual approval and Langfuse promotion function used by this notebook. Workshop Studio includes these resources in its **PromptPipeline** stack. For a self-led run, deploy the standalone template before starting the CI/CD section; skip it if those resources are already provisioned.

## Files Structure

```
04-deep-dive-topics/
├── README.md
├── 01-advanced-prompt-caching.ipynb
├── 02-production-prompt-lifecycle.ipynb
├── scripts/                          # CI/CD pipeline scripts
│   ├── evaluate_prompt.py            # Evaluation runner (Langfuse + Bedrock)
│   └── check_quality_gate.py         # Quality gate checker (no dependencies)
├── data/                             # Sample data and evaluation datasets
│   ├── analyst_system_prompt_v1.txt  # Analyst prompt v1 (caching demos)
│   ├── analyst_system_prompt_v2.txt  # Analyst prompt v2 (invalidation demos)
│   ├── analyst_tools.json            # Analyst tool definitions
│   ├── company_policies.txt          # Company policies (system prompt demos)
│   ├── customer-support-eval.json    # 12-item evaluation dataset (notebook 02)
│   ├── intent_classification_eval.json # 40-item intent eval (notebook 01)
│   ├── technical_documentation.txt   # Technical docs (caching demos)
│   └── weather_tools.json            # Weather tool definitions
└── utils/                            # Shared helper modules
    ├── __init__.py
    └── cache_metrics.py              # Cache metric extraction and ROI analysis
```

---

## Next Steps

- Apply techniques to your production use cases
- Monitor cache hit rates and optimize checkpoint placement
- Set up CI/CD pipelines for automated prompt evaluation
- Build evaluation datasets from production failures and edge cases
