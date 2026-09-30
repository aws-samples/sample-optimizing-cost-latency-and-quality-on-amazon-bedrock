# Optimizing Cost, Latency, and Quality on Amazon Bedrock

Build and measure an agentic application on Amazon Bedrock, then improve its cost and latency while checking answer quality.

This repository contains the runnable notebooks. At an AWS event, follow the walkthrough opened from your event dashboard. The [published AWS Workshop Studio guide](https://catalog.us-east-1.prod.workshops.aws/workshops/60d21a0a-c56f-47aa-9e5d-45181cd42507/en-US) is also available for self-paced use.

**Audience:** AI/ML developers, software engineers, and DevOps engineers working with agentic applications. Basic Python familiarity is expected; the optional Jupyter introduction covers notebook basics.

**Duration:** approximately **6.5 hours**. Use the Workshop Studio agenda to choose optional exercises and deep dives.

## What you will learn

| Objective | What you measure |
| --- | --- |
| Cost | Model input, output, cache writes/reads, and the cost of all attempts per solved task |
| Latency | End-to-end response time and, in streaming exercises, time to first visible token |
| Quality | Correctness against held-out references and, where agent spans are available, tool trajectories |

Keep the task and evaluation rules fixed, and change only the lever being tested. Model selection and routing deliberately vary the model; other comparisons keep the model/API/profile fixed. Account for evaluator, tool, and infrastructure charges separately from model serving cost. Saved outputs describe their recorded run; your results can differ.

## Workshop path

Start with AWS-native observability and AgentCore evaluation. **Langfuse is provisioned up front and available as an alternative or fallback**, with its client already installed.

| Track | Notebooks | Focus |
| --- | --- | --- |
| Fundamentals | [01-fundamentals](01-fundamentals/) | Tokens, APIs, pricing, traces, and a first evaluation |
| Optimization Playbook | [02-optimization-playbook](02-optimization-playbook/) | 16 levers: five LOW, six MEDIUM, and five HIGH effort |
| Developer Journey | [03-developer-journey](03-developer-journey/) | TechMart agent versions, held-out evaluation, and optimization from traces |
| Optional deep dives | [04-deep-dive-topics](04-deep-dive-topics/) | Advanced caching and managed prompt lifecycle |
| Cleanup | [99-cleanup](99-cleanup.ipynb) | Review and remove resources recorded for your run |

Model calls, evaluation, and managed resources incur charges. Optional paid exercises are labelled and have explicit controls. An interrupted notebook or polling timeout does not cancel a server-side job; keep resource records for cleanup.

## Start in the provisioned Code Editor

Workshop setup prepares **Cognito, core infrastructure, Langfuse, Code Editor, and the prompt quality gate** before the lessons. The Code Editor environment includes Python 3.13, the locked notebook dependencies, the shared `workshop_utils` package, and both AWS-native and Langfuse clients.

1. Open Code Editor from the workshop environment.
2. Open this repository and select **Bedrock Workshop (Python 3.13)** as the notebook kernel.
3. Keep `OBSERVABILITY_BACKEND=agentcore` for the first lesson path.
4. Start [Jupyter Notebook 101](01-fundamentals/00-jupyter-notebook-101.ipynb), or go directly to [Prompts 101](01-fundamentals/01-prompts-101.ipynb) if you know Jupyter.

Use a fresh kernel for each notebook and run cells in order. The notebooks use the AWS credential chain supplied by the environment and load `.env` without replacing supplied settings.

For a self-paced workshop in your own account, follow the [Workshop Studio setup](https://catalog.us-east-1.prod.workshops.aws/workshops/60d21a0a-c56f-47aa-9e5d-45181cd42507/en-US). From the repository root in **CloudShell or your local terminal**, deploy the five shared stacks:

```bash
make -C 03-developer-journey deploy-all \
  CODE_EDITOR_REPO_REF="$(git rev-parse HEAD)"
```

This selects the same repository commit for Code Editor as your current checkout. Review the selected account and Region before deployment; infrastructure starts incurring charges when created. Wait for deployment to finish, then open the provisioned Code Editor using its stack outputs. Keep the original CloudShell or local terminal available for teardown.

## Optional local environment

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) from the repository root:

```bash
uv sync --frozen --python 3.13 --extra notebook --extra langfuse
uv run --no-sync python -m ipykernel install --user \
  --name bedrock-workshop-py313 \
  --display-name "Bedrock Workshop (Python 3.13)"
uv run --no-sync jupyter lab
```

The command creates the Python 3.13 environment and installs both client sets together from `uv.lock`. Use `--no-sync` for subsequent `uv run` commands to retain that environment. Select the registered kernel in Jupyter.

For an environment that requires pip, use a Python 3.13 virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-langfuse.lock
python -m pip install --no-deps -e .
```

On Windows, activate with `.venv\Scripts\activate` instead. The lock includes the notebook and Langfuse dependencies; the editable install makes the shared workshop package available from every notebook directory.

Configure an AWS CLI profile or the provided temporary credentials for the workshop account, in `us-east-1` unless directed otherwise. If using `.env`, copy `.env.example` and replace its sample values with the supplied settings, including `AWS_SESSION_TOKEN` for temporary credentials. Keep credentials out of notebook outputs.

Fundamentals includes a small billable model-connection check. Playbook setup displays model selections and capabilities; its exercise cells make the requests. Creating a boto3 client alone does not test connectivity. Use the workshop's selected Sonnet 5, Haiku 4.5, Opus 5 and GPT-5.6 examples, keeping the model fixed unless model selection or routing is the lever being compared.

## Switch to Langfuse

Open the provisioned Langfuse URL and follow **Langfuse Project Setup** in your workshop guide to create project API keys. Save `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY` through the environment or `.env`. Set the backend in the repository's `.env`:

```dotenv
OBSERVABILITY_BACKEND=langfuse
```

Use `both` to export to AWS and Langfuse together. **Restart the kernel and run from the top; no additional installation is needed.** Notebook setup honors this backend choice over Code Editor's default while retaining supplied AWS credentials. Configure telemetry before creating instrumented clients. Find the same session/trace IDs in the selected backend.

The backend setting selects trace export. AWS managed evaluation and promotion exercises still use AgentCore APIs; Langfuse trace views and prompt labels do not replace those gates. The workshop Langfuse server is v3: SDK v4 supports ingestion, while its newer observation/metrics read APIs require server v4. Use the documented v3-compatible reads.

Developer Journey deployments have a separate `RUNTIME_BACKEND` setting and build locked Linux ARM64 artifacts using the provisioned execution role and artifact bucket.

## Finish and clean up

Complete [99-cleanup](99-cleanup.ipynb) with your saved creation records. Stop recurring evaluations, A/B tests, and builds before removing their dependencies. Shut down observability in every active kernel before shared stack teardown; closing a notebook tab alone does not stop its exporters.

The cleanup notebook verifies ownership and removes only recorded run resources. For self-paced infrastructure teardown, first complete those steps, wait for deletion to finish, stop active jobs/builds, and retain any results you need outside Code Editor. Then return to the **original CloudShell or local terminal, outside the Code Editor instance**, and run from the repository root:

```bash
make -C 03-developer-journey delete-all RUNTIME_RESOURCES_CLEANED=1
```

This deletes the gate, Code Editor, Langfuse, core, and Cognito stacks in that order. **Do not run it inside Code Editor:** deleting that instance would terminate the command before the remaining stacks are removed. The template-upload bucket and account-level Transaction Search settings are retained.

## Helper organization

Helpers specific to a lesson live in that section's `utils/` directory:

| Section | Helpers |
| --- | --- |
| Fundamentals | Introductory pricing and Langfuse convenience functions |
| Optimization Playbook | Support tools, cache metrics and Gateway result parsing |
| Developer Journey | Agent configuration, deployment packaging, Gateway setup, evaluation workflow and resource cleanup |
| Deep Dives | Langfuse text-dataset compatibility used by the lifecycle notebook and evaluation script |

`workshop_utils/` holds the implementations used across sections: model capabilities and selection, pricing and request validation, telemetry and metrics, evaluation and trace collection, retrieval, and the managed lifecycle gates. These remain shared so the Playbook, Journey and deep dives use the same accounting and API behavior. The notebooks show the lesson's requests and settings; helper code handles repeated setup, serialization, polling and validation.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
