# Self-paced deployment uses the same templates as Workshop Studio.
# Core infrastructure builds its own Lambda code and pinned SDK layers.
# Read Metadata.WorkshopMigration in prerequisite/infrastructure.yaml before
# upgrading an existing stack that used the legacy custom-resource templates.
# Optional ../.env values override shell variables; command-line make values win.
# No AWS command is evaluated while parsing this file, including during make -n.
-include ../.env

SHELL := /bin/sh
.DEFAULT_GOAL := help
export AWS_PAGER :=

# Empty profile uses the normal AWS credential chain. Exporting AWS_PROFILE=""
# would make the CLI look for a profile with an empty name, so pass it only as an
# explicit CLI option when nonempty. Inherited credential variables still work.
AWS_PROFILE ?=
ifneq ($(strip $(AWS_PROFILE)),)
export AWS_PROFILE
else
unexport AWS_PROFILE
endif
AWS_REGION ?= $(if $(AWS_DEFAULT_REGION),$(AWS_DEFAULT_REGION),us-east-1)
AWS ?= aws
AWS_CLI = $(AWS) --region "$(AWS_REGION)" $(if $(strip $(AWS_PROFILE)),--profile "$(AWS_PROFILE)",)

COGNITO_STACK_NAME ?= prompt-workshop-cognito
INFRA_STACK_NAME ?= prompt-workshop-infra
LANGFUSE_STACK_NAME ?= prompt-workshop-langfuse
CODE_EDITOR_STACK_NAME ?= prompt-workshop-code-editor
GATE_STACK_NAME ?= prompt-workshop-gate
# Maintainers may select a published branch, tag or commit; participants use main.
CODE_EDITOR_REPO_REF ?= main
# Leave empty to create/reuse <account>-<region>-workshop-cfn-templates.
# A supplied bucket must already exist in the caller's account. Existing buckets
# are never reconfigured, emptied or deleted by this Makefile.
CFN_S3_BUCKET ?=
CFN_S3_PREFIX ?= workshop-templates/$(INFRA_STACK_NAME)
RUNTIME_RESOURCES_CLEANED ?= 0

# Resolve account-dependent defaults only when a recipe actually runs.
define resolve_template_bucket
account=$$($(AWS_CLI) sts get-caller-identity --query Account --output text); \
case "$$account" in ''|*[!0-9]*) echo "Invalid AWS account ID" >&2; exit 1;; esac; \
test "$${#account}" -eq 12 || { echo "Invalid AWS account ID" >&2; exit 1; }; \
bucket="$(CFN_S3_BUCKET)"; \
if [ -z "$$bucket" ]; then bucket="$$account-$(AWS_REGION)-workshop-cfn-templates"; fi; \
case "$(CFN_S3_PREFIX)" in ''|/|/*) echo "CFN_S3_PREFIX must be a non-empty relative template prefix" >&2; exit 1;; esac
endef

.PHONY: help check-env prepare-template-bucket create-bucket \
	deploy-all deploy-cognito deploy-infra deploy-langfuse deploy-code-editor deploy-gate \
	check-runtime-cleanup delete-all delete-gate delete-code-editor delete-langfuse delete-infra delete-cognito \
	status status-cognito status-infra status-langfuse status-code-editor status-gate

help:
	@echo "Self-paced workshop infrastructure"
	@echo "  make deploy-all       Deploy Cognito -> core -> Langfuse -> CodeEditor -> prompt quality gate"
	@echo "  make deploy-cognito   Deploy only Cognito"
	@echo "  make deploy-infra     Deploy only core (Cognito should already exist)"
	@echo "  make deploy-langfuse  Deploy Langfuse for the comparison labs"
	@echo "  make deploy-code-editor Deploy CodeEditor with the full workshop dependencies"
	@echo "  make deploy-gate      Deploy only the prompt quality gate"
	@echo "  make status           Show all five stack statuses"
	@echo "  make status-cognito / status-infra / status-langfuse / status-code-editor / status-gate"
	@echo "                        Show an individual stack status"
	@echo "  make delete-all       Delete gate -> CodeEditor -> Langfuse -> core -> Cognito"
	@echo "  make delete-gate / delete-code-editor / delete-langfuse / delete-infra / delete-cognito"
	@echo "                        Delete an individual stack (cleanup guard applies)"
	@echo "  make check-env        Verify the selected AWS identity and region"
	@echo "  make -n deploy-all    Preview commands without making AWS calls"
	@echo ""
	@echo "All five stacks are provisioned up front; AWS-native telemetry remains the default."
	@echo "CodeEditor source: CODE_EDITOR_REPO_REF=$(CODE_EDITOR_REPO_REF) (default: main)."
	@echo "AWS_PROFILE is optional; default credentials are used when it is empty."
	@echo "AWS_REGION=$(AWS_REGION)"
	@echo "Set CFN_S3_BUCKET to reuse a bucket; uploads use $(CFN_S3_PREFIX)/."
	@echo "The template upload bucket and its objects are preserved on stack deletion."
	@echo ""
	@echo "First clean up recorded notebook A/B tests, evaluations, targets, Gateways,"
	@echo "Runtimes, artifacts and guardrails; wait for asynchronous deletion to finish."
	@echo "Stop active gate builds and batch jobs before deleting their infrastructure."
	@echo "Then: make delete-all RUNTIME_RESOURCES_CLEANED=1"

check-env:
	@test -n "$(AWS_REGION)" || { echo "AWS_REGION must not be empty" >&2; exit 1; }
	@echo "Verifying AWS credentials in $(AWS_REGION)..."
	@$(AWS_CLI) sts get-caller-identity

prepare-template-bucket: check-env
	@set -eu; \
	$(resolve_template_bucket); \
	if details=$$($(AWS_CLI) s3api head-bucket --bucket "$$bucket" --expected-bucket-owner "$$account" 2>&1); then \
		echo "Reusing template bucket $$bucket without changing its configuration"; \
	else \
		if [ -n "$(CFN_S3_BUCKET)" ]; then \
			echo "CFN_S3_BUCKET must already exist and be accessible in account $$account." >&2; \
			printf '%s\n' "$$details" >&2; exit 1; \
		fi; \
		case "$$details" in \
			*"(404)"*|*"NoSuchBucket"*) ;; \
			*) printf '%s\n' "$$details" >&2; \
				echo "Bucket access failed; refusing to create or reconfigure it." >&2; exit 1;; \
		esac; \
		echo "Creating template upload bucket $$bucket"; \
		if [ "$(AWS_REGION)" = "us-east-1" ]; then \
			$(AWS_CLI) s3api create-bucket --bucket "$$bucket"; \
		else \
			$(AWS_CLI) s3api create-bucket --bucket "$$bucket" \
				--create-bucket-configuration "LocationConstraint=$(AWS_REGION)"; \
		fi; \
	fi

# Compatibility alias: this prepares templates, not prepackaged Lambda ZIPs.
create-bucket: prepare-template-bucket

deploy-cognito delete-cognito status-cognito: CFN_STACK = $(COGNITO_STACK_NAME)
deploy-infra delete-infra status-infra: CFN_STACK = $(INFRA_STACK_NAME)
deploy-langfuse delete-langfuse status-langfuse: CFN_STACK = $(LANGFUSE_STACK_NAME)
deploy-code-editor delete-code-editor status-code-editor: CFN_STACK = $(CODE_EDITOR_STACK_NAME)
deploy-gate delete-gate status-gate: CFN_STACK = $(GATE_STACK_NAME)
deploy-cognito: CFN_TEMPLATE = cognito.yaml
deploy-infra: CFN_TEMPLATE = infrastructure.yaml
deploy-langfuse: CFN_TEMPLATE = langfuse-ecs.yaml
deploy-code-editor: CFN_TEMPLATE = code-editor.yaml
deploy-gate: CFN_TEMPLATE = prompt-pipeline.yaml
# Provision the comparison service from the start while keeping the primary
# lesson on native telemetry. The canonical CodeEditor template installs the
# full notebook and Langfuse dependencies independently of telemetry selection.
deploy-langfuse: CFN_PARAMETERS = --parameter-overrides "DeployLangfuse=true"
deploy-code-editor: CFN_PARAMETERS = --parameter-overrides \
	"RepoRef=$(CODE_EDITOR_REPO_REF)" "ObservabilityBackend=agentcore" \
	"InfrastructureStackName=$(INFRA_STACK_NAME)"

deploy-cognito deploy-infra deploy-langfuse deploy-code-editor deploy-gate: prepare-template-bucket
	@set -eu; \
	$(resolve_template_bucket); \
	$(AWS_CLI) cloudformation deploy \
		--template-file "prerequisite/$(CFN_TEMPLATE)" \
		--stack-name "$(CFN_STACK)" \
		--capabilities CAPABILITY_NAMED_IAM \
		--s3-bucket "$$bucket" --s3-prefix "$(CFN_S3_PREFIX)/$(basename $(CFN_TEMPLATE))" \
		$(CFN_PARAMETERS) \
		--no-fail-on-empty-changeset
	@$(AWS_CLI) cloudformation describe-stacks --stack-name "$(CFN_STACK)" \
		--query 'Stacks[0].Outputs' --output table

# Recursive recipes preserve order even with make -j. During make -n each child
# inherits dry-run mode and only prints its AWS commands.
deploy-all:
	@$(MAKE) --no-print-directory deploy-cognito
	@$(MAKE) --no-print-directory deploy-infra
	@$(MAKE) --no-print-directory deploy-langfuse
	@$(MAKE) --no-print-directory deploy-code-editor
	@$(MAKE) --no-print-directory deploy-gate
	@echo "Cognito, core, Langfuse, CodeEditor and the prompt quality gate are ready."
	@echo "Start the notebooks with AWS-native telemetry; Langfuse is available for comparison."

check-runtime-cleanup:
	@if [ "$(RUNTIME_RESOURCES_CLEANED)" != "1" ]; then \
		echo "Complete notebook cleanup using saved resource IDs before stack deletion." >&2; \
		echo "Wait for Runtime/target/evaluation deletion and stop active jobs/builds." >&2; \
		echo "Keep those records until cleanup finishes; see this journey's README." >&2; \
		echo "Then rerun with RUNTIME_RESOURCES_CLEANED=1." >&2; exit 1; \
	fi

delete-gate delete-code-editor delete-langfuse delete-infra delete-cognito: check-runtime-cleanup check-env
	$(AWS_CLI) cloudformation delete-stack --stack-name "$(CFN_STACK)"
	$(AWS_CLI) cloudformation wait stack-delete-complete --stack-name "$(CFN_STACK)"

delete-all: check-runtime-cleanup
	@$(MAKE) --no-print-directory delete-gate
	@$(MAKE) --no-print-directory delete-code-editor
	@$(MAKE) --no-print-directory delete-langfuse
	@$(MAKE) --no-print-directory delete-infra
	@$(MAKE) --no-print-directory delete-cognito
	@echo "Stacks deleted. The template upload bucket and prefix are preserved."
	@echo "Account-level Transaction Search settings are preserved."

status-cognito status-infra status-langfuse status-code-editor status-gate: check-env
	@$(AWS_CLI) cloudformation describe-stacks --stack-name "$(CFN_STACK)" \
		--query 'Stacks[0].{Name:StackName,Status:StackStatus,Created:CreationTime}' --output table

status:
	@$(MAKE) --no-print-directory status-cognito
	@$(MAKE) --no-print-directory status-infra
	@$(MAKE) --no-print-directory status-langfuse
	@$(MAKE) --no-print-directory status-code-editor
	@$(MAKE) --no-print-directory status-gate
