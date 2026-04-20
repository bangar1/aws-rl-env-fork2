# Load .env if present so Make sees every KEY=value in it, both as $(VAR)
# inside the Makefile and as an environment variable exported to recipes.
# Precedence (highest → lowest):
#   1. CLI:           make run POOL_SIZE=16
#   2. Shell env:     POOL_SIZE=16 make run
#   3. .env file:     AWS_RL_ENV_POOL_SIZE=16 in ./.env
#   4. Makefile default (via ?=)
# Create one by copying the template:  cp .env.example .env
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Project settings
PROJECT_NAME := openenv-aws_rl_env
PYTHON := python3
UV := uv
DOCKER_IMAGE := aws-rl-env
DOCKER_TAG := latest
SERVER_HOST := 0.0.0.0
SERVER_PORT := 8000

# Parallel MiniStack pool (used by `make run`).
#   POOL_SIZE=1           → single MiniStack on MINISTACK_BASE_PORT (legacy behavior)
#   POOL_SIZE>1           → N MiniStacks on MINISTACK_BASE_PORT..BASE_PORT+N-1,
#                           server exposes N concurrent WebSocket sessions
# Override from CLI: `make run POOL_SIZE=8` or `POOL_SIZE=8 make run`.
POOL_SIZE ?= $(or $(AWS_RL_ENV_POOL_SIZE),1)
MINISTACK_BASE_PORT ?= $(or $(AWS_RL_ENV_MINISTACK_BASE_PORT),4566)

.DEFAULT_GOAL := help

# ──────────────────────────────────────────────
# Setup & Dependencies
# ──────────────────────────────────────────────

.PHONY: install
install: ## Install project dependencies
	$(UV) sync --frozen

.PHONY: install-dev
install-dev: ## Install project with dev dependencies
	$(UV) sync --frozen --extra dev


.PHONY: install-all
install-all: ## Install project with all dependencies (dev + training)
	$(UV) sync --frozen --all-extras

.PHONY: lock
lock: ## Update the lockfile
	$(UV) lock

# ──────────────────────────────────────────────
# Development
# ──────────────────────────────────────────────

.PHONY: run
run: ## Run MiniStack pool + FastAPI server. Env: POOL_SIZE (default 1), MINISTACK_BASE_PORT (default 4566)
	@echo "==> Starting $(POOL_SIZE) MiniStack(s) on ports $(MINISTACK_BASE_PORT)..$$(($(MINISTACK_BASE_PORT) + $(POOL_SIZE) - 1))"
	@for i in $$(seq 0 $$(($(POOL_SIZE) - 1))); do \
		port=$$(($(MINISTACK_BASE_PORT) + $$i)); \
		echo "    MiniStack :$$port"; \
		GATEWAY_PORT=$$port aws_infra -d; \
	done
	@sleep 2
	@echo "==> FastAPI server on $(SERVER_HOST):$(SERVER_PORT) (POOL_SIZE=$(POOL_SIZE))"
	AWS_RL_ENV_POOL_SIZE=$(POOL_SIZE) \
	AWS_RL_ENV_MINISTACK_BASE_PORT=$(MINISTACK_BASE_PORT) \
	$(UV) run uvicorn server.app:app --host $(SERVER_HOST) --port $(SERVER_PORT) --reload

.PHONY: run-stop
run-stop: ## Stop every MiniStack started by `make run` (uses current POOL_SIZE + MINISTACK_BASE_PORT)
	@for i in $$(seq 0 $$(($(POOL_SIZE) - 1))); do \
		port=$$(($(MINISTACK_BASE_PORT) + $$i)); \
		echo "    stopping MiniStack :$$port"; \
		GATEWAY_PORT=$$port aws_infra --stop || true; \
	done

# ──────────────────────────────────────────────
# Code Quality
# ──────────────────────────────────────────────

.PHONY: format
format: ## Format code with ruff
	$(UV) run ruff format .

.PHONY: lint
lint: ## Lint code with ruff
	$(UV) run ruff check .

.PHONY: lint-fix
lint-fix: ## Lint and auto-fix code with ruff
	$(UV) run ruff check --fix .

.PHONY: typecheck
typecheck: ## Run type checking with mypy
	$(UV) run mypy

.PHONY: check
check: lint typecheck

# ──────────────────────────────────────────────
# Docker
# ──────────────────────────────────────────────

.PHONY: docker-build
docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

.PHONY: docker-run
docker-run: ## Run Docker container
	docker run --rm --name $(DOCKER_IMAGE) -p $(SERVER_PORT):8000 $(DOCKER_IMAGE):$(DOCKER_TAG)

.PHONY: docker-run-dev
docker-run-dev: ## Run Docker container in dev mode with live reload
	docker run --rm --name $(DOCKER_IMAGE) -p $(SERVER_PORT):8000 -v $(PWD):/app/env -v /app/env/.venv -e DEV_MODE=1 $(DOCKER_IMAGE):$(DOCKER_TAG)

.PHONY: docker-run-detach
docker-run-detach: ## Run Docker container in background
	docker run -d --rm --name $(DOCKER_IMAGE) -p $(SERVER_PORT):8000 -v $(PWD):/app/env -v /app/env/.venv -e DEV_MODE=1 $(DOCKER_IMAGE):$(DOCKER_TAG)

.PHONY: docker-stop
docker-stop: ## Stop the running Docker container
	docker stop $(DOCKER_IMAGE)

.PHONY: docker-logs
docker-logs: ## Tail logs from the running Docker container
	docker logs -f $(DOCKER_IMAGE)

.PHONY: docker-shell
docker-shell: ## Open a shell in the running Docker container
	docker exec -it $(DOCKER_IMAGE) /bin/bash

.PHONY: docker-clean
docker-clean: ## Stop and remove all running containers for this image
	@docker ps -q --filter ancestor=$(DOCKER_IMAGE):$(DOCKER_TAG) | xargs -r docker rm -f

.PHONY: docker-test
docker-test: ## Run tests inside the running Docker container
	docker exec $(DOCKER_IMAGE) python -m pytest env/tests -v

.PHONY: docker-health
docker-health: ## Check health of the running container
	@curl -sf http://localhost:$(SERVER_PORT)/health && echo " OK" || echo " FAIL"

# ──────────────────────────────────────────────
# OpenEnv
# ──────────────────────────────────────────────

.PHONY: openenv-validate
openenv-validate: ## Validate the OpenEnv configuration
	openenv validate

.PHONY: openenv-build
openenv-build: ## Build the environment using OpenEnv CLI
	openenv build

.PHONY: openenv-push
openenv-push: ## Push the environment to Hugging Face Spaces
	openenv push

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────

.PHONY: clean
clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .eggs/
	rm -rf aws_infra/*.egg-info aws_infra/build/ aws_infra/dist/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

.PHONY: clean-all
clean-all: clean ## Remove all artifacts including venv
	rm -rf .venv/

# ──────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
