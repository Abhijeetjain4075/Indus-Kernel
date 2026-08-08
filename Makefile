# Indus Kernel — Makefile
# Single entrypoint for development, testing, and deployment.

.PHONY: help setup deps-up deps-down migrate dev test lint format typecheck \
        bench chaos hello clean build docker-build docker-push helm-package \
        release adr adr-new docs-api docs-serve

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ============================================================================
# Help
# ============================================================================
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# Setup
# ============================================================================
setup:  ## Install development dependencies (uv, pre-commit, etc.)
	@echo "==> Installing uv"
	@if ! command -v uv >/dev/null 2>&1; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo 'export PATH="$$HOME/.local/bin:$$PATH"' >> $$HOME/.bashrc; \
	fi
	@echo "==> Installing pre-commit"
	@uv tool install pre-commit
	@echo "==> Installing project (editable, all extras)"
	@uv sync --all-extras --dev
	@echo "==> Installing pre-commit hooks"
	@uv run pre-commit install
	@echo "==> Setup complete. Run 'make dev' to start."

# ============================================================================
# Dependencies
# ============================================================================
deps-up:  ## Start local dependencies (Postgres, Redis, NATS, Qdrant, Neo4j, Temporal)
	@echo "==> Starting dependencies"
	docker compose up -d postgres redis nats qdrant neo4j temporal temporal-ui
	@echo "==> Waiting for services to be healthy"
	@uv run python scripts/wait_for_services.py
	@echo "==> Dependencies ready"

deps-down:  ## Stop local dependencies
	docker compose down

deps-logs:  ## Tail logs from dependencies
	docker compose logs -f --tail=100

deps-ps:  ## Show status of dependencies
	docker compose ps

# ============================================================================
# Database
# ============================================================================
migrate:  ## Run database migrations
	@echo "==> Running Alembic migrations"
	@uv run alembic upgrade head
	@echo "==> Seeding Neo4j"
	@uv run python -c "from ik_memory.adapters.neo4j import seed_schema; seed_schema()"
	@echo "==> Migrations complete"

migrate-new:  ## Create a new Alembic migration
	@uv run alembic revision --autogenerate -m "$(name)"

migrate-rollback:  ## Rollback last migration
	@uv run alembic downgrade -1

db-shell:  ## Open psql shell
	docker compose exec postgres psql -U indus -d indus

# ============================================================================
# Development
# ============================================================================
dev:  ## Start the kernel in dev mode (hot reload)
	@echo "==> Starting Indus Kernel in dev mode"
	@uv run ik-kernel dev --reload

dev-debug:  ## Start with debugpy
	@uv run ik-kernel dev --reload --debug

shell:  ## Open an IPython shell with the kernel loaded
	@uv run ipython -i scripts/shell.py

# ============================================================================
# Testing
# ============================================================================
test:  ## Run all tests
	@echo "==> Running unit tests"
	@uv run pytest tests/unit -x --tb=short
	@echo "==> Running integration tests"
	@uv run pytest tests/integration -x --tb=short
	@echo "==> Running E2E tests"
	@uv run pytest tests/e2e -x --tb=short

test-unit:  ## Run unit tests only
	@uv run pytest tests/unit -x --tb=short

test-integration:  ## Run integration tests only
	@uv run pytest tests/integration -x --tb=short

test-e2e:  ## Run E2E tests only
	@uv run pytest tests/e2e -x --tb=short

test-coverage:  ## Run tests with coverage report
	@uv run pytest --cov=packages --cov-report=html --cov-report=term

test-watch:  ## Run tests in watch mode
	@uv run pytest-watch

hello:  ## Smoke test the hello-world agent
	@echo "==> Calling hello-world agent"
	@curl -s -X POST http://localhost:8000/api/v1/agents/runs \
		-H "Content-Type: application/json" \
		-d '{"goal": "Introduce Indus Kernel"}' | jq .

# ============================================================================
# Linting & Formatting
# ============================================================================
lint:  ## Run all linters
	@uv run ruff check packages apps
	@uv run mypy packages

format:  ## Format code
	@uv run ruff format packages apps
	@uv run ruff check --fix packages apps

typecheck:  ## Run type checker
	@uv run mypy packages

pre-commit:  ## Run pre-commit on all files
	@uv run pre-commit run --all-files

# ============================================================================
# Benchmarks
# ============================================================================
bench:  ## Run performance benchmarks
	@uv run pytest tests/benchmark -x --tb=short --benchmark-only

bench-router:  ## Benchmark LLM Router throughput
	@uv run python scripts/bench_router.py

bench-memory:  ## Benchmark Memory Engine
	@uv run python scripts/bench_memory.py

# ============================================================================
# Chaos
# ============================================================================
chaos:  ## Run chaos tests (requires deps running)
	@uv run pytest tests/chaos -x --tb=short

# ============================================================================
# Docker
# ============================================================================
docker-build:  ## Build the kernel Docker image
	docker build -t indus-kernel:latest -f docker/Dockerfile .

docker-build-multi:  ## Build multi-arch (amd64 + arm64)
	docker buildx build --platform linux/amd64,linux/arm64 \
		-t indus-kernel:latest -f docker/Dockerfile .

docker-push:  ## Push image to registry
	docker push indus-kernel:latest

docker-run:  ## Run kernel in Docker (with deps)
	docker compose up

# ============================================================================
# Helm
# ============================================================================
helm-package:  ## Package Helm chart
	@cd charts/indus-kernel && helm package .

helm-lint:  ## Lint Helm chart
	@cd charts/indus-kernel && helm lint .

# ============================================================================
# Docs
# ============================================================================
docs-api:  ## Generate API docs from OpenAPI
	@uv run python scripts/generate_api_docs.py

docs-serve:  ## Serve docs locally
	@uv run mkdocs serve

# ============================================================================
# ADRs
# ============================================================================
adr:  ## List all ADRs
	@ls -1 docs/adr/ | sort

adr-new:  ## Create a new ADR
	@read -p "ADR number: " n; \
	read -p "ADR title: " t; \
	slug=$$(echo $$t | tr '[:upper:]' '[:lower:]' | tr ' ' '-'); \
	uv run python scripts/new_adr.py $$n "$$t" "$$slug"

# ============================================================================
# Release
# ============================================================================
build:  ## Build distribution packages
	@uv build
	@echo "==> Built packages in dist/"

release:  ## Tag and release
	@read -p "Version (e.g. 0.1.0): " v; \
	git tag -a v$$v -m "Release v$$v"; \
	git push origin v$$v; \
	echo "==> Released v$$v"

# ============================================================================
# Utilities
# ============================================================================
clean:  ## Clean build artifacts
	@rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "==> Cleaned"

clean-all: clean deps-down  ## Deep clean (incl. docker)
	@docker system prune -f
	@echo "==> Deep cleaned"

deps: setup deps-up migrate  ## Full dependency setup
	@echo "==> All dependencies ready"
