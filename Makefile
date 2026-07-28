# Agentic-RAG-Bench — developer entry points.
# Requires uv (https://docs.astral.sh/uv/). `make setup` bootstraps everything.

.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test check ingest benchmark experiment evaluate report docker-build clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and pre-commit hooks
	uv sync
	uv run pre-commit install

lint: ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format and fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## mypy (strict)
	uv run mypy

test: ## Run the test suite with coverage
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test ## Everything CI runs, locally

ingest: ## Ingest the corpus (Phase 2)
	uv run python scripts/ingest_corpus.py

benchmark: ## Build the benchmark tasks (Phase 5)
	uv run python scripts/build_benchmark.py

experiment: ## Run the experiment matrix (Phase 6)
	uv run python scripts/run_experiment.py

evaluate: ## Evaluate stored experiment results (Phase 5/6)
	uv run python scripts/run_experiment.py --evaluate-only

report: ## Generate the comparison report (Phase 7)
	uv run python scripts/generate_report.py

docker-build: ## Build the Docker image
	docker build -t agentic-rag-bench:dev .

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
