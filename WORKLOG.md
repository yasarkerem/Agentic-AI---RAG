# WORKLOG

Development log for **Agentic-RAG-Bench**. Newest entries at the top of each
phase section. Every substantive change should be recorded here.

---

## Phase 1 — Repository foundation

### Plan (2026-07-28)

Scope is Phase 1 only (per the implementation brief, §23): no retrieval, no
RAG pipelines, no benchmark tasks yet.

1. **Repository structure** — full directory tree from the brief (§5), with
   placeholder packages for later phases (`ingestion/`, `chunking/`,
   `embeddings/`, `retrieval/`, `reranking/`, `generation/`,
   `standard_rag/`, `agentic_rag/`, `evaluation/`).
2. **Python project configuration** — `pyproject.toml` (Python 3.12,
   hatchling, uv-managed lockfile), Ruff (lint + format), mypy in strict
   mode with the Pydantic plugin, pytest + coverage.
3. **Typed settings** — `config.py`: `AppSettings` (pydantic-settings,
   `ARB_` env prefix, `.env` support, `SecretStr` for API keys) plus typed
   experiment/system config models (`SystemConfig` and friends) loaded from
   layered YAML (`configs/default.yaml` merged with a system-specific
   overlay). Unknown keys are rejected; validation errors surface as
   `ConfigError`.
4. **Core schemas** — Pydantic v2 models under `schemas/`: documents and
   chunks (with deterministic chunk IDs), retrieval results, citations (the
   §10 shape), benchmark tasks and task levels, pipeline answers, agent
   traces, token usage + cost calculation, experiment records, and
   evaluation metric containers.
5. **Logging setup** — structlog-based structured logging in
   `observability/logging.py` with a secret-redaction processor (§15: never
   log secrets).
6. **Docker** — multi-stage `Dockerfile` (uv, python:3.12-slim) and
   `docker-compose.yml` (app + Qdrant service for Phase 2).
7. **Makefile** — `setup / lint / format / typecheck / test / check /
   docker-build` working now; `ingest / benchmark / experiment / evaluate /
   report` wired to placeholder scripts that exit non-zero with a clear
   "later phase" message (honest, not silently green).
8. **pre-commit** — hygiene hooks + Ruff lint/format.
9. **GitHub Actions** — lint, typecheck, test, and Docker build jobs.
10. **Unit-test foundation** — meaningful tests for config loading
    (including failure modes), schema invariants, chunk-ID determinism,
    citation shape, cost calculation, log redaction, and the CLI.
11. **README skeleton + docs skeletons** — §17 sections with explicit
    "no results yet" placeholders; `docs/architecture.md`,
    `docs/methodology.md`, `docs/experiment_design.md` skeletons.

Assumptions (documented per working rules):

- The repository root *is* the project root (the brief's
  `agentic-rag-bench/` tree maps onto the repo root).
- Runtime dependencies for Phase 1 are only what the foundation needs:
  `pydantic`, `pydantic-settings`, `PyYAML`, `structlog`. Retrieval/LLM
  dependencies arrive with the phases that use them.
- The CLI is argparse-based to keep dependencies minimal; it can be swapped
  later if subcommand complexity grows.

### Log

- **2026-07-28** — Phase 1 implemented as planned. Checks run locally:
  pytest **68 passed** (with branch coverage), Ruff lint + format clean,
  mypy `--strict` clean (33 files), pre-commit clean on all files,
  `docker compose config` valid, CLI smoke-tested (`arb info`,
  `arb validate-config`), placeholder make targets exit non-zero with a
  clear phase pointer as designed.
  - Fixed during self-review: `RetrievalConfig.dense_weight` default (0.5)
    contradicted its own validator when `use_bm25=false` → default is now
    1.0; `observability/logging.py` renamed to `logs.py` (Ruff A005 stdlib
    shadowing); mypy test-module override wired via
    `explicit_package_bases`; pre-commit Ruff rev aligned with the locked
    Ruff version.
  - **Environment limitation:** the full `docker build` cannot run in this
    development sandbox — its network policy denies Docker Hub / GHCR image
    pulls (proxy CONNECT 403). The Dockerfile parses cleanly and the CI
    `docker` job performs the real build + smoke test on GitHub Actions.

---

## Phase 2 — Ingestion and retrieval

_Not started. Blocked on Phase 1 approval._
