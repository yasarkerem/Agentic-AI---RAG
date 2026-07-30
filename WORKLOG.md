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

### Plan (2026-07-28)

- **Loaders** — Markdown (YAML frontmatter + heading-aware sections) and PDF
  (pypdf, page-aware). Deterministic document IDs from slugified filenames.
- **Chunking** — word-count sliding window with overlap; chunks never cross
  page/section boundaries; deterministic chunk IDs
  (`{doc}_p{page}_c{index-within-page}`); SHA-256 duplicate detection.
- **Embeddings** — `Embedder` protocol; default is a deterministic
  feature-hashing embedder (no downloads, no API, fully free). Real
  providers plug in behind the same interface later.
- **Vector index** — in-memory cosine index (numpy) persisted to disk.
  *Assumption documented:* one implementation path that runs everywhere
  (tests, CI, laptops) beats a Qdrant hard dependency now; the compose
  service stays for a later adapter.
- **BM25** — Okapi BM25 implemented in-repo (tested against hand-computed
  values); hybrid fusion = min-max normalized weighted sum.
- **Reranking** — `Reranker` protocol; default deterministic lexical-overlap
  reranker (a cross-encoder would need torch — deliberately avoided to keep
  the project free and light; the interface is ready for one).
- **Sample corpus** — original, redistributable Markdown documents about
  RAG topics under `data/corpus_sample/` (committed; `data/raw/` stays for
  users' own papers and is gitignored).
- `scripts/ingest_corpus.py` becomes real; `make ingest` works end to end.

### Log

- **2026-07-28** — Implemented as planned (commit `d52bc85`). 49 new tests.
  One fix during development: the hashed embedder needed stopword filtering
  so question boilerplate ("what is the main…") doesn't drown content terms.

---

## Phase 3 — Standard RAG baseline & Phase 4 — Agentic RAG

### Log (2026-07-28, commit `828dc1b`)

- `LLMProvider` protocol with a deterministic purpose-keyed mock (default)
  and an opt-in Anthropic SDK provider (requires explicit config + API key;
  never touched by tests). Pricing table cached from published API pricing.
- Prompts are versioned and embed a machine-readable payload so the mock
  stays deterministic while real models read the same prompt text.
- `StandardRAGPipeline`: single pass, context budget fitting, citations
  formatted verbatim from retrieved chunks (never model-invented).
- `AgenticRAGPipeline`: analyzer → planner → bounded retrieve/assess/rewrite
  loop → contradiction check → synthesize → deterministic citation
  verification → finalize. Hard stops: rounds, subquestions, token budget,
  cost budget, timeout, retry cap, and a no-progress stop when no weak
  query could be rewritten. Full `AgentTrace` + `StopReason` per run.
- 35 new tests incl. budget/timeout/retry/failure paths end to end.

---

## Phase 5 — Benchmark & evaluation, Phase 6 — Runner, Phase 7 — Reporting

### Log (2026-07-28, commit `aa5b214`)

- Benchmark v0.1: 25 tasks (5 per level) with reference answers and
  document-level gold evidence, authored against the sample corpus;
  committed JSON regenerable via `make benchmark` (drift-tested).
- Retrieval metrics (Recall@k, Precision@k, MRR, nDCG, coverage, unique
  docs), deterministic judge with documented proxy limitations, versioned
  LLM-judge abstraction (never the only evaluator).
- Experiment runner persists metadata (config snapshot, git commit,
  benchmark version, models, prompt versions, seed) + results JSONL;
  per-task crash isolation; `make evaluate` re-scores stored runs.
- Statistics: seeded bootstrap CIs; in-repo Wilcoxon signed-rank with tie
  and continuity correction, hand-verified in tests; small samples flagged.
- Report generator: metric tables, paired agentic-vs-standard tests by
  level group, five figures (validated categorical palette, direct value
  labels). Mock runs are stamped "harness validation" in the report.
- Full sample loop verified: `make ingest && make experiment && make report`
  runs 4 systems x 25 tasks and produces the report — free, no API calls.

---

## Phase 8 — Documentation & release prep

### Log (2026-07-28)

- README rewritten to reflect the completed harness, with a real example
  from an actual sample run (labeled mock) and explicit no-results framing.
- docs/architecture.md, methodology.md, experiment_design.md updated from
  skeletons to implemented state; deterministic-proxy limitation table added.
- CONTRIBUTING.md added; Dockerfile now ships the sample corpus + benchmark
  so the container can run the sample experiment.
- Final checks: full test suite, Ruff, mypy --strict, pre-commit all green.

### Definition of Done (brief §21) status

- ✅ both systems run on the same corpus (fairness-tested)
- ✅ 25 benchmark tasks, all 5 levels represented
- ✅ results saved reproducibly (metadata + JSONL)
- ✅ citations structured and deterministically verifiable
- ✅ retrieval, answer, cost, latency metrics calculated
- ✅ comparison report generated (tables, stats, figures)
- ✅ tests pass; lint and type checks pass
- ✅ README contains no fabricated results (mock runs labeled)
- ✅ one command sequence reproduces the sample experiment
- ✅ Docker build + GitHub Actions verified: CI run #4 on commit
  `18da331` passed all four jobs (lint, typecheck, test, docker build +
  container smoke test). Two CI fixes were needed: trigger on `claude/**`
  pushes, and `load: true` on the docker build step so the smoke test can
  find the image.

### Recommended next steps

1. Real-model experiments: API embeddings + generation + LLM judge.
2. Real paper corpus (open-access PDFs) and benchmark v0.2 with
   chunk-level gold labels.
3. Qdrant adapter and a cross-encoder reranker variant.
