# Contributing

Thanks for your interest in Agentic-RAG-Bench.

## Setup

```bash
make setup      # uv sync + pre-commit hooks
make check      # lint + typecheck + tests — must pass before any PR
```

Python 3.12 and [uv](https://docs.astral.sh/uv/) are the only prerequisites.
The test suite runs entirely on mock providers: **no API keys, no network, no
cost.** If a change makes a test require either, the change is wrong.

## Ground rules

- **Never commit secrets.** `.env` is gitignored; keys live only there.
- **Never invent experimental results.** Mock-run numbers are harness
  validation, and the report generator labels them as such — keep it that way.
- **Determinism is a feature.** Chunking, retrieval, mock generation,
  evaluation, and statistics (seeded bootstrap) are reproducible; new
  components should be too, or must document why not.
- **Fairness controls are tested.** Anything that lets the two systems see
  different corpora, embedders, or generation models breaks the benchmark —
  `tests/unit/test_config.py` guards this.
- Add tests with each feature (failure modes included), run `make check`,
  and update `WORKLOG.md` with what changed and why.

## Where things live

| Area | Path |
|---|---|
| Config system + typed settings | `src/agentic_rag_bench/config.py` |
| Schemas (single source of truth) | `src/agentic_rag_bench/schemas/` |
| Ingestion / chunking / retrieval | `ingestion/`, `chunking/`, `embeddings/`, `retrieval/`, `reranking/` |
| Pipelines | `standard_rag/`, `agentic_rag/` |
| Benchmark + metrics + judges + stats | `evaluation/`, `data/benchmark/` |
| Runner + report | `experiments.py`, `reporting.py`, `scripts/` |

## Adding corpus documents

Only redistributable material. Drop PDF/Markdown into `data/raw/` (gitignored,
your local corpus) or extend `data/corpus_sample/` (committed — original or
permissively licensed text only), then `make ingest`.

## Benchmark changes

Task definitions live in `scripts/build_benchmark.py`. Edit there, bump
`BENCHMARK_VERSION`, run `make benchmark`, and commit both the script and the
regenerated JSON — a test fails if they drift apart.
