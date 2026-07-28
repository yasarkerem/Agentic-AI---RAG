# Architecture

> **Status:** skeleton (Phase 1). Sections are filled in as their components
> are implemented; unimplemented components are marked with their phase.

## Main components

| Component | Package | Phase |
|---|---|---|
| Configuration (env + layered YAML) | `agentic_rag_bench.config` | 1 ✅ |
| Core schemas | `agentic_rag_bench.schemas` | 1 ✅ |
| Structured logging & tracing | `agentic_rag_bench.observability` | 1 ✅ (logging) |
| Ingestion (PDF/Markdown) | `agentic_rag_bench.ingestion` | 2 |
| Chunking | `agentic_rag_bench.chunking` | 2 |
| Embeddings | `agentic_rag_bench.embeddings` | 2 |
| Retrieval (dense/BM25/hybrid) | `agentic_rag_bench.retrieval` | 2 |
| Reranking | `agentic_rag_bench.reranking` | 2 |
| Generation (provider abstraction) | `agentic_rag_bench.generation` | 3 |
| Standard RAG pipeline | `agentic_rag_bench.standard_rag` | 3 |
| Agentic RAG state machine | `agentic_rag_bench.agentic_rag` | 4 |
| Evaluation | `agentic_rag_bench.evaluation` | 5 |

## Data flow

```
corpus (PDF/MD) -> ingestion -> chunks (deterministic IDs, provenance)
                -> embedding -> vector index (Qdrant) + BM25 index
task -> pipeline (A or B) -> PipelineAnswer {answer, citations, usage, trace}
     -> evaluation -> EvaluationScores -> report
```

Key invariant: **every chunk carries provenance** (document ID, title, page,
section, source path, year) and a deterministic ID
(`{document_id}_p{page}_c{index}`), so citations are verifiable end to end.

## Standard RAG pipeline (Phase 3)

Fixed sequence: query normalization → dense retrieval → optional BM25 +
fusion → optional rerank → context construction → single generation →
citation formatting. **No iteration.**

## Agentic state machine (Phase 4)

Nodes (see `schemas.AgentNode`): query analyzer → planner → retriever →
evidence assessor → query rewriter (loop) → contradiction checker →
synthesizer → citation verifier → finalizer.

Termination is guaranteed by hard bounds in `AgentConfig`
(`max_retrieval_rounds`, `max_subquestions`, `max_total_tokens`,
`max_cost_usd`, `max_retries`, `timeout_seconds`) plus deterministic stop
rules recorded as `schemas.StopReason`.

## Storage choices

- **Vector store:** Qdrant (docker-compose service) — one implementation
  path first; pgvector is a possible later alternative.
- **Experiment results:** JSONL rows of `schemas.TaskResult` under
  `reports/results/` (Parquet optional later).
- **Benchmark:** versioned JSON/JSONL under `data/benchmark/`.

## Failure handling

- Configuration problems fail loudly at load time (`ConfigError`).
- Pipeline failures are recorded (`PipelineAnswer.failed` +
  `failure_reason`), never silently dropped; the experiment runner counts
  them into the failure rate.
- Empty retrieval is a modeled state (`RetrievalResult.is_empty`), not an
  exception.
- Secrets are redacted at the logging layer, unconditionally.
