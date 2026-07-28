# Architecture

All components below are implemented and tested. The default stack is
deliberately free and deterministic (mock providers); real providers plug in
behind the same interfaces.

## Main components

| Component | Package | Notes |
|---|---|---|
| Configuration (env + layered YAML) | `agentic_rag_bench.config` | `AppSettings` from env/`.env`; `SystemConfig` from `configs/default.yaml` + overlay |
| Core schemas | `agentic_rag_bench.schemas` | Pydantic v2, `extra="forbid"` everywhere |
| Structured logging | `agentic_rag_bench.observability` | structlog + unconditional secret redaction |
| Ingestion (PDF/Markdown) | `agentic_rag_bench.ingestion` | pypdf page-aware; Markdown frontmatter + sections; rebuild-from-scratch |
| Chunking | `agentic_rag_bench.chunking` | word windows, never cross page/section, SHA-256 dedupe |
| Embeddings | `agentic_rag_bench.embeddings` | `Embedder` protocol; default deterministic feature-hashing (free, offline) |
| Retrieval | `agentic_rag_bench.retrieval` | in-memory cosine index (numpy, persisted), in-repo Okapi BM25, min-max weighted fusion |
| Reranking | `agentic_rag_bench.reranking` | `Reranker` protocol; default lexical-overlap (torch-free by design) |
| Generation | `agentic_rag_bench.generation` | `LLMProvider` protocol; deterministic mock (default) + opt-in Anthropic SDK provider; pricing table |
| Standard RAG pipeline | `agentic_rag_bench.standard_rag` | fixed single-pass, no iteration |
| Agentic RAG state machine | `agentic_rag_bench.agentic_rag` | bounded loop, full trace, explicit stop reasons |
| Evaluation | `agentic_rag_bench.evaluation` | retrieval metrics, citation checks, judges, stats |
| Experiment runner | `agentic_rag_bench.experiments` | matrix runs, persistence, re-evaluation |
| Reporting | `agentic_rag_bench.reporting` | tables, Wilcoxon comparisons, matplotlib figures |

## Data flow

```
corpus (PDF/MD) -> ingestion -> chunks (deterministic IDs, provenance)
                -> hashed embeddings -> in-memory cosine index (+ BM25 at load)
task -> pipeline (A or B) -> PipelineAnswer {answer, citations, usage, trace}
     -> Evaluator -> EvaluationScores -> results.jsonl -> report.md + figures
```

Key invariant: **every chunk carries provenance** (document ID, title, page,
section, source path, year) and a deterministic ID
(`{document_id}_p{page}_c{index}`), so citations are verifiable end to end.
Citations are formatted by pipeline code from retrieved chunks — the model
never invents citation labels.

## Standard RAG pipeline

Fixed sequence: query normalization → dense retrieval → optional BM25 +
min-max weighted fusion → optional rerank → context budget fitting → single
generation → citation formatting. **No iteration.**

## Agentic state machine

Nodes (see `schemas.AgentNode`): query analyzer → planner →
[retriever → evidence assessor → query rewriter]\* → contradiction checker →
synthesizer → citation verifier → finalizer.

Termination is guaranteed by hard bounds in `AgentConfig`
(`max_retrieval_rounds`, `max_subquestions`, `max_total_tokens`,
`max_cost_usd`, `max_retries`, `timeout_seconds`) plus a deterministic
no-progress stop (if no weak query could be rewritten, another round would
repeat identical searches). Every run records an `AgentTrace` with per-node
steps and exactly one `StopReason`.

## Provider abstraction and the mock stack

All model-shaped components sit behind protocols (`Embedder`, `Reranker`,
`LLMProvider`). Defaults are deterministic and free:

- `HashedEmbedder` — feature-hashing bag-of-words with stopword filtering.
- `LexicalOverlapReranker` — query-term coverage.
- `MockLLMProvider` — purpose-keyed template responses driven by a
  machine-readable payload embedded in every prompt (real models read the
  human-readable part; the mock parses the payload).

The `AnthropicProvider` (official SDK) activates only when a config selects
`provider: anthropic` **and** `ANTHROPIC_API_KEY` is set — tests never touch it.

## Storage choices

- **Vector index:** in-memory exact cosine (numpy), persisted as `.npz` —
  one implementation path that runs everywhere. A Qdrant adapter is future
  work; the compose service is already provisioned.
- **Chunks/documents:** JSONL under `data/processed/` (rebuildable).
- **Benchmark:** versioned JSON under `data/benchmark/` (committed).
- **Experiment results:** `metadata.json` + `results.jsonl` per run under
  `reports/results/`.

## Failure handling

- Configuration problems fail loudly at load time (`ConfigError`).
- Pipeline failures are recorded (`PipelineAnswer.failed` + reason), never
  silently dropped; the runner isolates per-task crashes and continues.
- Empty retrieval is a modeled state (`RetrievalResult.is_empty`).
- LLM calls retry up to `max_retries`, then fail the run with
  `StopReason.ERROR`.
- Secrets are redacted at the logging layer, unconditionally.
