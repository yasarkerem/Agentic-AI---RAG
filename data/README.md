# Data layout

| Directory | Contents | Versioned? |
|---|---|---|
| `corpus_sample/` | Sample dev corpus: original Markdown documents about RAG topics, written for this project (redistributable) | **Yes** |
| `raw/` | Your own source documents (PDF, Markdown) | No — local only |
| `processed/` | Ingested documents, chunks, and index artifacts | No — rebuildable via `make ingest` |
| `benchmark/` | Benchmark task files (JSON/JSONL) + gold evidence | **Yes** |

The sample corpus is *not* the real research corpus: it exists so the whole
pipeline (and the benchmark harness) runs end to end with zero downloads
and zero API keys. Real experiments should ingest actual open-access papers
into `raw/`.

Rules:

- Only redistributable documents may be added to the corpus (open-access
  papers, permissively licensed technical docs). Do not commit copyrighted
  or private material.
- `processed/` must always be reproducible from `raw/` + configuration;
  never hand-edit it.
- Benchmark files are versioned releases — changes bump
  `benchmark_version`.
