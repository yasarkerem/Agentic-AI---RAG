# Data layout

| Directory | Contents | Versioned? |
|---|---|---|
| `raw/` | Source documents (PDF, Markdown) as obtained | No — local only |
| `processed/` | Ingested documents, chunks, and index artifacts | No — rebuildable via `make ingest` |
| `benchmark/` | Benchmark task files (JSON/JSONL) + gold evidence | **Yes** |

Rules:

- Only redistributable documents may be added to the corpus (open-access
  papers, permissively licensed technical docs). Do not commit copyrighted
  or private material.
- `processed/` must always be reproducible from `raw/` + configuration;
  never hand-edit it.
- Benchmark files are versioned releases — changes bump
  `benchmark_version`.
