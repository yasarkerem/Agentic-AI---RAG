"""Ingest a corpus of PDF/Markdown files into chunks and a vector index.

Rebuilds data/processed from scratch on every run (reproducibility rule).

Usage:
    uv run python scripts/ingest_corpus.py                       # sample corpus
    uv run python scripts/ingest_corpus.py --raw data/raw        # your own papers
"""

import argparse
import sys
from pathlib import Path

from agentic_rag_bench.config import ConfigError, load_system_config
from agentic_rag_bench.ingestion import IngestionError, ingest_corpus
from agentic_rag_bench.observability import configure_logging

DEFAULT_RAW = Path("data/corpus_sample")
DEFAULT_OUT = Path("data/processed")
DEFAULT_CONFIG = Path("configs/default.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="Corpus directory")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Processed output dir")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="System config (chunking/embedding)"
    )
    args = parser.parse_args()

    configure_logging()
    try:
        config = load_system_config(args.config)
        report = ingest_corpus(
            raw_dir=args.raw,
            processed_dir=args.out,
            chunking=config.chunking,
            embedding=config.embedding,
        )
    except (ConfigError, IngestionError) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Ingested {report.documents} documents into {report.chunks} chunks "
        f"-> {report.output_dir}"
        + (f" (skipped: {report.skipped_files})" if report.skipped_files else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
