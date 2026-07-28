"""Document ingestion: loaders and the rebuild-from-scratch pipeline."""

from agentic_rag_bench.ingestion.loaders import (
    IngestionError,
    LoadedDocument,
    RawSegment,
    load_document,
    load_markdown,
    load_pdf,
    make_document_id,
)
from agentic_rag_bench.ingestion.pipeline import (
    IngestionReport,
    discover_corpus_files,
    ingest_corpus,
)

__all__ = [
    "IngestionError",
    "IngestionReport",
    "LoadedDocument",
    "RawSegment",
    "discover_corpus_files",
    "ingest_corpus",
    "load_document",
    "load_markdown",
    "load_pdf",
    "make_document_id",
]
