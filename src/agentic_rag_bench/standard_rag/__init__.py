"""Standard (single-pass) RAG baseline pipeline."""

from agentic_rag_bench.standard_rag.pipeline import (
    StandardRAGPipeline,
    build_citations,
    fit_context,
)

__all__ = ["StandardRAGPipeline", "build_citations", "fit_context"]
