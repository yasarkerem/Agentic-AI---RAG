"""Deterministic citation verification."""

from agentic_rag_bench.evaluation import verify_citation
from agentic_rag_bench.retrieval import ChunkStore
from agentic_rag_bench.schemas import Chunk, Citation, CitationVerdict


def _store() -> ChunkStore:
    return ChunkStore(
        [
            Chunk(
                chunk_id="d_p0_c0",
                document_id="d",
                document_title="Doc",
                text="RAPTOR builds a hierarchical tree of summaries over chunks.",
                source_path="d.md",
                chunk_index=0,
            )
        ]
    )


def _citation(chunk_id: str = "d_p0_c0", quote: str = "RAPTOR builds a hierarchical tree"):
    return Citation(
        document_id="d",
        title="Doc",
        page=None,
        chunk_id=chunk_id,
        quoted_evidence=quote,
        retrieval_score=0.9,
    )


class TestVerifyCitation:
    def test_valid_citation_supported(self):
        result = verify_citation(_citation(), _store())
        assert result.verdict is CitationVerdict.SUPPORTED
        assert result.is_valid

    def test_missing_chunk_detected(self):
        result = verify_citation(_citation(chunk_id="nope_p0_c9"), _store())
        assert result.verdict is CitationVerdict.CHUNK_NOT_FOUND

    def test_fabricated_quote_detected(self):
        result = verify_citation(_citation(quote="RAPTOR cures hallucination entirely"), _store())
        assert result.verdict is CitationVerdict.QUOTE_NOT_IN_CHUNK

    def test_whitespace_and_case_normalized(self):
        result = verify_citation(_citation(quote="raptor  builds a\nhierarchical tree"), _store())
        assert result.verdict is CitationVerdict.SUPPORTED
