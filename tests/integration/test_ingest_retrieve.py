"""Integration: ingest the sample corpus, then retrieve from the artifacts."""

from pathlib import Path

import pytest

from agentic_rag_bench.config import ChunkingConfig, EmbeddingConfig, RetrievalConfig
from agentic_rag_bench.embeddings import build_embedder
from agentic_rag_bench.ingestion import IngestionError, ingest_corpus
from agentic_rag_bench.retrieval import Retriever

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CORPUS = REPO_ROOT / "data" / "corpus_sample"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def processed_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    ingest_corpus(
        raw_dir=SAMPLE_CORPUS,
        processed_dir=out,
        chunking=ChunkingConfig(chunk_size=120, chunk_overlap=20),
        embedding=EmbeddingConfig(),
    )
    return out


class TestIngestion:
    def test_sample_corpus_ingests_fully(self, processed_dir):
        assert (processed_dir / "chunks.jsonl").is_file()
        assert (processed_dir / "documents.jsonl").is_file()
        assert (processed_dir / "index.npz").is_file()

    def test_malformed_file_skipped_not_fatal(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "good.md").write_text("# Good\n\nSome body text here.\n", encoding="utf-8")
        (corpus / "empty.md").write_text("", encoding="utf-8")
        report = ingest_corpus(
            raw_dir=corpus,
            processed_dir=tmp_path / "out",
            chunking=ChunkingConfig(),
            embedding=EmbeddingConfig(),
        )
        assert report.documents == 1
        assert any("empty.md" in s for s in report.skipped_files)

    def test_empty_corpus_dir_raises(self, tmp_path):
        empty = tmp_path / "nothing"
        empty.mkdir()
        with pytest.raises(IngestionError, match="No ingestible files"):
            ingest_corpus(
                raw_dir=empty,
                processed_dir=tmp_path / "out",
                chunking=ChunkingConfig(),
                embedding=EmbeddingConfig(),
            )

    def test_duplicate_content_detected(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        body = "# Title\n\nIdentical body for duplicate detection.\n"
        (corpus / "one.md").write_text(body, encoding="utf-8")
        (corpus / "two.md").write_text(body, encoding="utf-8")
        report = ingest_corpus(
            raw_dir=corpus,
            processed_dir=tmp_path / "out",
            chunking=ChunkingConfig(),
            embedding=EmbeddingConfig(),
        )
        assert report.documents == 1
        assert report.duplicate_documents == ["two"]


class TestRetrievalOverSample:
    def _retriever(self, processed_dir, **kwargs) -> Retriever:
        config = RetrievalConfig(**kwargs)
        return Retriever.from_processed_dir(
            processed_dir, build_embedder(EmbeddingConfig()), config
        )

    def test_dense_retrieval_finds_raptor(self, processed_dir):
        result = self._retriever(processed_dir, top_k=5).retrieve(
            "What is the main contribution of RAPTOR?"
        )
        assert not result.is_empty
        assert any(r.chunk.document_id == "raptor_overview" for r in result.chunks)

    def test_hybrid_retrieval_finds_raptor_top(self, processed_dir):
        result = self._retriever(processed_dir, top_k=5, use_bm25=True, dense_weight=0.5).retrieve(
            "RAPTOR hierarchical tree summarization"
        )
        assert result.chunks[0].chunk.document_id == "raptor_overview"
        assert result.method.value == "hybrid"

    def test_empty_query_returns_empty_result(self, processed_dir):
        result = self._retriever(processed_dir, top_k=5).retrieve("   ")
        assert result.is_empty

    def test_ranks_are_sequential(self, processed_dir):
        result = self._retriever(processed_dir, top_k=4).retrieve("reranking effectiveness")
        assert [r.rank for r in result.chunks] == list(range(1, len(result.chunks) + 1))
