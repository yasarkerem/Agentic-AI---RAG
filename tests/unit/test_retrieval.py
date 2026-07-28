"""BM25, fusion, vector index, embedder, and reranker behavior."""

import math

import pytest

from agentic_rag_bench.config import ConfigError, EmbeddingConfig, RerankerConfig
from agentic_rag_bench.embeddings import HashedEmbedder, build_embedder
from agentic_rag_bench.reranking import LexicalOverlapReranker, build_reranker
from agentic_rag_bench.retrieval import BM25Index, InMemoryVectorIndex, fuse
from agentic_rag_bench.retrieval.index import VectorIndexError
from agentic_rag_bench.schemas import Chunk, RetrievalMethod, RetrievedChunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="d",
        document_title="D",
        text=text,
        source_path="d.md",
        chunk_index=0,
    )


def _retrieved(chunk_id: str, text: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=_chunk(chunk_id, text),
        score=1.0 / rank,
        method=RetrievalMethod.DENSE,
        rank=rank,
        query="q",
    )


class TestBM25:
    def _index(self) -> BM25Index:
        index = BM25Index()
        index.fit(
            ["c1", "c2", "c3"],
            [
                "raptor builds a hierarchical summary tree",
                "graphrag extracts a knowledge graph of entities",
                "bm25 is a sparse lexical retrieval method",
            ],
        )
        return index

    def test_exact_term_ranks_its_document_first(self):
        results = self._index().search("raptor tree", top_k=3)
        assert results[0][0] == "c1"

    def test_zero_score_documents_excluded(self):
        results = self._index().search("raptor", top_k=3)
        assert [doc_id for doc_id, _ in results] == ["c1"]

    def test_unknown_terms_return_empty(self):
        assert self._index().search("zzz qqq", top_k=3) == []

    def test_empty_index_returns_empty(self):
        assert BM25Index().search("anything", top_k=5) == []

    def test_idf_floor_never_negative(self):
        index = BM25Index()
        index.fit(["a", "b"], ["common term here", "common term there"])
        assert index._idf("common") >= 0.0

    def test_scores_match_hand_computation(self):
        """One term, k1=1.5, b=0.75, verified against the Okapi formula by hand."""
        index = BM25Index()
        index.fit(["a", "b"], ["x x y", "y z"])  # doc a: x twice of 3 tokens
        # idf(x) = log((2 - 1 + 0.5)/(1 + 0.5) + 1) = log(2)
        # tf part = 2*(1.5+1) / (2 + 1.5*(1 - 0.75 + 0.75*3/2.5)) = 5/(2+1.725)
        expected = math.log(2.0) * 5.0 / 3.725
        results = index.search("x", top_k=2)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(expected, rel=1e-9)


class TestFusion:
    def test_weighted_sum_orders_results(self):
        dense = [("a", 0.9), ("b", 0.5)]
        sparse = [("b", 12.0), ("c", 6.0)]
        fused = fuse(dense, sparse, dense_weight=0.5, top_k=3)
        # normalized: dense a=1, b=0; sparse b=1, c=0 -> a=0.5, b=0.5, c=0.0
        scores = dict(fused)
        assert scores["a"] == pytest.approx(0.5)
        assert scores["b"] == pytest.approx(0.5)
        assert scores["c"] == pytest.approx(0.0)

    def test_pure_dense_weight_ignores_sparse(self):
        fused = fuse([("a", 1.0)], [("b", 5.0)], dense_weight=1.0, top_k=2)
        assert fused[0][0] == "a"
        assert dict(fused)["b"] == 0.0

    def test_empty_lists_ok(self):
        assert fuse([], [], dense_weight=0.5, top_k=5) == []

    def test_constant_scores_normalize_to_one(self):
        fused = fuse([("a", 3.0), ("b", 3.0)], [], dense_weight=1.0, top_k=2)
        assert dict(fused) == {"a": 1.0, "b": 1.0}

    def test_deterministic_tiebreak(self):
        fused = fuse([("b", 1.0), ("a", 1.0)], [], dense_weight=1.0, top_k=2)
        assert [doc_id for doc_id, _ in fused] == ["a", "b"]

    def test_invalid_weight_rejected(self):
        with pytest.raises(ValueError, match="dense_weight"):
            fuse([], [], dense_weight=1.5, top_k=5)


class TestVectorIndex:
    def test_cosine_top_k(self):
        index = InMemoryVectorIndex()
        index.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
        results = index.search([1.0, 0.1], top_k=2)
        assert results[0][0] == "a"
        assert results[0][1] > results[1][1]

    def test_duplicate_ids_rejected(self):
        index = InMemoryVectorIndex()
        index.add(["a"], [[1.0, 0.0]])
        with pytest.raises(VectorIndexError, match="Duplicate"):
            index.add(["a"], [[0.0, 1.0]])

    def test_mismatched_inputs_rejected(self):
        with pytest.raises(VectorIndexError, match="out of sync"):
            InMemoryVectorIndex().add(["a", "b"], [[1.0, 0.0]])

    def test_empty_index_returns_empty(self):
        assert InMemoryVectorIndex().search([1.0, 0.0], top_k=3) == []

    def test_save_load_round_trip(self, tmp_path):
        index = InMemoryVectorIndex()
        index.add(["a", "b"], [[1.0, 0.0], [0.6, 0.8]])
        path = tmp_path / "index.npz"
        index.save(path)
        restored = InMemoryVectorIndex.load(path)
        assert len(restored) == 2
        assert restored.search([1.0, 0.0], top_k=1)[0][0] == "a"

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(VectorIndexError, match="not found"):
            InMemoryVectorIndex.load(tmp_path / "nope.npz")


class TestHashedEmbedder:
    def test_deterministic(self):
        embedder = HashedEmbedder(dimension=64)
        assert embedder.embed_texts(["same text"]) == embedder.embed_texts(["same text"])

    def test_dimension_and_normalization(self):
        embedder = HashedEmbedder(dimension=64)
        vector = embedder.embed_texts(["retrieval augmented generation"])[0]
        assert len(vector) == 64
        assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0)

    def test_shared_vocabulary_is_more_similar(self):
        embedder = HashedEmbedder(dimension=256)
        a, b, c = embedder.embed_texts(
            [
                "raptor hierarchical tree retrieval",
                "raptor tree retrieval strategy",
                "completely unrelated cooking recipe",
            ]
        )
        sim_ab = sum(x * y for x, y in zip(a, b, strict=True))
        sim_ac = sum(x * y for x, y in zip(a, c, strict=True))
        assert sim_ab > sim_ac

    def test_registry_rejects_unknown_provider(self):
        with pytest.raises(ConfigError, match="Unknown embedding provider"):
            build_embedder(EmbeddingConfig(provider="real-api", model="x"))


class TestReranker:
    def test_reorders_by_query_overlap(self):
        reranker = LexicalOverlapReranker()
        chunks = [
            _retrieved("c1", "totally unrelated content", rank=1),
            _retrieved("c2", "raptor tree retrieval explained", rank=2),
        ]
        result = reranker.rerank("raptor tree retrieval", chunks, top_n=2)
        assert result[0].chunk.chunk_id == "c2"
        assert result[0].method is RetrievalMethod.RERANKED
        assert [r.rank for r in result] == [1, 2]

    def test_top_n_truncates(self):
        reranker = LexicalOverlapReranker()
        chunks = [_retrieved(f"c{i}", f"text {i}", rank=i + 1) for i in range(5)]
        assert len(reranker.rerank("text", chunks, top_n=2)) == 2

    def test_registry_rejects_unknown_model(self):
        with pytest.raises(ConfigError, match="Unknown reranker"):
            build_reranker(RerankerConfig(enabled=True, model="cross-encoder/foo", top_n=5))
