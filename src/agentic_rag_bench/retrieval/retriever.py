"""Retriever facade: dense, BM25, and hybrid retrieval over ingested artifacts."""

import time
from pathlib import Path

from agentic_rag_bench.config import RetrievalConfig
from agentic_rag_bench.embeddings.base import Embedder
from agentic_rag_bench.retrieval.bm25 import BM25Index
from agentic_rag_bench.retrieval.fusion import fuse
from agentic_rag_bench.retrieval.index import InMemoryVectorIndex
from agentic_rag_bench.retrieval.store import ChunkStore
from agentic_rag_bench.schemas import RetrievalMethod, RetrievalResult, RetrievedChunk

CHUNKS_FILE = "chunks.jsonl"
DOCUMENTS_FILE = "documents.jsonl"
INDEX_FILE = "index.npz"


def normalize_query(query: str) -> str:
    """Whitespace normalization — the only query mutation the baseline does."""
    return " ".join(query.split())


class Retriever:
    """Executes one retrieval pass; both pipelines share this component."""

    def __init__(
        self,
        store: ChunkStore,
        index: InMemoryVectorIndex,
        embedder: Embedder,
        config: RetrievalConfig,
    ) -> None:
        self._store = store
        self._index = index
        self._embedder = embedder
        self._config = config
        self._bm25 = BM25Index()
        if config.use_bm25:
            chunks = store.all_chunks()
            self._bm25.fit([c.chunk_id for c in chunks], [c.text for c in chunks])

    @classmethod
    def from_processed_dir(
        cls, processed_dir: Path, embedder: Embedder, config: RetrievalConfig
    ) -> "Retriever":
        store = ChunkStore.load(processed_dir / CHUNKS_FILE)
        index = InMemoryVectorIndex.load(processed_dir / INDEX_FILE)
        return cls(store=store, index=index, embedder=embedder, config=config)

    @property
    def store(self) -> ChunkStore:
        return self._store

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """One retrieval pass. Empty results are a modeled state, not an error."""
        started = time.perf_counter()
        normalized = normalize_query(query)
        k = top_k if top_k is not None else self._config.top_k
        if not normalized:
            return RetrievalResult(
                query=query or " ",
                method=RetrievalMethod.DENSE,
                chunks=[],
                latency_seconds=time.perf_counter() - started,
            )

        vector = self._embedder.embed_texts([normalized])[0]
        dense = self._index.search(vector, top_k=k)

        if self._config.use_bm25:
            sparse = self._bm25.search(normalized, top_k=k)
            ranked = fuse(dense, sparse, dense_weight=self._config.dense_weight, top_k=k)
            method = RetrievalMethod.HYBRID
        else:
            ranked = dense
            method = RetrievalMethod.DENSE

        retrieved = []
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            chunk = self._store.get(chunk_id)
            if chunk is None:
                continue  # index/store drift; skip rather than crash mid-experiment
            retrieved.append(
                RetrievedChunk(chunk=chunk, score=score, method=method, rank=rank, query=normalized)
            )
        return RetrievalResult(
            query=normalized,
            method=method,
            chunks=retrieved,
            latency_seconds=time.perf_counter() - started,
        )
