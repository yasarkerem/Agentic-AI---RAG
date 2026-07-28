"""Deterministic lexical-overlap reranker.

A cross-encoder reranker would require torch — deliberately avoided to keep
the project free and light. This reranker is deterministic and shared by
both systems (fairness); a cross-encoder can implement the same protocol.
"""

from typing import Protocol

from agentic_rag_bench.config import ConfigError, RerankerConfig
from agentic_rag_bench.retrieval.bm25 import tokenize
from agentic_rag_bench.schemas import RetrievalMethod, RetrievedChunk


class Reranker(Protocol):
    """Reorders retrieved chunks by estimated relevance to the query."""

    @property
    def model_name(self) -> str: ...

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


class LexicalOverlapReranker:
    """Scores chunks by weighted query-term coverage (deterministic)."""

    @property
    def model_name(self) -> str:
        return "lexical-overlap"

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")
        query_terms = set(tokenize(query))
        if not query_terms:
            return chunks[:top_n]
        scored: list[tuple[float, RetrievedChunk]] = []
        for retrieved in chunks:
            chunk_terms = set(tokenize(retrieved.chunk.text))
            overlap = len(query_terms & chunk_terms) / len(query_terms)
            scored.append((overlap, retrieved))
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk.chunk_id))
        return [
            RetrievedChunk(
                chunk=r.chunk,
                score=score,
                method=RetrievalMethod.RERANKED,
                rank=rank,
                query=r.query,
            )
            for rank, (score, r) in enumerate(scored[:top_n], start=1)
        ]


def build_reranker(config: RerankerConfig) -> Reranker:
    if config.model in ("mock-reranker", "lexical-overlap"):
        return LexicalOverlapReranker()
    raise ConfigError(
        f"Unknown reranker model {config.model!r} — available: 'lexical-overlap'. "
        "Cross-encoder rerankers plug in behind the Reranker protocol."
    )
