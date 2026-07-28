"""Retrieval results: scored chunks and how they were obtained."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag_bench.schemas.documents import Chunk


class RetrievalMethod(StrEnum):
    """How a chunk was retrieved. Hybrid results record the fused method."""

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    RERANKED = "reranked"


class RetrievedChunk(BaseModel):
    """A chunk plus the score and method that surfaced it."""

    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    score: float = Field(description="Similarity / fusion / reranker score, method-dependent")
    method: RetrievalMethod
    rank: int = Field(ge=1, description="1-based rank within its result list")
    query: str = Field(min_length=1, description="The query string that produced this result")


class RetrievalResult(BaseModel):
    """The full result of one retrieval call for one query."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    method: RetrievalMethod
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    latency_seconds: float = Field(ge=0)

    @property
    def is_empty(self) -> bool:
        """True when retrieval found nothing — a failure mode callers must handle."""
        return len(self.chunks) == 0
