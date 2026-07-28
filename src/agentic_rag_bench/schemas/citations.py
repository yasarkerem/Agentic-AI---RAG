"""Structured citations grounded in retrieved chunks (brief §10)."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """A citation attached to a final answer.

    Citations are built from retrieved chunks, never invented by the model:
    ``chunk_id`` must reference a chunk that was actually retrieved, and
    ``quoted_evidence`` must appear in that chunk's text.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=0)
    chunk_id: str = Field(min_length=1)
    quoted_evidence: str = Field(min_length=1)
    retrieval_score: float


class CitationVerdict(StrEnum):
    """Outcome of verifying one citation against the retrieved evidence."""

    SUPPORTED = "supported"
    QUOTE_NOT_IN_CHUNK = "quote_not_in_chunk"
    CHUNK_NOT_FOUND = "chunk_not_found"
    CLAIM_NOT_SUPPORTED = "claim_not_supported"


class CitationVerification(BaseModel):
    """Result of the deterministic + judged checks for one citation."""

    model_config = ConfigDict(extra="forbid")

    citation: Citation
    verdict: CitationVerdict
    detail: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.verdict is CitationVerdict.SUPPORTED
