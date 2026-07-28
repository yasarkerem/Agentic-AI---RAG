"""Documents and chunks: the units that ingestion produces and retrieval returns."""

import hashlib
import re

from pydantic import BaseModel, ConfigDict, Field

_DOCUMENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")


def make_chunk_id(document_id: str, page: int | None, chunk_index: int) -> str:
    """Build the deterministic chunk ID used across the whole system.

    Format: ``{document_id}_p{page}_c{chunk_index}`` (e.g. ``paper_001_p4_c2``).
    Documents without page numbers (e.g. Markdown) use ``p0``.
    """
    if chunk_index < 0:
        raise ValueError(f"chunk_index must be >= 0, got {chunk_index}")
    if page is not None and page < 0:
        raise ValueError(f"page must be >= 0, got {page}")
    return f"{document_id}_p{page if page is not None else 0}_c{chunk_index}"


def content_hash(text: str) -> str:
    """Stable content hash used for duplicate detection during ingestion."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DocumentMetadata(BaseModel):
    """Bibliographic and provenance metadata carried by every document."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    source_path: str = Field(min_length=1, description="Path or URL the document was loaded from")
    publication_year: int | None = Field(default=None, ge=1900, le=2100)
    authors: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class Document(BaseModel):
    """A source document in the corpus (a paper or technical document)."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(pattern=_DOCUMENT_ID_PATTERN.pattern)
    metadata: DocumentMetadata
    num_pages: int | None = Field(default=None, ge=1)
    content_sha256: str | None = Field(
        default=None, description="Hash of the raw document content, for duplicate detection"
    )


class Chunk(BaseModel):
    """A retrievable unit of text with full provenance back to its document."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=0)
    section: str | None = Field(default=None, description="Section heading, when available")
    source_path: str = Field(min_length=1)
    publication_year: int | None = Field(default=None, ge=1900, le=2100)
    chunk_index: int = Field(ge=0, description="Position of this chunk within its document")
    text_sha256: str | None = Field(default=None, description="Hash for duplicate detection")
