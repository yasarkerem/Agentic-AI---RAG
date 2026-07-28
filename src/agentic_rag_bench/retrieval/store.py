"""Chunk store: chunk_id -> Chunk, persisted as JSONL."""

from collections.abc import Iterable
from pathlib import Path

from agentic_rag_bench.schemas import Chunk, Document


class StoreError(Exception):
    """Raised on missing or corrupt persisted artifacts."""


class ChunkStore:
    """In-memory chunk lookup with JSONL persistence."""

    def __init__(self, chunks: Iterable[Chunk] = ()) -> None:
        self._chunks: dict[str, Chunk] = {}
        for chunk in chunks:
            if chunk.chunk_id in self._chunks:
                raise StoreError(f"Duplicate chunk_id: {chunk.chunk_id}")
            self._chunks[chunk.chunk_id] = chunk

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._chunks

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def all_chunks(self) -> list[Chunk]:
        """Chunks in deterministic (document, position) order."""
        return sorted(self._chunks.values(), key=lambda c: (c.document_id, c.chunk_index))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for chunk in self.all_chunks():
                fh.write(chunk.model_dump_json() + "\n")

    @classmethod
    def load(cls, path: Path) -> "ChunkStore":
        if not path.is_file():
            raise StoreError(f"Chunk store not found: {path} — run `make ingest` first")
        chunks = []
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    chunks.append(Chunk.model_validate_json(line))
                except ValueError as exc:
                    raise StoreError(f"Corrupt chunk at {path}:{line_no}: {exc}") from exc
        return cls(chunks)


def save_documents(documents: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for document in sorted(documents, key=lambda d: d.document_id):
            fh.write(document.model_dump_json() + "\n")


def load_documents(path: Path) -> list[Document]:
    if not path.is_file():
        raise StoreError(f"Document manifest not found: {path} — run `make ingest` first")
    documents = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                documents.append(Document.model_validate_json(line))
            except ValueError as exc:
                raise StoreError(f"Corrupt document at {path}:{line_no}: {exc}") from exc
    return documents
