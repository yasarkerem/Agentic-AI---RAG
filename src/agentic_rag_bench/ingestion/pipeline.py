"""Corpus ingestion: load -> chunk -> embed -> persist (rebuildable from scratch)."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agentic_rag_bench.chunking.chunker import chunk_document
from agentic_rag_bench.config import ChunkingConfig, EmbeddingConfig
from agentic_rag_bench.embeddings.base import build_embedder
from agentic_rag_bench.ingestion.loaders import LOADERS, IngestionError, load_document
from agentic_rag_bench.observability import get_logger
from agentic_rag_bench.retrieval.index import InMemoryVectorIndex
from agentic_rag_bench.retrieval.retriever import CHUNKS_FILE, DOCUMENTS_FILE, INDEX_FILE
from agentic_rag_bench.retrieval.store import ChunkStore, save_documents
from agentic_rag_bench.schemas import Chunk, Document

logger = get_logger("ingestion")


class IngestionReport(BaseModel):
    """What one ingestion run produced."""

    model_config = ConfigDict(extra="forbid")

    documents: int
    chunks: int
    skipped_files: list[str]
    duplicate_documents: list[str]
    output_dir: str


def discover_corpus_files(raw_dir: Path) -> list[Path]:
    """All supported files under ``raw_dir``, deterministic order."""
    if not raw_dir.is_dir():
        raise IngestionError(f"Corpus directory not found: {raw_dir}")
    return sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in LOADERS and p.is_file())


def ingest_corpus(
    raw_dir: Path,
    processed_dir: Path,
    chunking: ChunkingConfig,
    embedding: EmbeddingConfig,
) -> IngestionReport:
    """Rebuild the full processed corpus (documents, chunks, vector index).

    Always rebuilds from scratch — `processed/` must be reproducible, never
    incrementally patched. Malformed files are skipped and reported, not
    silently ignored.
    """
    files = discover_corpus_files(raw_dir)
    if not files:
        raise IngestionError(f"No ingestible files (md/pdf) found under {raw_dir}")

    documents: list[Document] = []
    all_chunks: list[Chunk] = []
    skipped: list[str] = []
    duplicates: list[str] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()

    for path in files:
        try:
            loaded = load_document(path)
        except IngestionError as exc:
            logger.warning("document_skipped", path=str(path), error=str(exc))
            skipped.append(str(path))
            continue

        doc = loaded.document
        if doc.document_id in seen_ids:
            logger.warning("duplicate_document_id", document_id=doc.document_id, path=str(path))
            duplicates.append(doc.document_id)
            continue

        chunks = chunk_document(loaded, chunking)
        if not chunks:
            skipped.append(str(path))
            continue

        corpus_digest = "".join(c.text_sha256 or "" for c in chunks)
        if corpus_digest in seen_content:
            logger.warning("duplicate_document_content", document_id=doc.document_id)
            duplicates.append(doc.document_id)
            continue
        seen_content.add(corpus_digest)
        seen_ids.add(doc.document_id)
        documents.append(doc)
        all_chunks.extend(chunks)
        logger.info("document_ingested", document_id=doc.document_id, chunks=len(chunks))

    if not documents:
        raise IngestionError(f"Every file under {raw_dir} failed to ingest: {skipped}")

    embedder = build_embedder(embedding)
    index = InMemoryVectorIndex()
    vectors = embedder.embed_texts([c.text for c in all_chunks])
    index.add([c.chunk_id for c in all_chunks], vectors)

    processed_dir.mkdir(parents=True, exist_ok=True)
    ChunkStore(all_chunks).save(processed_dir / CHUNKS_FILE)
    save_documents(documents, processed_dir / DOCUMENTS_FILE)
    index.save(processed_dir / INDEX_FILE)

    report = IngestionReport(
        documents=len(documents),
        chunks=len(all_chunks),
        skipped_files=skipped,
        duplicate_documents=duplicates,
        output_dir=str(processed_dir),
    )
    logger.info("ingestion_complete", **report.model_dump())
    return report
