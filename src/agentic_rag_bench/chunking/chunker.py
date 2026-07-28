"""Word-window chunking that never crosses page or section boundaries.

Sizes are counted in whitespace-delimited words — a documented approximation
of tokens that keeps chunking deterministic and dependency-free.
"""

from agentic_rag_bench.config import ChunkingConfig
from agentic_rag_bench.ingestion.loaders import LoadedDocument
from agentic_rag_bench.schemas import Chunk, content_hash, make_chunk_id


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = size - overlap
    result = []
    start = 0
    while start < len(words):
        window = words[start : start + size]
        result.append(window)
        if start + size >= len(words):
            break
        start += step
    return result


def chunk_document(loaded: LoadedDocument, config: ChunkingConfig) -> list[Chunk]:
    """Chunk every segment of a document into overlapping word windows.

    Deterministic: chunk IDs are ``{doc}_p{page}_c{index-within-page}`` and
    identical text within a document is emitted only once (duplicate
    detection via SHA-256).
    """
    doc = loaded.document
    chunks: list[Chunk] = []
    seen_hashes: set[str] = set()
    per_page_counter: dict[int | None, int] = {}
    global_index = 0

    for segment in loaded.segments:
        words = segment.text.split()
        if not words:
            continue
        for window in _windows(words, config.chunk_size, config.chunk_overlap):
            text = " ".join(window)
            digest = content_hash(text)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            within_page = per_page_counter.get(segment.page, 0)
            per_page_counter[segment.page] = within_page + 1
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.document_id, segment.page, within_page),
                    document_id=doc.document_id,
                    document_title=doc.metadata.title,
                    text=text,
                    page=segment.page,
                    section=segment.section,
                    source_path=doc.metadata.source_path,
                    publication_year=doc.metadata.publication_year,
                    chunk_index=global_index,
                    text_sha256=digest,
                )
            )
            global_index += 1
    return chunks
