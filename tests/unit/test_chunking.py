"""Chunking: windows, overlap, boundaries, dedupe, deterministic IDs."""

from agentic_rag_bench.chunking import chunk_document
from agentic_rag_bench.config import ChunkingConfig
from agentic_rag_bench.ingestion import LoadedDocument, RawSegment
from agentic_rag_bench.schemas import Document, DocumentMetadata


def _doc(segments: list[RawSegment]) -> LoadedDocument:
    return LoadedDocument(
        document=Document(
            document_id="doc_a",
            metadata=DocumentMetadata(title="Doc A", source_path="data/raw/a.md"),
        ),
        segments=segments,
    )


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


class TestWindows:
    def test_short_segment_is_single_chunk(self):
        loaded = _doc([RawSegment(text=words(10), page=None, section="Intro")])
        chunks = chunk_document(loaded, ChunkingConfig(chunk_size=50, chunk_overlap=10))
        assert len(chunks) == 1
        assert chunks[0].section == "Intro"

    def test_long_segment_splits_with_overlap(self):
        loaded = _doc([RawSegment(text=words(100), page=None, section=None)])
        chunks = chunk_document(loaded, ChunkingConfig(chunk_size=40, chunk_overlap=10))
        assert len(chunks) == 3
        first_words = chunks[0].text.split()
        second_words = chunks[1].text.split()
        # overlap: last 10 words of chunk 0 == first 10 words of chunk 1
        assert first_words[-10:] == second_words[:10]

    def test_all_words_covered(self):
        loaded = _doc([RawSegment(text=words(100), page=None, section=None)])
        chunks = chunk_document(loaded, ChunkingConfig(chunk_size=40, chunk_overlap=10))
        covered = set()
        for chunk in chunks:
            covered.update(chunk.text.split())
        assert covered == set(words(100).split())

    def test_chunks_never_cross_segments(self):
        loaded = _doc(
            [
                RawSegment(text=words(5, "a"), page=1, section=None),
                RawSegment(text=words(5, "b"), page=2, section=None),
            ]
        )
        chunks = chunk_document(loaded, ChunkingConfig(chunk_size=50, chunk_overlap=10))
        assert len(chunks) == 2
        assert chunks[0].page == 1 and "b0" not in chunks[0].text
        assert chunks[1].page == 2 and "a0" not in chunks[1].text


class TestIdsAndDedupe:
    def test_chunk_ids_deterministic_and_page_scoped(self):
        loaded = _doc(
            [
                RawSegment(text=words(100, "a"), page=1, section=None),
                RawSegment(text=words(5, "b"), page=2, section=None),
            ]
        )
        chunks = chunk_document(loaded, ChunkingConfig(chunk_size=60, chunk_overlap=10))
        assert [c.chunk_id for c in chunks] == ["doc_a_p1_c0", "doc_a_p1_c1", "doc_a_p2_c0"]
        assert [c.chunk_index for c in chunks] == [0, 1, 2]

    def test_duplicate_text_within_document_dropped(self):
        segment = RawSegment(text=words(10), page=None, section=None)
        loaded = _doc([segment, segment])
        chunks = chunk_document(loaded, ChunkingConfig(chunk_size=50, chunk_overlap=5))
        assert len(chunks) == 1

    def test_rechunking_is_deterministic(self):
        loaded = _doc([RawSegment(text=words(80), page=3, section="S")])
        config = ChunkingConfig(chunk_size=30, chunk_overlap=5)
        first = chunk_document(loaded, config)
        second = chunk_document(loaded, config)
        assert [c.model_dump() for c in first] == [c.model_dump() for c in second]

    def test_provenance_preserved(self):
        loaded = _doc([RawSegment(text=words(10), page=4, section="Methods")])
        chunk = chunk_document(loaded, ChunkingConfig())[0]
        assert chunk.document_title == "Doc A"
        assert chunk.page == 4
        assert chunk.section == "Methods"
        assert chunk.source_path == "data/raw/a.md"
        assert chunk.text_sha256 is not None
