"""Markdown/PDF loaders: frontmatter, sections, failure modes."""

import pytest

from agentic_rag_bench.ingestion import IngestionError, load_document, load_markdown
from agentic_rag_bench.ingestion.loaders import make_document_id

MD = """---
title: "Sample Paper"
year: 2024
tags: [rag, test]
authors: [A. Author]
---

# Sample Paper

Intro text before any subsection.

## Methods

Method details here.

## Results

Result details here.
"""


class TestMarkdown:
    def test_frontmatter_parsed(self, tmp_path):
        path = tmp_path / "sample_paper.md"
        path.write_text(MD, encoding="utf-8")
        loaded = load_markdown(path)
        meta = loaded.document.metadata
        assert loaded.document.document_id == "sample_paper"
        assert meta.title == "Sample Paper"
        assert meta.publication_year == 2024
        assert meta.tags == ["rag", "test"]
        assert meta.authors == ["A. Author"]

    def test_sections_become_segments(self, tmp_path):
        path = tmp_path / "sample.md"
        path.write_text(MD, encoding="utf-8")
        loaded = load_markdown(path)
        sections = [seg.section for seg in loaded.segments]
        assert sections == ["Sample Paper", "Methods", "Results"]
        assert "Method details" in loaded.segments[1].text

    def test_no_frontmatter_falls_back_to_h1(self, tmp_path):
        path = tmp_path / "plain.md"
        path.write_text("# Plain Title\n\nBody text.\n", encoding="utf-8")
        loaded = load_markdown(path)
        assert loaded.document.metadata.title == "Plain Title"
        assert loaded.document.metadata.publication_year is None

    def test_empty_file_rejected(self, tmp_path):
        path = tmp_path / "empty.md"
        path.write_text("", encoding="utf-8")
        with pytest.raises(IngestionError, match="no text content"):
            load_markdown(path)

    def test_malformed_frontmatter_rejected(self, tmp_path):
        path = tmp_path / "bad.md"
        path.write_text("---\ntitle: [unclosed\n---\n\nBody\n", encoding="utf-8")
        with pytest.raises(IngestionError, match="frontmatter"):
            load_markdown(path)


class TestDispatch:
    def test_unsupported_extension_rejected(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("hello", encoding="utf-8")
        with pytest.raises(IngestionError, match="Unsupported document type"):
            load_document(path)


class TestDocumentIds:
    def test_slugified_and_deterministic(self, tmp_path):
        assert make_document_id(tmp_path / "My Paper (v2).md") == "my_paper_v2"
        assert make_document_id(tmp_path / "My Paper (v2).md") == "my_paper_v2"

    def test_valid_against_schema_pattern(self, tmp_path):
        from agentic_rag_bench.schemas.documents import _DOCUMENT_ID_PATTERN

        doc_id = make_document_id(tmp_path / "RAPTOR-2024_final.pdf")
        assert _DOCUMENT_ID_PATTERN.match(doc_id)
