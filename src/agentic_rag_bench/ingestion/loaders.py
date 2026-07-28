"""Document loaders: Markdown (section-aware) and PDF (page-aware)."""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from pypdf import PdfReader

from agentic_rag_bench.schemas import Document, DocumentMetadata


class IngestionError(Exception):
    """Raised when a document cannot be loaded or parsed."""


@dataclass(frozen=True)
class RawSegment:
    """A contiguous piece of document text that chunking must not cross."""

    text: str
    page: int | None
    section: str | None


@dataclass(frozen=True)
class LoadedDocument:
    """A parsed document plus its segments, ready for chunking."""

    document: Document
    segments: list[RawSegment]


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def make_document_id(path: Path) -> str:
    """Deterministic document ID from the file name (slugified stem)."""
    slug = _SLUG_RE.sub("_", path.stem.lower()).strip("_")
    if not slug:
        raise IngestionError(f"Cannot derive a document ID from {path.name!r}")
    if not slug[0].isalnum():
        slug = f"d{slug}"
    return slug


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        loaded = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise IngestionError(f"Invalid YAML frontmatter: {exc}") from exc
    meta = loaded if isinstance(loaded, dict) else {}
    return meta, text[match.end() :]


def load_markdown(path: Path) -> LoadedDocument:
    """Load a Markdown file, splitting into per-section segments."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IngestionError(f"Cannot read {path}: {exc}") from exc

    frontmatter, body = _parse_frontmatter(text)

    headings = list(_HEADING_RE.finditer(body))
    segments: list[RawSegment] = []
    first_h1: str | None = None

    preamble = body[: headings[0].start()] if headings else body
    if preamble.strip():
        segments.append(RawSegment(text=preamble.strip(), page=None, section=None))

    for i, heading in enumerate(headings):
        section_title = heading.group(2).strip()
        if first_h1 is None and heading.group(1) == "#":
            first_h1 = section_title
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        content = body[heading.end() : end].strip()
        if content:
            segments.append(RawSegment(text=content, page=None, section=section_title))

    if not segments:
        raise IngestionError(f"{path} contains no text content")

    raw_year = frontmatter.get("year")
    raw_authors = frontmatter.get("authors")
    raw_tags = frontmatter.get("tags")
    title = str(frontmatter.get("title") or first_h1 or path.stem)

    document = Document(
        document_id=make_document_id(path),
        metadata=DocumentMetadata(
            title=title,
            source_path=str(path),
            publication_year=int(raw_year) if isinstance(raw_year, int) else None,
            authors=[str(a) for a in raw_authors] if isinstance(raw_authors, list) else [],
            tags=[str(t) for t in raw_tags] if isinstance(raw_tags, list) else [],
        ),
        num_pages=None,
    )
    return LoadedDocument(document=document, segments=segments)


def load_pdf(path: Path) -> LoadedDocument:
    """Load a PDF file, one segment per page."""
    try:
        reader = PdfReader(path)
    except Exception as exc:  # pypdf raises many exception types
        raise IngestionError(f"Cannot parse PDF {path}: {exc}") from exc

    segments: list[RawSegment] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            segments.append(RawSegment(text=text, page=page_number, section=None))

    if not segments:
        raise IngestionError(f"{path} contains no extractable text")

    pdf_title = None
    if reader.metadata is not None and reader.metadata.title:
        pdf_title = str(reader.metadata.title).strip() or None

    document = Document(
        document_id=make_document_id(path),
        metadata=DocumentMetadata(
            title=pdf_title or path.stem,
            source_path=str(path),
        ),
        num_pages=len(reader.pages),
    )
    return LoadedDocument(document=document, segments=segments)


LOADERS = {".md": load_markdown, ".markdown": load_markdown, ".pdf": load_pdf}


def load_document(path: Path) -> LoadedDocument:
    """Load any supported document type; raises IngestionError otherwise."""
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise IngestionError(f"Unsupported document type: {path.name} (supported: md, pdf)")
    return loader(path)
