"""Deterministic citation verification (brief §10).

Checks that require no model: the cited chunk exists, and the quoted
evidence appears verbatim in that chunk. Claim-level support checking
(entailment) is a judged metric added in the evaluation layer.
"""

import re

from agentic_rag_bench.retrieval.store import ChunkStore
from agentic_rag_bench.schemas import Citation, CitationVerdict, CitationVerification

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def verify_citation(citation: Citation, store: ChunkStore) -> CitationVerification:
    """Verify one citation against the chunk store."""
    chunk = store.get(citation.chunk_id)
    if chunk is None:
        return CitationVerification(
            citation=citation,
            verdict=CitationVerdict.CHUNK_NOT_FOUND,
            detail=f"chunk {citation.chunk_id} is not in the store",
        )
    if _normalize(citation.quoted_evidence) not in _normalize(chunk.text):
        return CitationVerification(
            citation=citation,
            verdict=CitationVerdict.QUOTE_NOT_IN_CHUNK,
            detail="quoted evidence is not a verbatim passage of the cited chunk",
        )
    return CitationVerification(citation=citation, verdict=CitationVerdict.SUPPORTED)


def verify_citations(citations: list[Citation], store: ChunkStore) -> list[CitationVerification]:
    return [verify_citation(c, store) for c in citations]
