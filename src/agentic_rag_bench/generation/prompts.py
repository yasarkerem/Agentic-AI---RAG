"""Versioned prompt builders shared by both pipelines.

Every prompt embeds a machine-readable payload (see generation.base) so the
mock provider stays deterministic. PROMPT_VERSIONS is recorded with every
experiment (brief §12): scores are only comparable across identical prompts.
"""

from typing import Any

from agentic_rag_bench.generation.base import embed_payload
from agentic_rag_bench.schemas import RetrievedChunk

PROMPT_VERSIONS: dict[str, str] = {
    "answer": "answer/v1",
    "analyze": "analyze/v1",
    "plan": "plan/v1",
    "assess": "assess/v1",
    "rewrite": "rewrite/v1",
    "contradictions": "contradictions/v1",
    "synthesize": "synthesize/v1",
}


def _chunk_payload(retrieved: list[RetrievedChunk], text_limit: int = 700) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": r.chunk.chunk_id,
            "title": r.chunk.document_title,
            "text": r.chunk.text[:text_limit],
        }
        for r in retrieved
    ]


def _render_context(retrieved: list[RetrievedChunk]) -> str:
    blocks = []
    for r in retrieved:
        header = f"[{r.chunk.chunk_id}] {r.chunk.document_title}"
        if r.chunk.section:
            header += f" — {r.chunk.section}"
        blocks.append(f"{header}\n{r.chunk.text}")
    return "\n\n".join(blocks) if blocks else "(no context retrieved)"


def answer_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    instructions = (
        "Answer the question using ONLY the context below. Cite evidence by "
        "referencing chunk IDs in square brackets, e.g. [paper_001_p0_c2]. "
        "If the context is insufficient, say so explicitly instead of guessing.\n\n"
        f"QUESTION: {question}\n\nCONTEXT:\n{_render_context(retrieved)}"
    )
    return embed_payload(instructions, {"question": question, "chunks": _chunk_payload(retrieved)})


def analyze_prompt(question: str) -> str:
    instructions = (
        "Classify the complexity of this research question as 'simple' (single "
        "lookup) or 'complex' (multi-step research), and list key entities. "
        'Respond with JSON: {"complexity": ..., "entities": [...]}.\n\n'
        f"QUESTION: {question}"
    )
    return embed_payload(instructions, {"question": question})


def plan_prompt(question: str, max_subquestions: int) -> str:
    instructions = (
        f"Decompose this research question into at most {max_subquestions} focused "
        "subquestions that together cover the evidence needed. Respond with JSON: "
        '{"subquestions": [...]}.\n\n'
        f"QUESTION: {question}"
    )
    return embed_payload(instructions, {"question": question, "max_subquestions": max_subquestions})


def assess_prompt(question: str, evidence_counts: dict[str, int], min_evidence: int) -> str:
    instructions = (
        "Assess whether the evidence retrieved per subquestion is sufficient to "
        "answer the overall question. Respond with JSON: "
        '{"sufficient": bool, "weak_subquestions": [...]}.\n\n'
        f"QUESTION: {question}\nEVIDENCE COUNTS: {evidence_counts}"
    )
    return embed_payload(
        instructions,
        {
            "question": question,
            "evidence_counts": evidence_counts,
            "min_evidence": min_evidence,
        },
    )


def rewrite_prompt(query: str) -> str:
    instructions = (
        "This search query retrieved weak results. Rewrite it to be more effective "
        'for lexical and dense retrieval. Respond with JSON: {"query": ...}.\n\n'
        f"QUERY: {query}"
    )
    return embed_payload(instructions, {"query": query})


def contradictions_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    instructions = (
        "Identify claims in the context that conflict with each other across "
        "sources. Respond with JSON: "
        '{"contradictions": [{"chunk_ids": [...], "description": ...}]}.\n\n'
        f"QUESTION: {question}\n\nCONTEXT:\n{_render_context(retrieved)}"
    )
    return embed_payload(
        instructions, {"question": question, "chunks": _chunk_payload(retrieved, 400)}
    )


def synthesize_prompt(
    question: str, retrieved: list[RetrievedChunk], contradictions: list[dict[str, Any]]
) -> str:
    contradiction_note = f"\nKNOWN CONTRADICTIONS: {contradictions}" if contradictions else ""
    instructions = (
        "Write a structured, well-organized answer to the research question using "
        "ONLY the context below. Cite evidence by chunk ID in square brackets. "
        "Present conflicting findings fairly and distinguish evidence from "
        f"interpretation.{contradiction_note}\n\n"
        f"QUESTION: {question}\n\nCONTEXT:\n{_render_context(retrieved)}"
    )
    return embed_payload(
        instructions,
        {
            "question": question,
            "chunks": _chunk_payload(retrieved),
            "contradictions": contradictions,
        },
    )
