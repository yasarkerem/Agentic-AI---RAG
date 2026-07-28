"""Answer-quality judges (brief §11).

Two evaluators, always used in combination — the LLM judge is never the
only evaluator:

- :class:`DeterministicJudge` — versioned heuristic proxies computed from
  citations, retrieved evidence, and the reference answer. Free, exact,
  reproducible. Its limits are documented per metric below.
- :class:`LLMJudge` — a versioned judged pass behind the LLMProvider
  abstraction. With the mock provider it returns flat mid-scale scores
  (harness validation only); real judging requires a real model.
"""

import json
import re

from agentic_rag_bench.evaluation.citation_checks import verify_citations
from agentic_rag_bench.generation import LLMError, LLMProvider, embed_payload
from agentic_rag_bench.retrieval.bm25 import tokenize
from agentic_rag_bench.retrieval.store import ChunkStore
from agentic_rag_bench.schemas import (
    AnswerMetrics,
    BenchmarkTask,
    EvaluatorInfo,
    PipelineAnswer,
    TaskLevel,
)

_CONTRAST_MARKERS = (
    "however",
    "whereas",
    "in contrast",
    "disagree",
    "conflict",
    "on the other hand",
    "differs",
    "opposing",
)


def _token_f1(candidate: str, reference: str) -> tuple[float, float, float]:
    """(precision, recall, f1) over content tokens."""
    cand = tokenize(candidate)
    ref = tokenize(reference)
    if not cand or not ref:
        return 0.0, 0.0, 0.0
    cand_set, ref_set = set(cand), set(ref)
    overlap = cand_set & ref_set
    precision = len(overlap) / len(cand_set)
    recall = len(overlap) / len(ref_set)
    f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
    return precision, recall, f1


class DeterministicJudge:
    """Heuristic, fully reproducible answer scoring.

    Metric proxies (documented limitations):
    - factual_correctness: token-F1 against the reference answer — measures
      lexical agreement, not truth.
    - completeness: token recall of the reference answer's content words.
    - citation_correctness: fraction of citations passing deterministic checks.
    - citation_completeness: fraction of gold documents that are cited.
    - groundedness: fraction of answer content words present in retrieved
      evidence — lexical proxy for support.
    - hallucination_rate: 1 - groundedness.
    - contradiction_handling (L4+ only): cites >=2 distinct gold documents
      AND uses contrastive language.
    - structure_readability: crude length/structure heuristic.
    """

    info = EvaluatorInfo(name="deterministic-judge", kind="deterministic", version="v1")

    def score(
        self, task: BenchmarkTask, answer: PipelineAnswer, store: ChunkStore
    ) -> AnswerMetrics:
        text = answer.answer or ""
        reference = task.reference_answer or ""
        gold_docs = {g.document_id for g in task.gold_evidence}

        _, ref_recall, f1 = _token_f1(text, reference)

        verifications = verify_citations(answer.citations, store)
        valid = [v for v in verifications if v.is_valid]
        citation_correctness = len(valid) / len(verifications) if verifications else 0.0
        cited_docs = {c.document_id for c in answer.citations}
        citation_completeness = len(cited_docs & gold_docs) / len(gold_docs) if gold_docs else 0.0

        evidence_tokens: set[str] = set()
        for retrieved in answer.retrieved_chunks:
            evidence_tokens.update(tokenize(retrieved.chunk.text))
        answer_tokens = set(tokenize(re.sub(r"\[[\w\-]+\]", " ", text)))
        groundedness = (
            len(answer_tokens & evidence_tokens) / len(answer_tokens) if answer_tokens else 0.0
        )

        contradiction_handling: float | None = None
        if task.level >= TaskLevel.CONTRADICTION_ANALYSIS:
            uses_contrast = any(marker in text.lower() for marker in _CONTRAST_MARKERS)
            covers_sides = len(cited_docs & gold_docs) >= 2
            contradiction_handling = (
                1.0 if (uses_contrast and covers_sides) else 0.5 if covers_sides else 0.0
            )

        words = len(text.split())
        if words == 0:
            structure = 0.0
        elif words < 30:
            structure = 0.5
        else:
            structure = 1.0 if "\n" in text.strip() or words >= 60 else 0.75

        return AnswerMetrics(
            factual_correctness=f1,
            completeness=ref_recall,
            citation_correctness=citation_correctness,
            citation_completeness=citation_completeness,
            groundedness=groundedness,
            hallucination_rate=1.0 - groundedness,
            contradiction_handling=contradiction_handling,
            structure_readability=structure,
        )


JUDGE_PROMPT_VERSION = "judge/v1"


class LLMJudge:
    """LLM-as-judge behind the provider abstraction. Versioned (brief §11)."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self.info = EvaluatorInfo(
            name="llm-judge",
            kind="llm_judge",
            version="v1",
            judge_model=provider.model_name,
            prompt_version=JUDGE_PROMPT_VERSION,
        )

    def score(
        self, task: BenchmarkTask, answer: PipelineAnswer, store: ChunkStore
    ) -> AnswerMetrics | None:
        """Judged scores, or None when the judge call fails or returns junk."""
        fields = [
            "factual_correctness",
            "completeness",
            "citation_correctness",
            "citation_completeness",
            "groundedness",
            "structure_readability",
        ]
        instructions = (
            "You are grading a RAG system's answer. Score each dimension from 0 to 1. "
            f"Respond with JSON containing exactly these keys: {fields} plus "
            '"contradiction_handling" (0-1 or null).\n\n'
            f"QUESTION: {task.question}\n\nREFERENCE ANSWER: {task.reference_answer}\n\n"
            f"CANDIDATE ANSWER:\n{answer.answer}"
        )
        prompt = embed_payload(
            instructions,
            {"question": task.question, "task_level": int(task.level)},
        )
        try:
            response = self._provider.complete(prompt, purpose="judge")
        except LLMError:
            return None
        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        try:
            scores = {name: float(parsed[name]) for name in fields}
        except (KeyError, TypeError, ValueError):
            return None
        raw_contra = parsed.get("contradiction_handling")
        contradiction = float(raw_contra) if isinstance(raw_contra, int | float) else None
        return AnswerMetrics(
            **{name: max(0.0, min(1.0, value)) for name, value in scores.items()},
            hallucination_rate=max(0.0, min(1.0, 1.0 - scores["groundedness"])),
            contradiction_handling=contradiction,
        )
