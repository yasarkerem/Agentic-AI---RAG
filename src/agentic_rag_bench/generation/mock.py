"""Deterministic mock LLM provider — the default for tests and dev runs.

Behavior is keyed by ``purpose`` and driven by the structured payload in the
prompt, so runs are fully reproducible and free. It is NOT a language model:
answers are extractive templates. It validates the harness, not the science.
"""

import json
import time
from typing import Any

from agentic_rag_bench.generation.base import (
    LLMResponse,
    approximate_tokens,
    extract_payload,
)
from agentic_rag_bench.schemas import TokenUsage

_COMPLEX_MARKERS = (
    "compare",
    "contrast",
    "disagree",
    "conflict",
    "review",
    "synthesis",
    "literature",
    "trade-off",
    "tradeoff",
    "why",
    "most suitable",
)


def _first_sentences(text: str, limit: int = 2) -> str:
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:limit]).rstrip(".") + "."


class MockLLMProvider:
    """Purpose-aware deterministic completions."""

    @property
    def model_name(self) -> str:
        return "mock-llm"

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        started = time.perf_counter()
        payload = extract_payload(prompt)
        text = self._respond(purpose, payload)
        usage = TokenUsage(
            input_tokens=approximate_tokens(prompt),
            output_tokens=approximate_tokens(text),
        )
        return LLMResponse(
            text=text,
            usage=usage,
            model=self.model_name,
            latency_seconds=time.perf_counter() - started,
        )

    def _respond(self, purpose: str, payload: dict[str, Any]) -> str:
        if purpose == "analyze":
            return self._analyze(payload)
        if purpose == "plan":
            return self._plan(payload)
        if purpose == "assess":
            return self._assess(payload)
        if purpose == "rewrite":
            return self._rewrite(payload)
        if purpose == "contradictions":
            return json.dumps({"contradictions": []})
        if purpose in ("answer", "synthesize"):
            return self._answer(payload)
        return json.dumps({"purpose": purpose, "note": "mock default response"})

    def _analyze(self, payload: dict[str, Any]) -> str:
        question = str(payload.get("question", "")).lower()
        complex_task = any(marker in question for marker in _COMPLEX_MARKERS)
        return json.dumps({"complexity": "complex" if complex_task else "simple", "entities": []})

    def _plan(self, payload: dict[str, Any]) -> str:
        question = str(payload.get("question", ""))
        max_subquestions = int(payload.get("max_subquestions", 3))
        parts = [p.strip(" ?.") for p in question.replace(";", " and ").split(" and ")]
        subquestions = [f"{part}?" for part in parts if part][:max_subquestions]
        if not subquestions:
            subquestions = [question or "What does the corpus say?"]
        return json.dumps({"subquestions": subquestions})

    def _assess(self, payload: dict[str, Any]) -> str:
        counts = payload.get("evidence_counts", {})
        weak = [
            subq
            for subq, count in counts.items()
            if isinstance(count, int) and count < int(payload.get("min_evidence", 2))
        ]
        return json.dumps({"sufficient": not weak, "weak_subquestions": weak})

    def _rewrite(self, payload: dict[str, Any]) -> str:
        query = str(payload.get("query", ""))
        # Deterministic broadening: strip question boilerplate, keep content terms.
        stripped = " ".join(
            word
            for word in query.strip(" ?").split()
            if word.lower() not in ("what", "which", "how", "why", "is", "are", "the", "do", "does")
        )
        return json.dumps({"query": stripped or query})

    def _answer(self, payload: dict[str, Any]) -> str:
        question = str(payload.get("question", ""))
        chunks = payload.get("chunks", [])
        if not chunks:
            return (
                "The retrieved context does not contain evidence to answer this question, "
                "so no answer can be given."
            )
        lines = [f"Regarding the question: {question}"]
        for chunk in chunks[:4]:
            if not isinstance(chunk, dict):
                continue
            chunk_id = chunk.get("chunk_id", "unknown")
            text = str(chunk.get("text", ""))
            lines.append(f"According to [{chunk_id}]: {_first_sentences(text)}")
        return "\n".join(lines)
