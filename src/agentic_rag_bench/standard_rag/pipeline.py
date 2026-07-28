"""System A — the standard (single-pass) RAG baseline.

Fixed sequence: normalize -> retrieve (dense or hybrid) -> optional rerank ->
context construction -> one generation -> citation formatting. No iteration,
by design (brief §7).
"""

import time

from agentic_rag_bench.config import SystemConfig
from agentic_rag_bench.generation import LLMError, LLMProvider, estimate_cost
from agentic_rag_bench.generation.prompts import answer_prompt
from agentic_rag_bench.observability import get_logger
from agentic_rag_bench.reranking.lexical import Reranker
from agentic_rag_bench.retrieval import Retriever
from agentic_rag_bench.schemas import (
    Citation,
    LLMCallRecord,
    PipelineAnswer,
    RetrievedChunk,
    SystemType,
)

logger = get_logger("standard_rag")

QUOTE_WORDS = 25


def build_citations(retrieved: list[RetrievedChunk], limit: int) -> list[Citation]:
    """Format citations directly from retrieved chunks — never invented by the model.

    quoted_evidence is a verbatim prefix of the chunk text, so the deterministic
    citation verifier can always check it (brief §10).
    """
    citations = []
    for r in retrieved[:limit]:
        words = r.chunk.text.split()
        quote = " ".join(words[:QUOTE_WORDS])
        citations.append(
            Citation(
                document_id=r.chunk.document_id,
                title=r.chunk.document_title,
                page=r.chunk.page,
                chunk_id=r.chunk.chunk_id,
                quoted_evidence=quote,
                retrieval_score=r.score,
            )
        )
    return citations


def fit_context(retrieved: list[RetrievedChunk], token_limit: int) -> list[RetrievedChunk]:
    """Keep top-ranked chunks until the (word-approximated) context budget is full."""
    kept: list[RetrievedChunk] = []
    used = 0
    for r in retrieved:
        cost = len(r.chunk.text.split())
        if kept and used + cost > token_limit:
            break
        kept.append(r)
        used += cost
    return kept


class StandardRAGPipeline:
    """One retrieval pass, one generation. All parameters from SystemConfig."""

    def __init__(
        self,
        config: SystemConfig,
        retriever: Retriever,
        llm: LLMProvider,
        reranker: Reranker | None = None,
    ) -> None:
        if config.system_type is not SystemType.STANDARD_RAG:
            raise ValueError(
                f"StandardRAGPipeline requires standard_rag config, got {config.system_type}"
            )
        if config.reranker.enabled and reranker is None:
            raise ValueError("config enables reranking but no reranker was provided")
        self._config = config
        self._retriever = retriever
        self._llm = llm
        self._reranker = reranker

    def run(self, query_id: str, question: str) -> PipelineAnswer:
        started = time.perf_counter()
        log = logger.bind(query_id=query_id, system_type=SystemType.STANDARD_RAG.value)

        result = self._retriever.retrieve(question)
        retrieved = result.chunks
        log.info("retrieval_done", method=result.method.value, chunks=len(retrieved))

        if self._config.reranker.enabled and self._reranker is not None and retrieved:
            retrieved = self._reranker.rerank(
                question, retrieved, top_n=self._config.reranker.top_n
            )
            log.info("rerank_done", kept=len(retrieved))

        context = fit_context(retrieved, self._config.generation.context_token_limit)

        try:
            response = self._llm.complete(
                answer_prompt(question, context),
                purpose="answer",
                max_tokens=self._config.generation.max_output_tokens,
                temperature=self._config.generation.temperature,
            )
        except LLMError as exc:
            log.error("generation_failed", error=str(exc))
            return PipelineAnswer(
                query_id=query_id,
                system_type=SystemType.STANDARD_RAG,
                question=question,
                answer="",
                retrieved_chunks=retrieved,
                latency_seconds=time.perf_counter() - started,
                failed=True,
                failure_reason=str(exc),
            )

        cost = estimate_cost(response.model, response.usage)
        call = LLMCallRecord(
            model=response.model,
            purpose="answer",
            usage=response.usage,
            latency_seconds=response.latency_seconds,
            cost_usd=cost,
        )
        return PipelineAnswer(
            query_id=query_id,
            system_type=SystemType.STANDARD_RAG,
            question=question,
            answer=response.text,
            citations=build_citations(context, limit=len(context)),
            retrieved_chunks=retrieved,
            llm_calls=[call],
            usage=response.usage,
            cost_usd=cost,
            latency_seconds=time.perf_counter() - started,
            retrieval_rounds=1,
        )
