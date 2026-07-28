"""System B — the agentic RAG pipeline, implemented as a controlled state machine.

Nodes (brief §8): query analyzer -> planner -> [retriever -> evidence assessor
-> query rewriter]* -> contradiction checker -> synthesizer -> citation
verifier -> finalizer. Every loop is bounded by AgentConfig; every run records
a full AgentTrace with an explicit StopReason. It cannot loop indefinitely.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

from agentic_rag_bench.config import SystemConfig
from agentic_rag_bench.evaluation.citation_checks import verify_citations
from agentic_rag_bench.generation import LLMError, LLMProvider, LLMResponse, estimate_cost, prompts
from agentic_rag_bench.observability import get_logger
from agentic_rag_bench.reranking.lexical import Reranker
from agentic_rag_bench.retrieval import Retriever
from agentic_rag_bench.schemas import (
    AgentNode,
    AgentStep,
    AgentTrace,
    Citation,
    LLMCallRecord,
    PipelineAnswer,
    RetrievedChunk,
    StopReason,
    SystemType,
    TokenUsage,
)
from agentic_rag_bench.standard_rag.pipeline import build_citations, fit_context

logger = get_logger("agentic_rag")


@dataclass
class _RunState:
    """Mutable state threaded through the machine for one run."""

    question: str
    query_id: str
    deadline: float
    subquestions: list[str] = field(default_factory=list)
    queries: dict[str, str] = field(default_factory=dict)  # subquestion -> current query
    evidence: dict[str, list[RetrievedChunk]] = field(default_factory=dict)
    seen_chunk_ids: set[str] = field(default_factory=set)
    steps: list[AgentStep] = field(default_factory=list)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    rounds: int = 0
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: StopReason | None = None

    def all_evidence(self) -> list[RetrievedChunk]:
        """Evidence across subquestions, deduplicated, best score first."""
        best: dict[str, RetrievedChunk] = {}
        for chunks in self.evidence.values():
            for r in chunks:
                current = best.get(r.chunk.chunk_id)
                if current is None or r.score > current.score:
                    best[r.chunk.chunk_id] = r
        return sorted(best.values(), key=lambda r: (-r.score, r.chunk.chunk_id))


class AgenticRAGPipeline:
    """Iterative retrieval under hard budget controls."""

    def __init__(
        self,
        config: SystemConfig,
        retriever: Retriever,
        llm: LLMProvider,
        reranker: Reranker | None = None,
    ) -> None:
        if config.system_type is not SystemType.AGENTIC_RAG or config.agent is None:
            raise ValueError(
                "AgenticRAGPipeline requires an agentic_rag config with an agent section"
            )
        if config.reranker.enabled and reranker is None:
            raise ValueError("config enables reranking but no reranker was provided")
        self._config = config
        self._agent = config.agent
        self._retriever = retriever
        self._llm = llm
        self._reranker = reranker

    # -- infrastructure -----------------------------------------------------

    def _budget_stop(self, state: _RunState) -> StopReason | None:
        if state.usage.total_tokens > self._agent.max_total_tokens:
            return StopReason.TOKEN_BUDGET_EXHAUSTED
        if state.cost_usd > self._agent.max_cost_usd:
            return StopReason.COST_BUDGET_EXHAUSTED
        if time.monotonic() > state.deadline:
            return StopReason.TIMEOUT
        return None

    def _call_llm(
        self, state: _RunState, node: AgentNode, purpose: str, prompt: str
    ) -> LLMResponse:
        """LLM call with bounded retries; accounting recorded on state."""
        last_error: LLMError | None = None
        for attempt in range(self._agent.max_retries + 1):
            try:
                response = self._llm.complete(
                    prompt,
                    purpose=purpose,
                    max_tokens=self._config.generation.max_output_tokens,
                    temperature=self._config.generation.temperature,
                )
                break
            except LLMError as exc:
                last_error = exc
                logger.warning("llm_retry", node=node.value, attempt=attempt + 1, error=str(exc))
        else:
            raise last_error if last_error else LLMError("provider failed")

        cost = estimate_cost(response.model, response.usage)
        state.usage = state.usage + response.usage
        state.cost_usd += cost
        state.llm_calls.append(
            LLMCallRecord(
                model=response.model,
                purpose=purpose,
                usage=response.usage,
                latency_seconds=response.latency_seconds,
                cost_usd=cost,
            )
        )
        return response

    def _record(
        self,
        state: _RunState,
        node: AgentNode,
        summary: str,
        usage: TokenUsage | None = None,
        **snapshots: Any,
    ) -> None:
        state.steps.append(
            AgentStep(
                step_index=len(state.steps),
                node=node,
                round_index=state.rounds,
                summary=summary,
                output_snapshot=snapshots,
                usage=usage if usage is not None else TokenUsage(),
            )
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    # -- nodes ---------------------------------------------------------------

    def _analyze(self, state: _RunState) -> None:
        response = self._call_llm(
            state, AgentNode.QUERY_ANALYZER, "analyze", prompts.analyze_prompt(state.question)
        )
        parsed = self._parse_json(response.text)
        complexity = parsed.get("complexity", "unknown")
        self._record(
            state,
            AgentNode.QUERY_ANALYZER,
            f"classified task as {complexity}",
            usage=response.usage,
            complexity=complexity,
        )

    def _plan(self, state: _RunState) -> None:
        response = self._call_llm(
            state,
            AgentNode.PLANNER,
            "plan",
            prompts.plan_prompt(state.question, self._agent.max_subquestions),
        )
        parsed = self._parse_json(response.text)
        raw = parsed.get("subquestions", [])
        subquestions = [str(s) for s in raw if str(s).strip()][: self._agent.max_subquestions]
        if not subquestions:
            subquestions = [state.question]
        state.subquestions = subquestions
        state.queries = {subq: subq for subq in subquestions}
        self._record(
            state,
            AgentNode.PLANNER,
            f"planned {len(subquestions)} subquestions",
            usage=response.usage,
            subquestions=subquestions,
        )

    def _retrieve_round(self, state: _RunState, targets: list[str]) -> None:
        state.rounds += 1
        found = 0
        for subq in targets:
            result = self._retriever.retrieve(state.queries[subq])
            chunks = result.chunks
            if self._config.reranker.enabled and self._reranker is not None and chunks:
                chunks = self._reranker.rerank(
                    state.queries[subq], chunks, top_n=self._config.reranker.top_n
                )
            fresh = [r for r in chunks if r.chunk.chunk_id not in state.seen_chunk_ids]
            state.seen_chunk_ids.update(r.chunk.chunk_id for r in fresh)
            state.evidence.setdefault(subq, []).extend(fresh)
            found += len(fresh)
        self._record(
            state,
            AgentNode.RETRIEVER,
            f"round {state.rounds}: retrieved {found} new chunks for {len(targets)} subquestions",
            new_chunks=found,
            targets=targets,
        )

    def _assess(self, state: _RunState) -> tuple[bool, list[str]]:
        counts = {subq: len(state.evidence.get(subq, [])) for subq in state.subquestions}
        response = self._call_llm(
            state,
            AgentNode.EVIDENCE_ASSESSOR,
            "assess",
            prompts.assess_prompt(state.question, counts, min_evidence=2),
        )
        parsed = self._parse_json(response.text)
        sufficient = bool(parsed.get("sufficient", False))
        weak = [str(s) for s in parsed.get("weak_subquestions", []) if str(s) in state.queries]
        self._record(
            state,
            AgentNode.EVIDENCE_ASSESSOR,
            f"evidence {'sufficient' if sufficient else 'insufficient'}",
            usage=response.usage,
            sufficient=sufficient,
            weak_subquestions=weak,
            evidence_counts=counts,
        )
        return sufficient, weak

    def _rewrite(self, state: _RunState, weak: list[str]) -> list[str]:
        rewritten = []
        for subq in weak:
            response = self._call_llm(
                state,
                AgentNode.QUERY_REWRITER,
                "rewrite",
                prompts.rewrite_prompt(state.queries[subq]),
            )
            parsed = self._parse_json(response.text)
            new_query = str(parsed.get("query", "")).strip()
            if new_query and new_query != state.queries[subq]:
                state.queries[subq] = new_query
                rewritten.append(subq)
        self._record(
            state,
            AgentNode.QUERY_REWRITER,
            f"rewrote {len(rewritten)} weak queries",
            rewritten=rewritten,
        )
        return rewritten

    def _check_contradictions(self, state: _RunState, evidence: list[RetrievedChunk]) -> None:
        response = self._call_llm(
            state,
            AgentNode.CONTRADICTION_CHECKER,
            "contradictions",
            prompts.contradictions_prompt(state.question, evidence),
        )
        parsed = self._parse_json(response.text)
        raw = parsed.get("contradictions", [])
        state.contradictions = [c for c in raw if isinstance(c, dict)]
        self._record(
            state,
            AgentNode.CONTRADICTION_CHECKER,
            f"found {len(state.contradictions)} contradictions",
            usage=response.usage,
            contradictions=state.contradictions,
        )

    def _synthesize(self, state: _RunState, evidence: list[RetrievedChunk]) -> str:
        response = self._call_llm(
            state,
            AgentNode.SYNTHESIZER,
            "synthesize",
            prompts.synthesize_prompt(state.question, evidence, state.contradictions),
        )
        self._record(state, AgentNode.SYNTHESIZER, "synthesized answer", usage=response.usage)
        return response.text

    def _verify_citations(self, state: _RunState, candidates: list[Citation]) -> list[Citation]:
        verifications = verify_citations(candidates, self._retriever.store)
        kept = [v.citation for v in verifications if v.is_valid]
        dropped = len(candidates) - len(kept)
        self._record(
            state,
            AgentNode.CITATION_VERIFIER,
            f"verified {len(candidates)} citations, dropped {dropped}",
            dropped=dropped,
        )
        return kept

    # -- run -----------------------------------------------------------------

    def run(self, query_id: str, question: str) -> PipelineAnswer:
        started = time.perf_counter()
        state = _RunState(
            question=question,
            query_id=query_id,
            deadline=time.monotonic() + self._agent.timeout_seconds,
        )
        log = logger.bind(query_id=query_id, system_type=SystemType.AGENTIC_RAG.value)

        try:
            self._analyze(state)
            self._plan(state)

            targets = list(state.subquestions)
            while state.rounds < self._agent.max_retrieval_rounds:
                self._retrieve_round(state, targets)
                stop = self._budget_stop(state)
                if stop is not None:
                    state.stop_reason = stop
                    break
                sufficient, weak = self._assess(state)
                if sufficient:
                    state.stop_reason = StopReason.EVIDENCE_SUFFICIENT
                    break
                if state.rounds >= self._agent.max_retrieval_rounds:
                    break
                rewritten = self._rewrite(state, weak)
                if not rewritten:
                    # No query changed: another round would repeat the same
                    # searches. Deterministic stop instead of a futile loop.
                    state.stop_reason = StopReason.MAX_ROUNDS_REACHED
                    break
                targets = rewritten
                stop = self._budget_stop(state)
                if stop is not None:
                    state.stop_reason = stop
                    break
            if state.stop_reason is None:
                state.stop_reason = StopReason.MAX_ROUNDS_REACHED

            evidence = fit_context(
                state.all_evidence(), self._config.generation.context_token_limit
            )
            self._check_contradictions(state, evidence)
            answer_text = self._synthesize(state, evidence)
            citations = self._verify_citations(
                state, build_citations(evidence, limit=len(evidence))
            )
        except LLMError as exc:
            log.error("agent_failed", error=str(exc))
            state.stop_reason = StopReason.ERROR
            return self._finalize(
                state, started, answer="", citations=[], failed=True, reason=str(exc)
            )

        log.info(
            "agent_done",
            stop_reason=state.stop_reason.value,
            rounds=state.rounds,
            llm_calls=len(state.llm_calls),
        )
        return self._finalize(state, started, answer=answer_text, citations=citations)

    def _finalize(
        self,
        state: _RunState,
        started: float,
        answer: str,
        citations: list[Citation],
        failed: bool = False,
        reason: str | None = None,
    ) -> PipelineAnswer:
        self._record(state, AgentNode.FINALIZER, "finalized answer")
        trace = AgentTrace(
            steps=state.steps,
            stop_reason=state.stop_reason,
            retrieval_rounds=state.rounds,
            subquestions=state.subquestions,
        )
        return PipelineAnswer(
            query_id=state.query_id,
            system_type=SystemType.AGENTIC_RAG,
            question=state.question,
            answer=answer,
            citations=citations,
            retrieved_chunks=state.all_evidence(),
            llm_calls=state.llm_calls,
            usage=state.usage,
            cost_usd=state.cost_usd,
            latency_seconds=time.perf_counter() - started,
            retrieval_rounds=state.rounds,
            trace=trace,
            failed=failed,
            failure_reason=reason,
        )
