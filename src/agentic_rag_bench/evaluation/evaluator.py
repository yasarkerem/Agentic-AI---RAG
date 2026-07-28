"""Combine retrieval, answer, and system metrics into EvaluationScores."""

from agentic_rag_bench.evaluation.judge import DeterministicJudge, LLMJudge
from agentic_rag_bench.evaluation.retrieval_metrics import compute_retrieval_metrics
from agentic_rag_bench.retrieval.store import ChunkStore
from agentic_rag_bench.schemas import (
    BenchmarkTask,
    EvaluationScores,
    PipelineAnswer,
    SystemMetrics,
)


def compute_system_metrics(answer: PipelineAnswer) -> SystemMetrics:
    return SystemMetrics(
        latency_seconds=answer.latency_seconds,
        retrieval_rounds=answer.retrieval_rounds,
        llm_calls=len(answer.llm_calls),
        input_tokens=answer.usage.input_tokens,
        output_tokens=answer.usage.output_tokens,
        estimated_cost_usd=answer.cost_usd,
        failed=answer.failed,
    )


class Evaluator:
    """Deterministic checks always run; the LLM judge is optional and additive."""

    def __init__(self, store: ChunkStore, k: int, llm_judge: LLMJudge | None = None) -> None:
        self._store = store
        self._k = k
        self._deterministic = DeterministicJudge()
        self._llm_judge = llm_judge

    def evaluate(self, task: BenchmarkTask, answer: PipelineAnswer) -> EvaluationScores:
        retrieval = compute_retrieval_metrics(
            answer.retrieved_chunks, task.gold_evidence, k=self._k
        )
        answer_metrics = self._deterministic.score(task, answer, self._store)
        evaluators = [self._deterministic.info]

        if self._llm_judge is not None:
            judged = self._llm_judge.score(task, answer, self._store)
            if judged is not None:
                evaluators.append(self._llm_judge.info)
                # Judged scores are recorded alongside; deterministic values win
                # for citation metrics (they are exact), judged values win for
                # graded qualities where lexical proxies are weakest.
                answer_metrics = answer_metrics.model_copy(
                    update={
                        "factual_correctness": judged.factual_correctness,
                        "completeness": judged.completeness,
                        "groundedness": judged.groundedness,
                        "hallucination_rate": judged.hallucination_rate,
                        "structure_readability": judged.structure_readability,
                        "contradiction_handling": (
                            judged.contradiction_handling
                            if judged.contradiction_handling is not None
                            else answer_metrics.contradiction_handling
                        ),
                    }
                )

        return EvaluationScores(
            retrieval=retrieval,
            answer=answer_metrics,
            system=compute_system_metrics(answer),
            evaluators=evaluators,
        )
