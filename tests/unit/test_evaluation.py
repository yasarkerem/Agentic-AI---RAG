"""Retrieval metrics, judges, evaluator, and the committed benchmark file."""

from pathlib import Path

import pytest

from agentic_rag_bench.evaluation import (
    BenchmarkError,
    DeterministicJudge,
    Evaluator,
    LLMJudge,
    compute_retrieval_metrics,
    load_benchmark,
)
from agentic_rag_bench.generation import MockLLMProvider
from agentic_rag_bench.retrieval import ChunkStore
from agentic_rag_bench.schemas import (
    BenchmarkTask,
    Chunk,
    Citation,
    GoldEvidence,
    PipelineAnswer,
    RetrievalMethod,
    RetrievedChunk,
    SystemType,
    TaskLevel,
    TokenUsage,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = REPO_ROOT / "data" / "benchmark" / "benchmark_v0.1.json"
CORPUS = REPO_ROOT / "data" / "corpus_sample"


def _chunk(doc: str, index: int, text: str = "some text about retrieval") -> Chunk:
    return Chunk(
        chunk_id=f"{doc}_p0_c{index}",
        document_id=doc,
        document_title=doc.replace("_", " ").title(),
        text=text,
        source_path=f"{doc}.md",
        chunk_index=index,
    )


def _retrieved(doc: str, rank: int, text: str = "some text about retrieval") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=_chunk(doc, rank, text),
        score=1.0 / rank,
        method=RetrievalMethod.DENSE,
        rank=rank,
        query="q",
    )


class TestRetrievalMetrics:
    def test_perfect_retrieval(self):
        retrieved = [_retrieved("a", 1), _retrieved("b", 2)]
        gold = [GoldEvidence(document_id="a"), GoldEvidence(document_id="b")]
        metrics = compute_retrieval_metrics(retrieved, gold, k=2)
        assert metrics is not None
        assert metrics.recall_at_k == 1.0
        assert metrics.precision_at_k == 1.0
        assert metrics.mrr == 1.0
        assert metrics.ndcg == pytest.approx(1.0)
        assert metrics.source_coverage == 1.0
        assert metrics.unique_relevant_documents == 2

    def test_miss_everything(self):
        metrics = compute_retrieval_metrics(
            [_retrieved("x", 1)], [GoldEvidence(document_id="a")], k=5
        )
        assert metrics is not None
        assert metrics.recall_at_k == 0.0
        assert metrics.mrr == 0.0
        assert metrics.ndcg == 0.0

    def test_hit_at_rank_two(self):
        retrieved = [_retrieved("x", 1), _retrieved("a", 2)]
        metrics = compute_retrieval_metrics(retrieved, [GoldEvidence(document_id="a")], k=5)
        assert metrics is not None
        assert metrics.mrr == pytest.approx(0.5)

    def test_coverage_beyond_k(self):
        """A gold doc retrieved below the cutoff counts for coverage, not recall@k."""
        retrieved = [_retrieved("x", 1), _retrieved("y", 2), _retrieved("a", 3)]
        metrics = compute_retrieval_metrics(retrieved, [GoldEvidence(document_id="a")], k=2)
        assert metrics is not None
        assert metrics.recall_at_k == 0.0
        assert metrics.source_coverage == 1.0

    def test_no_gold_returns_none(self):
        assert compute_retrieval_metrics([_retrieved("a", 1)], [], k=5) is None

    def test_empty_retrieval_scores_zero(self):
        metrics = compute_retrieval_metrics([], [GoldEvidence(document_id="a")], k=5)
        assert metrics is not None
        assert metrics.recall_at_k == 0.0
        assert metrics.source_coverage == 0.0


def _task(level: TaskLevel = TaskLevel.SINGLE_DOC_FACTUAL) -> BenchmarkTask:
    return BenchmarkTask(
        task_id="t1",
        level=level,
        question="What is RAPTOR?",
        reference_answer="RAPTOR builds a hierarchical tree of recursive summaries.",
        gold_evidence=[GoldEvidence(document_id="raptor")],
    )


def _answer(text: str, citations: list[Citation], retrieved: list[RetrievedChunk]):
    return PipelineAnswer(
        query_id="t1",
        system_type=SystemType.STANDARD_RAG,
        question="What is RAPTOR?",
        answer=text,
        citations=citations,
        retrieved_chunks=retrieved,
        usage=TokenUsage(input_tokens=10, output_tokens=10),
        latency_seconds=0.1,
    )


class TestDeterministicJudge:
    def _setup(self):
        chunk_text = "RAPTOR builds a hierarchical tree of recursive summaries over chunks."
        retrieved = [_retrieved("raptor", 1, text=chunk_text)]
        store = ChunkStore([r.chunk for r in retrieved])
        citation = Citation(
            document_id="raptor",
            title="Raptor",
            page=0,
            chunk_id="raptor_p0_c1",
            quoted_evidence="RAPTOR builds a hierarchical tree",
            retrieval_score=1.0,
        )
        return retrieved, store, citation

    def test_good_answer_scores_high(self):
        retrieved, store, citation = self._setup()
        answer = _answer(
            "RAPTOR builds a hierarchical tree of recursive summaries. [raptor_p0_c1]",
            [citation],
            retrieved,
        )
        metrics = DeterministicJudge().score(_task(), answer, store)
        assert metrics.factual_correctness > 0.8
        assert metrics.citation_correctness == 1.0
        assert metrics.citation_completeness == 1.0
        assert metrics.groundedness > 0.9
        assert metrics.hallucination_rate < 0.1
        assert metrics.contradiction_handling is None  # level 1 task

    def test_ungrounded_answer_scores_low(self):
        retrieved, store, _ = self._setup()
        answer = _answer(
            "Quantum entanglement flavors the espresso singularity greatly.",
            [],
            retrieved,
        )
        metrics = DeterministicJudge().score(_task(), answer, store)
        assert metrics.groundedness < 0.3
        assert metrics.hallucination_rate > 0.7
        assert metrics.citation_correctness == 0.0

    def test_empty_answer_scores_zero(self):
        retrieved, store, _ = self._setup()
        metrics = DeterministicJudge().score(_task(), _answer("", [], retrieved), store)
        assert metrics.factual_correctness == 0.0
        assert metrics.structure_readability == 0.0

    def test_contradiction_handling_scored_for_level4(self):
        retrieved, store, citation = self._setup()
        second = Citation(
            document_id="other",
            title="Other",
            page=0,
            chunk_id="other_p0_c1",
            quoted_evidence="text",
            retrieval_score=0.5,
        )
        task = BenchmarkTask(
            task_id="t4",
            level=TaskLevel.CONTRADICTION_ANALYSIS,
            question="q?",
            reference_answer="ref",
            gold_evidence=[GoldEvidence(document_id="raptor"), GoldEvidence(document_id="other")],
        )
        answer = _answer(
            "One study finds gains; however the other disagrees.", [citation, second], retrieved
        )
        metrics = DeterministicJudge().score(task, answer, store)
        assert metrics.contradiction_handling == 1.0


class TestEvaluator:
    def test_combines_all_metric_families(self):
        chunk_text = "RAPTOR builds a hierarchical tree of recursive summaries over chunks."
        retrieved = [_retrieved("raptor", 1, text=chunk_text)]
        store = ChunkStore([r.chunk for r in retrieved])
        answer = _answer("RAPTOR builds a tree of summaries.", [], retrieved)
        scores = Evaluator(store, k=5).evaluate(_task(), answer)
        assert scores.retrieval is not None
        assert scores.answer is not None
        assert scores.system is not None
        assert scores.system.llm_calls == 0
        assert [e.kind for e in scores.evaluators] == ["deterministic"]

    def test_llm_judge_recorded_when_present(self):
        retrieved = [_retrieved("raptor", 1)]
        store = ChunkStore([r.chunk for r in retrieved])
        evaluator = Evaluator(store, k=5, llm_judge=LLMJudge(MockLLMProvider()))
        scores = evaluator.evaluate(_task(), _answer("text answer here", [], retrieved))
        kinds = [e.kind for e in scores.evaluators]
        assert kinds == ["deterministic", "llm_judge"]
        assert scores.answer is not None
        assert scores.answer.factual_correctness == 0.5  # mock judge flat score


class TestCommittedBenchmark:
    def test_loads_and_meets_size_requirements(self):
        manifest = load_benchmark(BENCHMARK)
        assert len(manifest.tasks) >= 25
        for level in TaskLevel:
            assert len(manifest.tasks_at_level(level)) >= 5, f"level {level} too small"

    def test_gold_documents_exist_in_sample_corpus(self):
        manifest = load_benchmark(BENCHMARK)
        corpus_ids = {p.stem for p in CORPUS.glob("*.md")}
        for task in manifest.tasks:
            for gold in task.gold_evidence:
                assert gold.document_id in corpus_ids, (
                    f"{task.task_id} references unknown document {gold.document_id}"
                )

    def test_every_task_has_reference_answer_and_gold(self):
        manifest = load_benchmark(BENCHMARK)
        for task in manifest.tasks:
            assert task.reference_answer, task.task_id
            assert task.gold_evidence, task.task_id

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(BenchmarkError, match="not found"):
            load_benchmark(tmp_path / "nope.json")

    def test_builder_matches_committed_file(self):
        """The committed benchmark must be regenerable from the builder (no drift)."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "build_benchmark", REPO_ROOT / "scripts" / "build_benchmark.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rebuilt = {t.task_id for t in module.build_tasks()}
        committed = {t.task_id for t in load_benchmark(BENCHMARK).tasks}
        assert rebuilt == committed
