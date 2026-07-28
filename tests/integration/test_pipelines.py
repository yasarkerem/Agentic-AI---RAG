"""End-to-end pipeline tests over the sample corpus, entirely on mocks."""

import time
from pathlib import Path

import pytest

from agentic_rag_bench.agentic_rag import AgenticRAGPipeline
from agentic_rag_bench.config import load_system_config
from agentic_rag_bench.embeddings import build_embedder
from agentic_rag_bench.generation import LLMError, MockLLMProvider, build_llm_provider
from agentic_rag_bench.ingestion import ingest_corpus
from agentic_rag_bench.reranking import build_reranker
from agentic_rag_bench.retrieval import Retriever
from agentic_rag_bench.schemas import StopReason, SystemType
from agentic_rag_bench.standard_rag import StandardRAGPipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
SAMPLE_CORPUS = REPO_ROOT / "data" / "corpus_sample"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def processed_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    config = load_system_config(CONFIGS / "default.yaml")
    ingest_corpus(
        raw_dir=SAMPLE_CORPUS,
        processed_dir=out,
        chunking=config.chunking,
        embedding=config.embedding,
    )
    return out


def _standard(processed_dir, **overrides) -> StandardRAGPipeline:
    config = load_system_config(
        CONFIGS / "standard_rag.yaml", defaults_path=CONFIGS / "default.yaml"
    )
    config = config.model_copy(update=overrides, deep=True)
    retriever = Retriever.from_processed_dir(
        processed_dir, build_embedder(config.embedding), config.retrieval
    )
    reranker = build_reranker(config.reranker) if config.reranker.enabled else None
    return StandardRAGPipeline(config, retriever, build_llm_provider(config.generation), reranker)


def _agentic(processed_dir, llm=None) -> AgenticRAGPipeline:
    config = load_system_config(
        CONFIGS / "agentic_rag.yaml", defaults_path=CONFIGS / "default.yaml"
    )
    retriever = Retriever.from_processed_dir(
        processed_dir, build_embedder(config.embedding), config.retrieval
    )
    provider = llm if llm is not None else build_llm_provider(config.generation)
    return AgenticRAGPipeline(config, retriever, provider)


class TestStandardRAG:
    def test_answers_with_grounded_citations(self, processed_dir):
        answer = _standard(processed_dir).run("t1", "What is the main contribution of RAPTOR?")
        assert not answer.failed
        assert answer.system_type is SystemType.STANDARD_RAG
        assert answer.answer
        assert answer.citations
        assert answer.retrieval_rounds == 1
        assert answer.trace is None
        # Every citation must point at a retrieved chunk and quote it verbatim
        by_id = {r.chunk.chunk_id: r.chunk for r in answer.retrieved_chunks}
        for citation in answer.citations:
            assert citation.chunk_id in by_id
            assert citation.quoted_evidence in by_id[citation.chunk_id].text

    def test_accounting_recorded(self, processed_dir):
        answer = _standard(processed_dir).run("t2", "How does hybrid retrieval work?")
        assert answer.usage.total_tokens > 0
        assert answer.cost_usd == 0.0  # mock provider is free
        assert answer.latency_seconds > 0
        assert len(answer.llm_calls) == 1

    def test_generation_failure_is_recorded_not_raised(self, processed_dir):
        class FailingLLM:
            model_name = "failing"

            def complete(self, prompt, *, purpose, max_tokens=1024, temperature=0.0):
                raise LLMError("provider timeout")

        config = load_system_config(
            CONFIGS / "standard_rag.yaml", defaults_path=CONFIGS / "default.yaml"
        )
        retriever = Retriever.from_processed_dir(
            processed_dir, build_embedder(config.embedding), config.retrieval
        )
        pipeline = StandardRAGPipeline(config, retriever, FailingLLM())
        answer = pipeline.run("t3", "What is RAPTOR?")
        assert answer.failed
        assert "timeout" in (answer.failure_reason or "")

    def test_wrong_config_type_rejected(self, processed_dir):
        config = load_system_config(
            CONFIGS / "agentic_rag.yaml", defaults_path=CONFIGS / "default.yaml"
        )
        retriever = Retriever.from_processed_dir(
            processed_dir, build_embedder(config.embedding), config.retrieval
        )
        with pytest.raises(ValueError, match="standard_rag"):
            StandardRAGPipeline(config, retriever, MockLLMProvider())


class TestAgenticRAG:
    def test_full_run_with_trace(self, processed_dir):
        answer = _agentic(processed_dir).run(
            "t1", "Compare GraphRAG and RAPTOR in terms of retrieval strategy"
        )
        assert not answer.failed
        assert answer.system_type is SystemType.AGENTIC_RAG
        assert answer.answer
        assert answer.trace is not None
        assert answer.trace.stop_reason in (
            StopReason.EVIDENCE_SUFFICIENT,
            StopReason.MAX_ROUNDS_REACHED,
        )
        assert answer.trace.subquestions
        node_order = [step.node.value for step in answer.trace.steps]
        assert node_order[0] == "query_analyzer"
        assert node_order[1] == "planner"
        assert "synthesizer" in node_order
        assert node_order[-1] == "finalizer"

    def test_citations_are_verified(self, processed_dir):
        answer = _agentic(processed_dir).run("t2", "Which papers disagree about reranking?")
        store_ids = {r.chunk.chunk_id for r in answer.retrieved_chunks}
        for citation in answer.citations:
            assert citation.chunk_id in store_ids

    def test_never_exceeds_round_cap(self, processed_dir):
        answer = _agentic(processed_dir).run(
            "t3", "Write a literature review on adaptive retrieval and reranking and evaluation"
        )
        assert answer.retrieval_rounds <= 3

    def test_provider_failure_produces_failed_answer(self, processed_dir):
        class FailingLLM:
            model_name = "failing"

            def complete(self, prompt, *, purpose, max_tokens=1024, temperature=0.0):
                raise LLMError("boom")

        answer = _agentic(processed_dir, llm=FailingLLM()).run("t4", "What is RAPTOR?")
        assert answer.failed
        assert answer.trace is not None
        assert answer.trace.stop_reason is StopReason.ERROR

    def test_token_budget_stops_the_loop(self, processed_dir):
        config = load_system_config(
            CONFIGS / "agentic_rag.yaml", defaults_path=CONFIGS / "default.yaml"
        )
        tiny_budget = config.model_copy(deep=True)
        assert tiny_budget.agent is not None
        tiny_budget.agent.max_total_tokens = 1
        retriever = Retriever.from_processed_dir(
            processed_dir, build_embedder(config.embedding), config.retrieval
        )
        pipeline = AgenticRAGPipeline(tiny_budget, retriever, MockLLMProvider())
        answer = pipeline.run("t5", "Compare GraphRAG and RAPTOR")
        assert answer.trace is not None
        assert answer.trace.stop_reason is StopReason.TOKEN_BUDGET_EXHAUSTED

    def test_timeout_stops_the_loop(self, processed_dir):
        class SlowLLM(MockLLMProvider):
            def complete(self, prompt, *, purpose, max_tokens=1024, temperature=0.0):
                time.sleep(0.05)
                return super().complete(
                    prompt, purpose=purpose, max_tokens=max_tokens, temperature=temperature
                )

        config = load_system_config(
            CONFIGS / "agentic_rag.yaml", defaults_path=CONFIGS / "default.yaml"
        )
        hurried = config.model_copy(deep=True)
        assert hurried.agent is not None
        hurried.agent.timeout_seconds = 0.01
        retriever = Retriever.from_processed_dir(
            processed_dir, build_embedder(config.embedding), config.retrieval
        )
        pipeline = AgenticRAGPipeline(hurried, retriever, SlowLLM())
        answer = pipeline.run("t6", "Compare GraphRAG and RAPTOR")
        assert answer.trace is not None
        assert answer.trace.stop_reason is StopReason.TIMEOUT

    def test_retries_then_succeeds(self, processed_dir):
        class FlakyLLM(MockLLMProvider):
            def __init__(self):
                self.calls = 0

            def complete(self, prompt, *, purpose, max_tokens=1024, temperature=0.0):
                self.calls += 1
                if self.calls == 1:
                    raise LLMError("transient")
                return super().complete(
                    prompt, purpose=purpose, max_tokens=max_tokens, temperature=temperature
                )

        answer = _agentic(processed_dir, llm=FlakyLLM()).run("t7", "What is RAPTOR?")
        assert not answer.failed


class TestFairness:
    def test_same_retriever_components_for_both_systems(self, processed_dir):
        """Both pipelines share retrieval infrastructure — same store, same index."""
        standard = _standard(processed_dir)
        agentic = _agentic(processed_dir)
        standard_answer = standard.run("f1", "What is the main contribution of RAPTOR?")
        agentic_answer = agentic.run("f1", "What is the main contribution of RAPTOR?")
        standard_docs = {r.chunk.document_id for r in standard_answer.retrieved_chunks}
        agentic_docs = {r.chunk.document_id for r in agentic_answer.retrieved_chunks}
        assert "raptor_overview" in standard_docs
        assert "raptor_overview" in agentic_docs
