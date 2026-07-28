"""Schema invariants: IDs, citation shape, task levels, traces, serialization."""

import json

import pytest
from pydantic import ValidationError

from agentic_rag_bench.schemas import (
    AgentNode,
    AgentStep,
    AgentTrace,
    BenchmarkManifest,
    BenchmarkTask,
    Chunk,
    Citation,
    PipelineAnswer,
    StopReason,
    SystemType,
    TaskLevel,
    TokenUsage,
    content_hash,
    make_chunk_id,
)


class TestChunkIds:
    def test_matches_brief_example_format(self):
        assert make_chunk_id("paper_001", 4, 2) == "paper_001_p4_c2"

    def test_deterministic(self):
        assert make_chunk_id("doc", 1, 0) == make_chunk_id("doc", 1, 0)

    def test_no_page_maps_to_page_zero(self):
        assert make_chunk_id("readme_doc", None, 3) == "readme_doc_p0_c3"

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError, match="chunk_index"):
            make_chunk_id("doc", 1, -1)

    def test_negative_page_rejected(self):
        with pytest.raises(ValueError, match="page"):
            make_chunk_id("doc", -2, 0)


class TestContentHash:
    def test_stable_across_calls(self):
        assert content_hash("same text") == content_hash("same text")

    def test_different_text_different_hash(self):
        assert content_hash("a") != content_hash("b")


class TestChunk:
    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            Chunk(
                chunk_id="d_p0_c0",
                document_id="d",
                document_title="T",
                text="",
                source_path="data/raw/d.pdf",
                chunk_index=0,
            )

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            Chunk(
                chunk_id="d_p0_c0",
                document_id="d",
                document_title="T",
                text="body",
                source_path="data/raw/d.pdf",
                chunk_index=0,
                embedding=[0.1],  # type: ignore[call-arg]  # not a Chunk field, must be rejected
            )


class TestCitation:
    def test_serializes_to_brief_shape(self):
        """The JSON shape must match brief §10 exactly."""
        citation = Citation(
            document_id="paper_001",
            title="Example Paper",
            page=4,
            chunk_id="paper_001_p4_c2",
            quoted_evidence="Relevant supporting passage",
            retrieval_score=0.84,
        )
        payload = json.loads(citation.model_dump_json())
        assert payload == {
            "document_id": "paper_001",
            "title": "Example Paper",
            "page": 4,
            "chunk_id": "paper_001_p4_c2",
            "quoted_evidence": "Relevant supporting passage",
            "retrieval_score": 0.84,
        }

    def test_empty_evidence_rejected(self):
        with pytest.raises(ValidationError):
            Citation(
                document_id="paper_001",
                title="Example Paper",
                page=4,
                chunk_id="paper_001_p4_c2",
                quoted_evidence="",
                retrieval_score=0.84,
            )


class TestTasks:
    def test_all_five_levels_exist(self):
        assert [level.value for level in TaskLevel] == [1, 2, 3, 4, 5]

    def test_level_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            BenchmarkTask(task_id="t1", level=6, question="q?")  # type: ignore[arg-type]

    def test_manifest_filters_by_level(self):
        manifest = BenchmarkManifest(
            benchmark_version="0.1.0",
            tasks=[
                BenchmarkTask(task_id="t1", level=TaskLevel.SINGLE_DOC_FACTUAL, question="a?"),
                BenchmarkTask(task_id="t2", level=TaskLevel.RESEARCH_SYNTHESIS, question="b?"),
            ],
        )
        level_five = manifest.tasks_at_level(TaskLevel.RESEARCH_SYNTHESIS)
        assert [task.task_id for task in level_five] == ["t2"]

    def test_manifest_requires_tasks(self):
        with pytest.raises(ValidationError):
            BenchmarkManifest(benchmark_version="0.1.0", tasks=[])


class TestAgentTrace:
    def _step(self, index: int, node: AgentNode, in_tokens: int = 0) -> AgentStep:
        return AgentStep(
            step_index=index,
            node=node,
            round_index=0,
            summary=f"{node.value} ran",
            usage=TokenUsage(input_tokens=in_tokens, output_tokens=in_tokens // 2),
        )

    def test_total_usage_accumulates(self):
        trace = AgentTrace(
            steps=[
                self._step(0, AgentNode.PLANNER, in_tokens=100),
                self._step(1, AgentNode.RETRIEVER, in_tokens=0),
                self._step(2, AgentNode.SYNTHESIZER, in_tokens=200),
            ],
            stop_reason=StopReason.EVIDENCE_SUFFICIENT,
        )
        assert trace.total_usage.input_tokens == 300
        assert trace.total_usage.output_tokens == 150

    def test_llm_call_count_ignores_zero_usage_steps(self):
        trace = AgentTrace(
            steps=[
                self._step(0, AgentNode.PLANNER, in_tokens=100),
                self._step(1, AgentNode.RETRIEVER, in_tokens=0),
            ]
        )
        assert trace.llm_call_count == 1

    def test_every_brief_node_is_modeled(self):
        expected = {
            "query_analyzer",
            "planner",
            "retriever",
            "evidence_assessor",
            "query_rewriter",
            "contradiction_checker",
            "synthesizer",
            "citation_verifier",
            "finalizer",
        }
        assert {node.value for node in AgentNode} == expected


class TestPipelineAnswer:
    def test_round_trips_through_json(self):
        answer = PipelineAnswer(
            query_id="t1",
            system_type=SystemType.STANDARD_RAG,
            question="What is RAPTOR?",
            answer="RAPTOR builds a hierarchical retrieval tree.",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        restored = PipelineAnswer.model_validate_json(answer.model_dump_json())
        assert restored == answer

    def test_standard_run_has_no_trace_by_default(self):
        answer = PipelineAnswer(
            query_id="t1",
            system_type=SystemType.STANDARD_RAG,
            question="q?",
            answer="a",
        )
        assert answer.trace is None
        assert answer.failed is False
