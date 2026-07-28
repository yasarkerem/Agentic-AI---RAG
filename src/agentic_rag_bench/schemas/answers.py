"""Final pipeline outputs: the answer a system returns for a task."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag_bench.schemas.agent import AgentTrace
from agentic_rag_bench.schemas.citations import Citation
from agentic_rag_bench.schemas.retrieval import RetrievedChunk
from agentic_rag_bench.schemas.usage import LLMCallRecord, TokenUsage


class SystemType(StrEnum):
    """The two systems under comparison."""

    STANDARD_RAG = "standard_rag"
    AGENTIC_RAG = "agentic_rag"


class PipelineAnswer(BaseModel):
    """What either pipeline returns for one query: answer, citations, and accounting."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1)
    system_type: SystemType
    question: str = Field(min_length=1)
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = Field(default=0.0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0)
    retrieval_rounds: int = Field(default=1, ge=0)
    trace: AgentTrace | None = Field(
        default=None, description="Present for agentic runs; None for the standard baseline"
    )
    failed: bool = Field(default=False, description="True when the pipeline errored or timed out")
    failure_reason: str | None = None
