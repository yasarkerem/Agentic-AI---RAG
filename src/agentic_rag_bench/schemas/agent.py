"""Agent workflow traces: every step the agentic pipeline takes is recorded."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag_bench.schemas.usage import TokenUsage


class AgentNode(StrEnum):
    """Nodes of the agentic state machine (brief §8)."""

    QUERY_ANALYZER = "query_analyzer"
    PLANNER = "planner"
    RETRIEVER = "retriever"
    EVIDENCE_ASSESSOR = "evidence_assessor"
    QUERY_REWRITER = "query_rewriter"
    CONTRADICTION_CHECKER = "contradiction_checker"
    SYNTHESIZER = "synthesizer"
    CITATION_VERIFIER = "citation_verifier"
    FINALIZER = "finalizer"


class StopReason(StrEnum):
    """Why the agent stopped iterating. Exactly one applies per run."""

    EVIDENCE_SUFFICIENT = "evidence_sufficient"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"
    COST_BUDGET_EXHAUSTED = "cost_budget_exhausted"
    TIMEOUT = "timeout"
    ERROR = "error"


class AgentStep(BaseModel):
    """One executed node, with enough detail to replay the run."""

    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(ge=0)
    node: AgentNode
    round_index: int = Field(ge=0, description="Retrieval round this step belongs to")
    summary: str = Field(min_length=1, description="Human-readable account of what happened")
    input_snapshot: dict[str, object] = Field(default_factory=dict)
    output_snapshot: dict[str, object] = Field(default_factory=dict)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_seconds: float = Field(ge=0, default=0.0)
    error: str | None = None


class AgentTrace(BaseModel):
    """The full trace of one agentic run."""

    model_config = ConfigDict(extra="forbid")

    steps: list[AgentStep] = Field(default_factory=list)
    stop_reason: StopReason | None = None
    retrieval_rounds: int = Field(default=0, ge=0)
    subquestions: list[str] = Field(default_factory=list)

    @property
    def total_usage(self) -> TokenUsage:
        total = TokenUsage()
        for step in self.steps:
            total = total + step.usage
        return total

    @property
    def llm_call_count(self) -> int:
        return sum(1 for step in self.steps if step.usage.total_tokens > 0)
