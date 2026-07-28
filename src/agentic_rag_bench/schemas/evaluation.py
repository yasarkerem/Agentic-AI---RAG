"""Evaluation metric containers (brief §11). Metric computation arrives in Phase 5."""

from pydantic import BaseModel, ConfigDict, Field


class RetrievalMetrics(BaseModel):
    """Quality of what was retrieved, against source-level gold labels."""

    model_config = ConfigDict(extra="forbid")

    recall_at_k: float = Field(ge=0, le=1)
    precision_at_k: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg: float = Field(ge=0, le=1)
    source_coverage: float = Field(
        ge=0, le=1, description="Fraction of gold documents represented in retrieved chunks"
    )
    unique_relevant_documents: int = Field(ge=0)
    k: int = Field(gt=0, description="The k these @k metrics were computed at")


class AnswerMetrics(BaseModel):
    """Quality of the final answer. Scores are normalized to [0, 1]."""

    model_config = ConfigDict(extra="forbid")

    factual_correctness: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    citation_correctness: float = Field(ge=0, le=1)
    citation_completeness: float = Field(ge=0, le=1)
    groundedness: float = Field(ge=0, le=1)
    hallucination_rate: float = Field(ge=0, le=1)
    contradiction_handling: float | None = Field(
        default=None, ge=0, le=1, description="Only scored for Level 4+ tasks"
    )
    structure_readability: float = Field(ge=0, le=1)


class SystemMetrics(BaseModel):
    """Operational cost of producing the answer."""

    model_config = ConfigDict(extra="forbid")

    latency_seconds: float = Field(ge=0)
    retrieval_rounds: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    failed: bool = False


class EvaluatorInfo(BaseModel):
    """Which evaluator produced a score set — judges are versioned (brief §11)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: str = Field(min_length=1, description="'deterministic', 'llm_judge', or 'human'")
    version: str = Field(min_length=1)
    judge_model: str | None = None
    prompt_version: str | None = None


class EvaluationScores(BaseModel):
    """All scores for one (task, system) pair, with evaluator provenance."""

    model_config = ConfigDict(extra="forbid")

    retrieval: RetrievalMetrics | None = None
    answer: AnswerMetrics | None = None
    system: SystemMetrics | None = None
    evaluators: list[EvaluatorInfo] = Field(default_factory=list)
