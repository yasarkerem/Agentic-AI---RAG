"""Experiment records: everything needed to reproduce and analyze one run (brief §12)."""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag_bench.schemas.answers import PipelineAnswer, SystemType
from agentic_rag_bench.schemas.evaluation import EvaluationScores


def utc_now() -> datetime:
    """Timezone-aware timestamp used everywhere results are stamped."""
    return datetime.now(tz=UTC)


class ExperimentMetadata(BaseModel):
    """Identity and provenance of one experiment run."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(min_length=1)
    system_type: SystemType
    config_name: str = Field(min_length=1, description="Config file the run was launched with")
    config_snapshot: dict[str, object] = Field(
        description="Fully-resolved configuration at launch time"
    )
    git_commit: str | None = Field(default=None, description="Repository commit hash at run time")
    benchmark_version: str = Field(min_length=1)
    model_names: dict[str, str] = Field(
        default_factory=dict, description="Role -> model, e.g. {'generation': '...'}"
    )
    prompt_versions: dict[str, str] = Field(
        default_factory=dict, description="Prompt name -> version identifier"
    )
    seed: int = 42
    started_at: datetime = Field(default_factory=utc_now)


class TaskResult(BaseModel):
    """One (task, system) outcome: a row in the results JSONL file."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task_level: int = Field(ge=1, le=5)
    answer: PipelineAnswer
    scores: EvaluationScores | None = Field(
        default=None, description="Filled by the evaluation stage; None until evaluated"
    )
    completed_at: datetime = Field(default_factory=utc_now)
