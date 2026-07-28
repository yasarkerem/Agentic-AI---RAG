"""Benchmark tasks: the five complexity levels and their gold evidence."""

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field


class TaskLevel(IntEnum):
    """Complexity levels from the benchmark design (brief §4)."""

    SINGLE_DOC_FACTUAL = 1
    MULTI_DOC_COMPARISON = 2
    MULTI_HOP_SYNTHESIS = 3
    CONTRADICTION_ANALYSIS = 4
    RESEARCH_SYNTHESIS = 5


class GoldEvidence(BaseModel):
    """A source-level gold label: evidence an ideal answer should draw on."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    chunk_ids: list[str] = Field(
        default_factory=list,
        description="Specific gold chunks when known; empty means document-level relevance only",
    )
    note: str | None = Field(default=None, description="Why this evidence matters for the task")


class BenchmarkTask(BaseModel):
    """One benchmark task, stored as a row in the machine-readable benchmark file."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-]*$")
    level: TaskLevel
    question: str = Field(min_length=1)
    reference_answer: str | None = Field(
        default=None, description="Reference answer for deterministic and judged comparison"
    )
    gold_evidence: list[GoldEvidence] = Field(default_factory=list)
    expected_behaviors: list[str] = Field(
        default_factory=list,
        description="Behavioral expectations for this level, e.g. 'cite each major claim'",
    )
    tags: list[str] = Field(default_factory=list)


class BenchmarkManifest(BaseModel):
    """Versioned container for a benchmark release."""

    model_config = ConfigDict(extra="forbid")

    benchmark_version: str = Field(min_length=1)
    tasks: list[BenchmarkTask] = Field(min_length=1)

    def tasks_at_level(self, level: TaskLevel) -> list[BenchmarkTask]:
        return [task for task in self.tasks if task.level is level]
