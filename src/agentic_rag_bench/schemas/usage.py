"""Token usage and cost accounting shared by both pipelines."""

from pydantic import BaseModel, ConfigDict, Field


class TokenUsage(BaseModel):
    """Input/output token counts, addable so calls can be accumulated."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class ModelPricing(BaseModel):
    """USD price per one million tokens for a given model."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    input_usd_per_mtok: float = Field(ge=0)
    output_usd_per_mtok: float = Field(ge=0)


def calculate_cost(usage: TokenUsage, pricing: ModelPricing) -> float:
    """Exact cost in USD for one usage record under one pricing table."""
    return (
        usage.input_tokens * pricing.input_usd_per_mtok
        + usage.output_tokens * pricing.output_usd_per_mtok
    ) / 1_000_000


class LLMCallRecord(BaseModel):
    """One model call, as recorded in traces and experiment results."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    purpose: str = Field(min_length=1, description="e.g. 'planner', 'synthesis', 'judge'")
    usage: TokenUsage
    latency_seconds: float = Field(ge=0)
    cost_usd: float = Field(ge=0)
