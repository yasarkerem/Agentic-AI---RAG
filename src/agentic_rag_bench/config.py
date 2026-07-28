"""Typed configuration: environment settings and layered YAML system configs.

Two kinds of configuration exist:

- :class:`AppSettings` — runtime environment (paths, logging, API keys),
  loaded from environment variables / ``.env`` with the ``ARB_`` prefix.
- :class:`SystemConfig` — the parameters of a RAG system under test, loaded
  from YAML. ``configs/default.yaml`` provides shared defaults and a
  system-specific file (e.g. ``configs/agentic_rag.yaml``) overlays it.

Secrets never appear in system configs; they live only in the environment.
"""

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_rag_bench.schemas.answers import SystemType


class ConfigError(Exception):
    """Raised when a configuration file is missing, unparsable, or invalid."""


class AppSettings(BaseSettings):
    """Process-level settings from the environment (never from experiment YAML)."""

    model_config = SettingsConfigDict(
        env_prefix="ARB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    log_level: str = "INFO"
    log_format: str = Field(default="console", description="'console' or 'json'")
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")


class ChunkingConfig(BaseModel):
    """Chunking parameters (sizes are in tokens of the embedding tokenizer)."""

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)

    @model_validator(mode="after")
    def overlap_smaller_than_size(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class EmbeddingConfig(BaseModel):
    """Embedding provider — the same for every compared system (fairness control)."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(default="mock", description="'mock', 'sentence_transformers', or API")
    model: str = Field(default="mock-embedder")


class RetrievalConfig(BaseModel):
    """Dense / sparse / hybrid retrieval parameters."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=8, gt=0)
    use_bm25: bool = Field(default=False, description="Enable sparse retrieval + hybrid fusion")
    dense_weight: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Hybrid fusion weight for dense scores; BM25 weight is 1 - dense_weight. "
        "Must stay 1.0 while use_bm25 is false.",
    )

    @model_validator(mode="after")
    def hybrid_weight_requires_bm25(self) -> Self:
        if not self.use_bm25 and self.dense_weight != 1.0:
            raise ValueError(
                "dense_weight must be 1.0 when use_bm25 is false (there is nothing to fuse with)"
            )
        return self


class RerankerConfig(BaseModel):
    """Optional cross-encoder reranking."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: str = Field(default="mock-reranker")
    top_n: int = Field(default=5, gt=0)


class GenerationConfig(BaseModel):
    """Final answer generation — identical across systems (fairness control)."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(default="mock", description="'mock' or an API provider name")
    model: str = Field(default="mock-llm")
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_output_tokens: int = Field(default=2048, gt=0)
    context_token_limit: int = Field(default=8192, gt=0)


class AgentConfig(BaseModel):
    """Loop controls for the agentic pipeline. Every bound is a hard stop (brief §8)."""

    model_config = ConfigDict(extra="forbid")

    max_retrieval_rounds: int = Field(default=3, ge=1)
    max_subquestions: int = Field(default=6, ge=1)
    max_total_tokens: int = Field(default=100_000, gt=0)
    max_cost_usd: float = Field(default=1.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    timeout_seconds: float = Field(default=300.0, gt=0)


class SystemConfig(BaseModel):
    """Full configuration of one system under test."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    system_type: SystemType
    seed: int = 42
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    agent: AgentConfig | None = Field(
        default=None, description="Required for agentic systems; must be absent for standard RAG"
    )

    @model_validator(mode="after")
    def agent_config_matches_system_type(self) -> Self:
        if self.system_type is SystemType.AGENTIC_RAG and self.agent is None:
            raise ValueError("agentic_rag systems require an 'agent' section")
        if self.system_type is SystemType.STANDARD_RAG and self.agent is not None:
            raise ValueError("standard_rag systems must not define an 'agent' section")
        return self


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(loaded).__name__}")
    return loaded


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_system_config(path: Path, defaults_path: Path | None = None) -> SystemConfig:
    """Load a system config, overlaying ``path`` on top of optional shared defaults.

    Raises :class:`ConfigError` for missing files, invalid YAML, unknown keys,
    and out-of-range values — configuration problems must fail loudly.
    """
    data = _read_yaml(path)
    if defaults_path is not None:
        data = deep_merge(_read_yaml(defaults_path), data)
    try:
        return SystemConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {path}: {exc}") from exc
