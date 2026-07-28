"""LLM answer generation and provider abstraction."""

from agentic_rag_bench.config import AppSettings, ConfigError, GenerationConfig
from agentic_rag_bench.generation.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    approximate_tokens,
    embed_payload,
    extract_payload,
)
from agentic_rag_bench.generation.mock import MockLLMProvider
from agentic_rag_bench.generation.pricing import PRICING_VERSION, estimate_cost, pricing_for


def build_llm_provider(
    config: GenerationConfig, settings: AppSettings | None = None
) -> LLMProvider:
    """Instantiate the provider named by the configuration.

    'mock' needs nothing. 'anthropic' requires ANTHROPIC_API_KEY — refused
    loudly otherwise so experiments never half-run.
    """
    if config.provider == "mock":
        return MockLLMProvider()
    if config.provider == "anthropic":
        from agentic_rag_bench.generation.anthropic_provider import AnthropicProvider

        resolved = settings if settings is not None else AppSettings()
        if resolved.anthropic_api_key is None:
            raise ConfigError("provider 'anthropic' requires ANTHROPIC_API_KEY in the environment")
        return AnthropicProvider(
            model=config.model, api_key=resolved.anthropic_api_key.get_secret_value()
        )
    raise ConfigError(f"Unknown LLM provider {config.provider!r} — available: 'mock', 'anthropic'")


__all__ = [
    "PRICING_VERSION",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "MockLLMProvider",
    "approximate_tokens",
    "build_llm_provider",
    "embed_payload",
    "estimate_cost",
    "extract_payload",
    "pricing_for",
]
