"""Anthropic Messages API provider (opt-in; never used by tests).

Only constructed when the experiment config selects ``provider: anthropic``
AND an API key is present — nothing in this repository spends money unless
the user explicitly configures both.
"""

import time

from agentic_rag_bench.generation.base import LLMError, LLMResponse
from agentic_rag_bench.observability import get_logger
from agentic_rag_bench.schemas import TokenUsage

logger = get_logger("anthropic_provider")

# Claude 4.7+ / 5-family models reject sampling parameters (400).
_NO_SAMPLING_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)


class AnthropicProvider:
    """Thin wrapper over the official ``anthropic`` SDK."""

    def __init__(self, model: str, api_key: str) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - anthropic is a declared dependency
            raise LLMError("The 'anthropic' package is not installed") from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        started = time.perf_counter()
        messages = [{"role": "user", "content": prompt}]
        use_temperature = temperature != 0.0 and not self._model.startswith(_NO_SAMPLING_PREFIXES)
        try:
            if use_temperature:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                )
            else:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    messages=messages,  # type: ignore[arg-type]
                )
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError(f"Anthropic connection error: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("Anthropic model refused the request (stop_reason=refusal)")

        text = "".join(block.text for block in response.content if block.type == "text")
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        latency = time.perf_counter() - started
        logger.info(
            "llm_call",
            model=self._model,
            purpose=purpose,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency=round(latency, 3),
        )
        return LLMResponse(text=text, usage=usage, model=response.model, latency_seconds=latency)
