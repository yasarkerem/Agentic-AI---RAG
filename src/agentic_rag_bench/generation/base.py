"""LLM provider abstraction.

Providers implement one method: ``complete``. Pipelines communicate intent via
``purpose`` (e.g. 'answer', 'plan', 'assess'), which lets the deterministic
mock provider produce sensible structured output without a real model.

Prompts carry a machine-readable payload between ``<payload>`` markers in
addition to human-readable instructions: real models read the whole prompt,
the mock parses just the payload. This keeps tests deterministic and free.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from agentic_rag_bench.schemas import TokenUsage

PAYLOAD_RE = re.compile(r"<payload>\s*(\{.*?\})\s*</payload>", re.DOTALL)


class LLMError(Exception):
    """Raised when a provider fails (network, auth, timeout, bad response)."""


@dataclass(frozen=True)
class LLMResponse:
    """One completion: text plus accounting."""

    text: str
    usage: TokenUsage
    model: str
    latency_seconds: float


class LLMProvider(Protocol):
    """Anything that can complete a prompt."""

    @property
    def model_name(self) -> str: ...

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


def embed_payload(instructions: str, payload: dict[str, Any]) -> str:
    """Build a prompt with a machine-readable payload section."""
    return f"{instructions}\n\n<payload>\n{json.dumps(payload, sort_keys=True)}\n</payload>\n"


def extract_payload(prompt: str) -> dict[str, Any]:
    """Parse the payload section out of a prompt. Returns {} when absent."""
    match = PAYLOAD_RE.search(prompt)
    if match is None:
        return {}
    try:
        loaded = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def approximate_tokens(text: str) -> int:
    """Deterministic token approximation used by the mock provider (words)."""
    return max(1, len(text.split()))
