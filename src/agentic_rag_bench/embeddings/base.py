"""Embedding provider abstraction."""

import hashlib
import math
import re
from typing import Protocol

from agentic_rag_bench.config import ConfigError, EmbeddingConfig

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimal English stopword list: keeps question boilerplate ("what is the
# main ... of") from drowning out content terms in bag-of-words vectors.
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have how in into is it its of on or "  # noqa: SIM905
    "such that the their there these this to was were what when where which who why "
    "will with".split()
)


class Embedder(Protocol):
    """Anything that can turn texts into fixed-size vectors."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class HashedEmbedder:
    """Deterministic feature-hashing embedder — free, offline, reproducible.

    Tokens are hashed into a fixed number of buckets with a hash-derived
    sign, then L2-normalized. Texts sharing vocabulary get similar vectors,
    which is enough for development, tests, and mock experiments. It is NOT
    a semantic embedder; real experiments should plug a real provider in
    behind the same interface.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return f"hashed-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _TOKEN_RE.findall(text.lower()):
            if token in _STOPWORDS:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]


def build_embedder(config: EmbeddingConfig) -> Embedder:
    """Instantiate the embedder named by the configuration."""
    if config.provider in ("mock", "hashed"):
        return HashedEmbedder()
    raise ConfigError(
        f"Unknown embedding provider {config.provider!r} — available: 'mock'/'hashed'. "
        "Real providers are added behind the Embedder protocol."
    )
