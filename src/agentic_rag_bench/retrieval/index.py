"""In-memory cosine vector index with disk persistence.

One implementation path that runs everywhere (tests, CI, laptops) — a
service-backed index (e.g. Qdrant) can implement the same interface later.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class VectorIndexError(Exception):
    """Raised on inconsistent index contents or missing persistence files."""


class InMemoryVectorIndex:
    """Exact cosine-similarity search over normalized vectors."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._matrix: NDArray[np.float64] | None = None

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        if len(chunk_ids) != len(vectors):
            raise VectorIndexError(
                f"{len(chunk_ids)} ids but {len(vectors)} vectors — inputs out of sync"
            )
        if not chunk_ids:
            return
        duplicate = set(chunk_ids) & set(self._ids)
        if duplicate:
            raise VectorIndexError(f"Duplicate chunk IDs added to index: {sorted(duplicate)[:5]}")
        block = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        block = block / norms
        self._matrix = block if self._matrix is None else np.vstack([self._matrix, block])
        self._ids.extend(chunk_ids)

    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """Return up to ``top_k`` (chunk_id, cosine score) pairs, best first."""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if self._matrix is None or not self._ids:
            return []
        query = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        query = query / norm
        scores = self._matrix @ query
        order = np.argsort(-scores)[:top_k]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in order]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = self._matrix if self._matrix is not None else np.zeros((0, 0))
        np.savez_compressed(path, ids=np.array(self._ids, dtype=np.str_), vectors=matrix)

    @classmethod
    def load(cls, path: Path) -> "InMemoryVectorIndex":
        if not path.is_file():
            raise VectorIndexError(f"Index file not found: {path}")
        data = np.load(path, allow_pickle=False)
        index = cls()
        ids = [str(i) for i in data["ids"].tolist()]
        matrix = data["vectors"]
        if len(ids) != matrix.shape[0]:
            raise VectorIndexError(f"Corrupt index {path}: {len(ids)} ids, {matrix.shape[0]} rows")
        if ids:
            index._ids = ids
            index._matrix = matrix.astype(np.float64)
        return index
