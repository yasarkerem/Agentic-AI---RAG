"""Hybrid result fusion: min-max normalized weighted sum of dense and BM25."""


def _normalize(results: list[tuple[str, float]]) -> dict[str, float]:
    if not results:
        return {}
    scores = [score for _, score in results]
    low, high = min(scores), max(scores)
    if high == low:
        return {doc_id: 1.0 for doc_id, _ in results}
    return {doc_id: (score - low) / (high - low) for doc_id, score in results}


def fuse(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    dense_weight: float,
    top_k: int,
) -> list[tuple[str, float]]:
    """Weighted-sum fusion over min-max normalized score lists.

    IDs present in only one list contribute only that list's weighted score.
    Ties break lexicographically for determinism.
    """
    if not 0.0 <= dense_weight <= 1.0:
        raise ValueError(f"dense_weight must be in [0, 1], got {dense_weight}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    dense_norm = _normalize(dense)
    sparse_norm = _normalize(sparse)
    fused: dict[str, float] = {}
    for chunk_id, score in dense_norm.items():
        fused[chunk_id] = fused.get(chunk_id, 0.0) + dense_weight * score
    for chunk_id, score in sparse_norm.items():
        fused[chunk_id] = fused.get(chunk_id, 0.0) + (1.0 - dense_weight) * score
    ranked = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
    return ranked[:top_k]
