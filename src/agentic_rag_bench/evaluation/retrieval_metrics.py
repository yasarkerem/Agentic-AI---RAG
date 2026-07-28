"""Retrieval quality metrics against document-level gold labels (brief §11).

Relevance is binary at the document level: a retrieved chunk is relevant iff
its document appears in the task's gold evidence. Chunk-level gold labels
(when present) refine nothing here yet — documented limitation until the
benchmark carries chunk-level annotations.
"""

import math

from agentic_rag_bench.schemas import GoldEvidence, RetrievalMetrics, RetrievedChunk


def compute_retrieval_metrics(
    retrieved: list[RetrievedChunk], gold: list[GoldEvidence], k: int
) -> RetrievalMetrics | None:
    """Metrics at cutoff ``k``. Returns None when the task has no gold labels."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    gold_docs = {g.document_id for g in gold}
    if not gold_docs:
        return None

    ranked_docs = [r.chunk.document_id for r in sorted(retrieved, key=lambda r: r.rank)]
    top_k = ranked_docs[:k]

    hits_in_top_k = {doc for doc in top_k if doc in gold_docs}
    recall = len(hits_in_top_k) / len(gold_docs)
    precision = sum(1 for doc in top_k if doc in gold_docs) / k

    reciprocal_rank = 0.0
    for rank, doc in enumerate(top_k, start=1):
        if doc in gold_docs:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, doc in enumerate(top_k, start=1) if doc in gold_docs
    )
    ideal_hits = min(len(gold_docs), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    all_docs = set(ranked_docs)
    covered = gold_docs & all_docs
    return RetrievalMetrics(
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=reciprocal_rank,
        ndcg=min(1.0, ndcg),
        source_coverage=len(covered) / len(gold_docs),
        unique_relevant_documents=len(covered),
        k=k,
    )
