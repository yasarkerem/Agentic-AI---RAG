---
title: "Hybrid Retrieval: Combining Sparse and Dense Signals"
year: 2023
tags: [hybrid-retrieval, bm25, dense-retrieval, fusion]
---

# Hybrid Retrieval: Combining Sparse and Dense Signals

## Why combine sparse and dense retrieval

Sparse lexical retrieval such as BM25 scores documents by exact term
overlap, weighted by term rarity and document length. It is precise for
rare keywords, identifiers, and names, but blind to synonyms and
paraphrase. Dense retrieval embeds queries and passages into a shared
vector space, capturing semantic similarity even without word overlap, but
it can miss exact rare terms and drifts under domain shift when the
embedding model was trained on different text. Hybrid retrieval runs both
and merges their rankings, keeping the strengths of each signal.

## Fusion methods

Two fusion families dominate. Score-based fusion normalizes each system's
scores — for example min-max normalization — and combines them with a
weighted sum, where the weight controls the dense-versus-sparse balance.
Rank-based fusion, most commonly reciprocal rank fusion (RRF), ignores raw
scores and combines reciprocal ranks, which is robust when score scales are
incomparable. Score fusion is more tunable; rank fusion needs fewer
assumptions and is harder to misconfigure.

## When hybrid helps

Hybrid retrieval helps most when queries mix semantic intent with specific
terminology — a question about "tree-structured summarization indexes" plus
the exact name "RAPTOR" is served by both signals at once. Technical corpora
with dense jargon, acronyms, and method names benefit consistently. When
queries are purely conversational and vocabulary overlap is low, dense
retrieval alone often suffices, and the sparse leg adds little.

## Practical notes

The dense weight is a real hyperparameter: values around 0.5 to 0.7 are
common starting points for technical text, but the best setting depends on
the corpus and should be validated on held-out queries rather than assumed.
Hybrid retrieval also increases recall of unique relevant documents, which
matters for multi-document synthesis tasks where covering all sources is as
important as ranking the best one first.
