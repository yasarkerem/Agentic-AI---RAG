---
title: "Cross-Encoder Reranking Substantially Improves RAG Precision"
year: 2023
tags: [reranking, cross-encoder, precision]
---

# Cross-Encoder Reranking Substantially Improves RAG Precision

## Setting

This study evaluates two-stage retrieval on open-domain question answering
benchmarks with short factoid answers, using a moderate-quality first-stage
retriever: BM25 or an early dense bi-encoder retrieving one hundred
candidates, followed by a cross-encoder that scores each query-passage pair
jointly with full attention.

## Findings

Reranking delivered large, consistent gains in this setting. Precision at
five improved by double digits across every benchmark tested, and answer
accuracy of the downstream generator rose accordingly, because the
generator's small context window was filled with genuinely relevant
passages instead of near-miss neighbors. The gains were largest exactly
when the first stage was weakest: noisy candidate pools benefit most from a
scorer that reads the query and passage together.

## Interpretation

The cross-encoder's advantage comes from joint attention over the query and
passage, which resolves ambiguity that independent embeddings cannot. Under
this evaluation — factoid queries, shallow candidate pools, a weak first
stage, and precision-oriented metrics at small cutoffs — reranking is one of
the highest-leverage additions available to a RAG pipeline, and the added
latency of scoring one hundred pairs is modest relative to the accuracy
gained.

## Recommendation

For pipelines whose first-stage retriever is BM25 or an older bi-encoder,
this study recommends always adding a cross-encoder reranker: the precision
improvement translates directly into fewer hallucinated answers, since the
generator sees less irrelevant context.
