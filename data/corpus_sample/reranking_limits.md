---
title: "When Reranking Does Not Help: Strong Retrievers and Domain Shift"
year: 2024
tags: [reranking, cross-encoder, domain-shift, negative-results]
---

# When Reranking Does Not Help: Strong Retrievers and Domain Shift

## Setting

This evaluation revisits cross-encoder reranking under different conditions
from earlier precision studies: the first-stage retriever is a modern,
strongly trained dense model already tuned to the target corpus; queries
are long, multi-faceted research questions rather than short factoids; and
evaluation emphasizes recall and source coverage at larger cutoffs, on
specialized scientific text far from the reranker's training distribution.

## Findings

Under these conditions the measured gains from reranking were marginal, and
in several configurations negative. When the first-stage dense retriever
was already strong, the candidate list was clean enough that reordering it
changed little. On out-of-domain scientific text the cross-encoder — trained
on general web question-answering pairs — systematically preferred
superficially fluent passages over technically precise ones, pushing the
correct specialized evidence below the cutoff. Reranking also reduced the
diversity of retrieved documents by concentrating the top ranks on one
source, which hurt multi-document synthesis tasks that need broad coverage.

## Explaining the disagreement with earlier studies

The disagreement with earlier positive results is largely explained by
experimental setup rather than contradiction in the underlying method:
first-stage strength (weak BM25 versus a tuned dense retriever), query type
(factoid versus multi-faceted research questions), evaluation metric
(precision at small cutoffs versus recall and coverage at larger ones), and
domain match between the reranker's training data and the corpus. Reranking
helps when the candidate pool is noisy and the metric rewards top-rank
precision; it stops paying its latency cost when those conditions do not
hold.

## Recommendation

Treat reranking as a conditional component, not a default: validate it on
the target corpus and metric before enabling it, and prefer domain-matched
rerankers for specialized text.
