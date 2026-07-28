---
title: "Adaptive Retrieval: Deciding When, What, and How Often to Retrieve"
year: 2024
tags: [adaptive-retrieval, self-reflection, query-rewriting, active-retrieval]
---

# Adaptive Retrieval: Deciding When, What, and How Often to Retrieve

## The idea

Standard RAG retrieves once, unconditionally, with the user's query as the
search string. Adaptive retrieval makes each of those choices dynamic: the
system decides whether retrieval is needed at all, reformulates queries
when initial evidence is weak, and retrieves again mid-generation when the
model reaches a claim it cannot support from current context.

## Major approaches

Three families recur in the literature. Self-reflective methods train or
prompt the model to emit control signals — retrieve or not, is this passage
relevant, is this draft supported — and use those judgments to steer the
pipeline. Active retrieval methods trigger new searches during generation:
when the model's next-sentence confidence drops or a forward-looking draft
sentence contains low-probability content, that sentence itself becomes the
new query. Query transformation methods rewrite the search string —
decomposing multi-part questions into subquestions, expanding acronyms, or
generating a hypothetical answer passage and retrieving neighbors of that
instead of the raw question.

## Evidence

Across reported evaluations, adaptive methods most improve multi-hop and
knowledge-intensive tasks, where single-shot retrieval predictably fails to
gather all needed evidence. On single-fact questions the gains shrink and
the added model calls cost latency; retrieving when unnecessary can even
inject distracting context that lowers accuracy.

## Limitations and open gaps

Reported limitations include the reliability of self-assessment — models
are imperfect judges of their own evidence sufficiency — and evaluation
practices that vary widely across papers, making methods hard to compare.
Open research gaps include principled stopping criteria for iterative
retrieval (most systems use fixed round caps), cost-aware policies that
weigh another retrieval round against its expected benefit, and benchmarks
that measure the retrieval decisions themselves rather than only final
answer accuracy.
