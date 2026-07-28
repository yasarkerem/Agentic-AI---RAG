---
title: "Reducing Hallucination Through Grounding and Citation Verification"
year: 2023
tags: [hallucination, groundedness, citations, verification]
---

# Reducing Hallucination Through Grounding and Citation Verification

## The problem

Language models fluently assert content that their evidence does not
support. In retrieval-augmented systems hallucination takes a specific
form: the answer contains claims not present in, or contradicted by, the
retrieved passages. Because the evidence set is explicit, RAG systems can
measure and reduce this failure in ways closed-book generation cannot.

## Grounding techniques

The most effective mitigations reported share one principle: constrain the
answer to the evidence. Instructing the model to answer only from provided
context and to say so when the context is insufficient reduces unsupported
claims substantially, at the cost of more abstentions. Requiring inline
attribution — every claim tied to a specific source passage — further
disciplines generation, because claims without an available citation become
visible. Lowering generation temperature reduces embellishment on factual
tasks.

## Citation verification

Post-hoc verification closes the loop: after generation, each cited
source is checked against the claim it supports. Cheap deterministic checks
catch a surprising share of failures — the cited passage must actually
exist in the retrieved set, and quoted spans must appear verbatim in the
source. Entailment-based checks then test whether the passage actually
supports the claim, catching citations that are real but irrelevant, a
failure mode where the model decorates an unsupported claim with a
plausible-looking reference.

## Measurement

Groundedness is measured as the fraction of answer claims supported by
retrieved evidence, and hallucination rate as its complement. Studies that
report both note that citation-verified pipelines shift errors from
confident fabrication to explicit abstention — a trade worth making in
research and high-stakes settings, where a wrong-but-cited answer is more
damaging than no answer.
