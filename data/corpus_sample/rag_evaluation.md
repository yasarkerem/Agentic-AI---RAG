---
title: "Evaluating RAG Systems: Metrics, Judges, and Pitfalls"
year: 2024
tags: [evaluation, llm-judge, metrics, benchmarks]
---

# Evaluating RAG Systems: Metrics, Judges, and Pitfalls

## Two layers to evaluate

RAG evaluation decomposes into retrieval quality and answer quality, and
conflating them hides the cause of failures. Retrieval is scored with
classical ranking metrics — recall and precision at a cutoff, mean
reciprocal rank, and normalized discounted cumulative gain — against gold
evidence labels. Answer quality covers correctness against a reference,
completeness of coverage, groundedness in the retrieved evidence, and
citation accuracy. A system can retrieve perfectly and still answer badly,
or answer plausibly from bad retrieval; only separated metrics distinguish
the two.

## LLM-as-judge

Using a strong model to grade answers scales far beyond human annotation
and correlates reasonably with human judgments on many tasks. But judge
models carry documented biases: they prefer longer answers, favor fluent
phrasing over factual precision, rate outputs whose style resembles their
own more highly, and are sensitive to the order in which candidates are
presented. Reported best practice is to treat the judge as one evaluator
among several — pair it with deterministic checks and gold labels, pin the
judge model and prompt version so scores are comparable across runs, and
audit a sample of judgments by hand.

## Pitfalls

Recurring evaluation mistakes include: measuring only answer quality and
never retrieval, so retrieval regressions masquerade as generation issues;
contaminated benchmarks whose answers the generator memorized during
pretraining, inflating closed-book performance and understating retrieval's
value; single-run comparisons of nondeterministic systems without variance
estimates; and unpaired comparisons across systems on different task
samples. Paired designs — every system answering the same tasks — with
statistical tests over the pairs are the reported remedy.

## Recommendation

Version everything the evaluation depends on: benchmark tasks, gold labels,
judge prompts, and judge model identifiers. An evaluation whose components
silently change cannot support longitudinal claims about system progress.
