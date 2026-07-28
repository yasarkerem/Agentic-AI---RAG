# Experiment design

> **Status:** implemented (Phases 5–7). Benchmark v0.1 is committed; the
> matrix, persistence, statistics, and reporting all run end to end on the
> sample corpus with mock providers.

## Benchmark construction

- Corpus: open AI/RAG research papers and technical documents (RAG, agentic
  RAG, GraphRAG, RAPTOR, reranking, hybrid retrieval, hallucination
  reduction, long-context, RAG evaluation, adaptive retrieval, multimodal
  RAG). No datasets that cannot be redistributed.
- ≥ 25 tasks, ≥ 5 per level, stored as versioned JSONL
  (`schemas.BenchmarkManifest`).
- Each task: question, level, reference answer, source-level gold evidence
  (`schemas.GoldEvidence`), expected behaviors.

## Task levels

1. Single-document factual retrieval
2. Multi-document comparison
3. Multi-hop synthesis
4. Contradiction and evidence analysis
5. Research synthesis

## Gold evidence creation

Gold labels are authored at the **document level** first (which papers an
ideal answer draws on), refined to chunk level where unambiguous. Authorship
notes are kept in the benchmark file (`GoldEvidence.note`).

## Experiment matrix

Minimum comparison (brief §12):

| # | System | Retrieval |
|---|---|---|
| 1 | Standard | dense |
| 2 | Standard | hybrid (dense + BM25) |
| 3 | Agentic | dense |
| 4 | Agentic | hybrid |

Later variants: ± reranking, max retrieval rounds, context budgets, models.

Each run persists (`schemas.ExperimentMetadata` + `schemas.TaskResult`):
configuration snapshot, git commit, timestamp, benchmark version, model
names, prompt versions, answers, retrieved chunks, agent traces, scores,
latency, cost. Format: JSONL under `reports/results/`.

## Statistical analysis plan (Phase 7)

- Per system × level × metric: mean, median, standard deviation, and
  confidence intervals (bootstrap).
- Paired comparisons across systems on the same tasks: Wilcoxon signed-rank
  as the primary test (no normality assumption); paired t-test reported
  alongside when its assumptions plausibly hold.
- Multiple-comparison awareness (Holm–Bonferroni) when testing across many
  metrics.
- **No significance claims without a test.** Absence of significance is
  reported as such.

Planned figures: quality by task level, cost vs. quality, latency vs.
quality, citation correctness by system, retrieval rounds by task
complexity.
