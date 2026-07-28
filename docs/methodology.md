# Methodology

> **Status:** implemented (Phases 1–7). The evaluation framework described
> here is in place and tested; real-model results have not yet been produced.

## Research question

Does an agentic retrieval pipeline produce more accurate, complete,
well-cited, and useful answers than standard RAG when tasks require
multi-step research and synthesis?

## Hypotheses

- **H1** — Agentic RAG outperforms standard RAG on Level 4–5 tasks in
  completeness and citation coverage.
- **H2** — Standard RAG has lower latency and cost on Level 1–2 tasks.
- **H3** — Agentic RAG retrieves a more diverse set of relevant sources on
  multi-hop tasks.
- **H4** — Agentic RAG reduces unsupported claims when citation verification
  is enabled.
- **H5** — The advantage of agentic RAG increases with task complexity.

Results will be reported whether or not they support these hypotheses.

## Variables

- **Independent:** pipeline type (standard vs. agentic), retrieval mode
  (dense vs. hybrid), task complexity level (1–5).
- **Dependent:** retrieval metrics, answer quality metrics, system metrics
  (see brief §11 and `schemas.evaluation`).
- **Controlled (fairness):** corpus, embedding model, vector store, reranker,
  generation model, token/cost budgets, seeds.

## Evaluation methodology

Layered evaluation; no single evaluator is trusted alone:

1. **Deterministic checks** — citation existence, quote presence in the
   cited chunk, retrieval metrics against gold labels, cost/latency.
2. **Reference answers + source-level gold labels** — authored with the
   benchmark (Phase 5).
3. **LLM-as-judge** — versioned prompts and judge models, stored with each
   result; used for graded qualities (completeness, groundedness, structure).
4. **Optional human evaluation** — spot checks on a sample.

## Fairness controls

Both systems share retrieval infrastructure, models, and budgets
(`configs/default.yaml` is the single shared base; tested in
`tests/unit/test_config.py::TestRepositoryConfigs::test_fairness_controls_identical_across_systems`).

## Deterministic metric proxies (documented limitations)

The `DeterministicJudge` (versioned `deterministic-judge/v1`) computes exact,
reproducible proxies whose limits are stated up front:

| Metric | Proxy | Limitation |
|---|---|---|
| factual_correctness | token-F1 vs reference answer | lexical agreement, not truth |
| completeness | token recall of the reference | rewards vocabulary overlap |
| citation_correctness | fraction of citations passing exact checks | exact by construction |
| citation_completeness | fraction of gold documents cited | document-level only |
| groundedness | answer content words found in retrieved evidence | lexical support proxy |
| hallucination_rate | 1 − groundedness | inherits the proxy's limits |
| contradiction_handling | cites ≥2 gold docs + contrastive language (L4+) | crude heuristic |
| structure_readability | length/paragraph heuristic | crude heuristic |

The LLM judge (versioned prompt + pinned model recorded per run) overrides
the graded qualities when enabled; citation metrics always stay deterministic.

## Limitations (known in advance)

- Small initial corpus (AI/RAG literature only) — findings may not
  generalize to other domains. The sample corpus is original summary prose
  written for this project, not actual papers.
- LLM-judge bias — mitigated but not eliminated by deterministic checks and
  judge versioning.
- Single-run latency/cost measurements are noisy — reported with spread
  (sd, bootstrap CI), not as point estimates.
- Mock-provider runs validate the harness only; the generated report stamps
  them as such and they must never be quoted as findings.
