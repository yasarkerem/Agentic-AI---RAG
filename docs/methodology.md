# Methodology

> **Status:** skeleton (Phase 1). The evaluation methodology is finalized in
> Phase 5; this records the design the implementation is being built toward.

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

## Limitations (known in advance)

- Small initial corpus (AI/RAG literature only) — findings may not
  generalize to other domains.
- LLM-judge bias — mitigated but not eliminated by deterministic checks and
  judge versioning.
- Single-run latency/cost measurements are noisy — reported with spread, not
  as point estimates.
