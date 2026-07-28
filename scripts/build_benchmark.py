"""Build the benchmark file: 25 tasks across 5 levels over the sample corpus.

Task definitions live here (single source of truth); the committed output is
data/benchmark/benchmark_v0.1.json. Reference answers and gold evidence are
authored against data/corpus_sample/ — original documents written for this
project, so gold labels are exact by construction.

Usage:
    uv run python scripts/build_benchmark.py [--out data/benchmark/benchmark_v0.1.json]
"""

import argparse
import sys
from pathlib import Path

from agentic_rag_bench.evaluation.benchmark import save_benchmark
from agentic_rag_bench.schemas import BenchmarkManifest, BenchmarkTask, GoldEvidence, TaskLevel

BENCHMARK_VERSION = "0.1"
DEFAULT_OUT = Path("data/benchmark/benchmark_v0.1.json")


def _gold(*document_ids: str) -> list[GoldEvidence]:
    return [GoldEvidence(document_id=d) for d in document_ids]


def build_tasks() -> list[BenchmarkTask]:
    level1 = [
        BenchmarkTask(
            task_id="l1_raptor_contribution",
            level=TaskLevel.SINGLE_DOC_FACTUAL,
            question="What is the main contribution of RAPTOR?",
            reference_answer=(
                "RAPTOR's main contribution is a hierarchical retrieval index built by "
                "recursively clustering text chunks and summarizing each cluster with a "
                "language model, producing a tree whose leaves are raw chunks and whose "
                "internal nodes are increasingly abstract summaries that can be retrieved "
                "at any level."
            ),
            gold_evidence=_gold("raptor_overview"),
            expected_behaviors=["retrieve one or two relevant passages", "cite the correct source"],
        ),
        BenchmarkTask(
            task_id="l1_graphrag_contribution",
            level=TaskLevel.SINGLE_DOC_FACTUAL,
            question="What is the main contribution of GraphRAG?",
            reference_answer=(
                "GraphRAG's main contribution is an indexing pipeline that extracts an "
                "entity-and-relationship knowledge graph from the corpus, partitions it "
                "into communities, and writes a summary for each community; global "
                "questions are answered by map-reducing over community summaries."
            ),
            gold_evidence=_gold("graphrag_overview"),
            expected_behaviors=["retrieve one or two relevant passages", "cite the correct source"],
        ),
        BenchmarkTask(
            task_id="l1_bm25_scoring",
            level=TaskLevel.SINGLE_DOC_FACTUAL,
            question="What does BM25 score documents by?",
            reference_answer=(
                "BM25 scores documents by exact term overlap with the query, weighted by "
                "term rarity and document length."
            ),
            gold_evidence=_gold("hybrid_retrieval"),
            expected_behaviors=["answer directly", "cite the correct source"],
        ),
        BenchmarkTask(
            task_id="l1_lost_in_middle",
            level=TaskLevel.SINGLE_DOC_FACTUAL,
            question="What is the 'lost in the middle' effect in long-context models?",
            reference_answer=(
                "Models recall information placed in the middle of very long contexts "
                "less reliably than information near the beginning or end, so more "
                "context is not automatically better context."
            ),
            gold_evidence=_gold("long_context_rag"),
            expected_behaviors=["answer directly", "cite the correct source"],
        ),
        BenchmarkTask(
            task_id="l1_hallucination_form",
            level=TaskLevel.SINGLE_DOC_FACTUAL,
            question="What specific form does hallucination take in retrieval-augmented systems?",
            reference_answer=(
                "In RAG systems hallucination takes the form of answers containing claims "
                "not present in, or contradicted by, the retrieved passages."
            ),
            gold_evidence=_gold("hallucination_grounding"),
            expected_behaviors=["answer directly", "cite the correct source"],
        ),
    ]

    level2 = [
        BenchmarkTask(
            task_id="l2_graphrag_vs_raptor",
            level=TaskLevel.MULTI_DOC_COMPARISON,
            question=(
                "Compare GraphRAG and RAPTOR in terms of retrieval strategy, strengths, "
                "and limitations."
            ),
            reference_answer=(
                "RAPTOR retrieves from a tree of recursively clustered and summarized "
                "chunks (tree traversal or collapsed-tree ranking), excelling at "
                "integrating information within long documents but with expensive "
                "index builds and summary blur. GraphRAG retrieves via an entity "
                "knowledge graph with community summaries (local vs global search), "
                "excelling at corpus-wide sensemaking but costing many extraction "
                "calls and degrading with noisy graphs; both add little for narrow "
                "factoid lookups."
            ),
            gold_evidence=_gold("graphrag_overview", "raptor_overview"),
            expected_behaviors=["compare using consistent dimensions", "cite each major claim"],
        ),
        BenchmarkTask(
            task_id="l2_index_build_costs",
            level=TaskLevel.MULTI_DOC_COMPARISON,
            question=(
                "Compare the index-construction costs of RAPTOR and GraphRAG and their causes."
            ),
            reference_answer=(
                "Both are index-time heavy because they make model calls during "
                "ingestion: RAPTOR pays for a summarization call per internal tree "
                "node and must rebuild or patch the tree when documents change, while "
                "GraphRAG pays for entity extraction and community summarization "
                "across the whole corpus."
            ),
            gold_evidence=_gold("raptor_overview", "graphrag_overview"),
            expected_behaviors=["retrieve evidence from both documents", "cite each major claim"],
        ),
        BenchmarkTask(
            task_id="l2_reranking_setups",
            level=TaskLevel.MULTI_DOC_COMPARISON,
            question=(
                "Compare the experimental setups of the two reranking evaluations in the "
                "corpus: first-stage retrievers, query types, and metrics."
            ),
            reference_answer=(
                "The positive study used weak first stages (BM25 or an early "
                "bi-encoder), short factoid queries, and precision at small cutoffs; "
                "the negative study used a strong tuned dense retriever, long "
                "multi-faceted research questions, recall and coverage at larger "
                "cutoffs, and out-of-domain scientific text."
            ),
            gold_evidence=_gold("reranking_gains", "reranking_limits"),
            expected_behaviors=["compare using consistent dimensions", "cite both studies"],
        ),
        BenchmarkTask(
            task_id="l2_standard_vs_agentic",
            level=TaskLevel.MULTI_DOC_COMPARISON,
            question="Compare standard single-pass RAG with agentic RAG pipelines.",
            reference_answer=(
                "Standard RAG retrieves once unconditionally and generates once; "
                "agentic RAG plans, decomposes into subquestions, retrieves "
                "iteratively with evidence assessment and query rewriting, and "
                "verifies citations — gaining on complex multi-hop and synthesis "
                "tasks while adding latency and cost that simple factoid lookups "
                "do not repay."
            ),
            gold_evidence=_gold("agentic_rag_survey", "adaptive_retrieval"),
            expected_behaviors=["compare using consistent dimensions", "cite each major claim"],
        ),
        BenchmarkTask(
            task_id="l2_longcontext_vs_retrieval",
            level=TaskLevel.MULTI_DOC_COMPARISON,
            question=(
                "Compare stuffing full documents into a long context window against "
                "retrieval-based approaches: when does each win?"
            ),
            reference_answer=(
                "Full-context wins for single long documents where chunking might cut "
                "relevant passages and evidence is diffuse; retrieval wins on cost, "
                "avoids the lost-in-the-middle effect, and yields an explicit citable "
                "evidence set — the practical division of labor is retrieval "
                "narrowing the corpus to a generous working set for the long window."
            ),
            gold_evidence=_gold("long_context_rag", "raptor_overview"),
            expected_behaviors=["present trade-offs from both sides", "cite each major claim"],
        ),
    ]

    level3 = [
        BenchmarkTask(
            task_id="l3_longcontext_retrieval_fit",
            level=TaskLevel.MULTI_HOP_SYNTHESIS,
            question=(
                "Which retrieval approaches are most suitable for long-context language "
                "models, and why?"
            ),
            reference_answer=(
                "Approaches that retrieve fewer, larger, more coherent units suit "
                "long-context models: whole sections or documents, hierarchical "
                "indexes like RAPTOR whose summary nodes compress many chunks into "
                "one retrievable unit, and coverage-oriented retrieval such as hybrid "
                "methods that fill a large budget with diverse sources rather than "
                "near-duplicate top hits."
            ),
            gold_evidence=_gold("long_context_rag", "raptor_overview", "hybrid_retrieval"),
            expected_behaviors=["connect evidence across documents", "explain trade-offs"],
        ),
        BenchmarkTask(
            task_id="l3_hallucination_stack",
            level=TaskLevel.MULTI_HOP_SYNTHESIS,
            question=(
                "How do grounding instructions, citation verification, and evaluation "
                "practice combine to reduce unsupported claims in RAG systems?"
            ),
            reference_answer=(
                "Grounding instructions constrain answers to provided context and "
                "shift errors toward abstention; deterministic and entailment-based "
                "citation verification catches nonexistent passages, unquoted spans, "
                "and irrelevant citations; and evaluation that measures groundedness "
                "and hallucination rate separately from retrieval quality makes the "
                "remaining failures visible and comparable."
            ),
            gold_evidence=_gold("hallucination_grounding", "rag_evaluation"),
            expected_behaviors=["combine findings across documents", "cite each major claim"],
        ),
        BenchmarkTask(
            task_id="l3_source_coverage",
            level=TaskLevel.MULTI_HOP_SYNTHESIS,
            question=(
                "Why does source coverage matter for multi-document synthesis, and which "
                "retrieval choices improve or harm it?"
            ),
            reference_answer=(
                "Synthesis tasks need all relevant sources covered, not just the best "
                "one ranked first; hybrid retrieval increases recall of unique "
                "relevant documents, while reranking can concentrate top ranks on one "
                "source and reduce diversity, harming coverage-sensitive tasks."
            ),
            gold_evidence=_gold("hybrid_retrieval", "reranking_limits"),
            expected_behaviors=["connect evidence across documents", "explain trade-offs"],
        ),
        BenchmarkTask(
            task_id="l3_when_iteration_pays",
            level=TaskLevel.MULTI_HOP_SYNTHESIS,
            question=(
                "For which task types does iterative or adaptive retrieval pay off, and "
                "when is single-pass retrieval enough?"
            ),
            reference_answer=(
                "Iterative and adaptive retrieval pays on multi-hop and "
                "knowledge-intensive tasks where one pass predictably misses needed "
                "evidence; on single-fact questions gains shrink, added calls cost "
                "latency, and unnecessary retrieval can inject distracting context — "
                "so systems route easy queries past the loop."
            ),
            gold_evidence=_gold("adaptive_retrieval", "agentic_rag_survey"),
            expected_behaviors=["combine findings", "explain trade-offs"],
        ),
        BenchmarkTask(
            task_id="l3_trustworthy_comparison",
            level=TaskLevel.MULTI_HOP_SYNTHESIS,
            question=("What design choices make a comparison between two RAG systems trustworthy?"),
            reference_answer=(
                "Separate retrieval metrics from answer metrics so failures are "
                "attributable; use paired designs where every system answers the same "
                "tasks, with statistical tests over pairs; never rely on an LLM judge "
                "alone — pin judge model and prompt versions, add deterministic "
                "checks and gold labels; and version benchmarks, prompts, and "
                "evaluators so scores stay comparable."
            ),
            gold_evidence=_gold("rag_evaluation"),
            expected_behaviors=["synthesize evaluation guidance", "cite each major claim"],
        ),
    ]

    level4 = [
        BenchmarkTask(
            task_id="l4_reranking_disagreement",
            level=TaskLevel.CONTRADICTION_ANALYSIS,
            question=(
                "Which papers disagree about the effectiveness of reranking in RAG, and "
                "what may explain the disagreement?"
            ),
            reference_answer=(
                "The corpus contains a study reporting large consistent precision "
                "gains from cross-encoder reranking and one reporting marginal or "
                "negative gains; the disagreement is explained by setup, not the "
                "method: first-stage strength, factoid versus multi-faceted queries, "
                "precision-at-small-cutoff versus recall/coverage metrics, and domain "
                "match between the reranker's training data and the corpus."
            ),
            gold_evidence=_gold("reranking_gains", "reranking_limits"),
            expected_behaviors=["identify conflicting findings", "present both sides fairly"],
        ),
        BenchmarkTask(
            task_id="l4_reranker_default",
            level=TaskLevel.CONTRADICTION_ANALYSIS,
            question=(
                "Should a reranker be enabled by default in a RAG pipeline? Present the "
                "conflicting recommendations in the corpus."
            ),
            reference_answer=(
                "One study recommends always adding a cross-encoder when the first "
                "stage is BM25 or an older bi-encoder; the other recommends treating "
                "reranking as conditional — validated per corpus and metric, with "
                "domain-matched rerankers for specialized text — because gains "
                "disappear with strong first stages and out-of-domain content."
            ),
            gold_evidence=_gold("reranking_gains", "reranking_limits"),
            expected_behaviors=["present both recommendations", "explain the conditions"],
        ),
        BenchmarkTask(
            task_id="l4_retrieval_obsolete",
            level=TaskLevel.CONTRADICTION_ANALYSIS,
            question=(
                "Does long context make retrieval obsolete? Present the tension and how "
                "the corpus resolves it."
            ),
            reference_answer=(
                "Full-context stuffing can beat retrieval for single long documents, "
                "which suggests retrieval may be unnecessary; but cost, positional "
                "recall failures, and the need for citable provenance keep retrieval "
                "relevant — the corpus resolves the tension as a division of labor "
                "where retrieval narrows and the long window consumes generously."
            ),
            gold_evidence=_gold("long_context_rag"),
            expected_behaviors=["present the tension fairly", "explain the resolution"],
        ),
        BenchmarkTask(
            task_id="l4_more_or_less_retrieval",
            level=TaskLevel.CONTRADICTION_ANALYSIS,
            question=(
                "Reconcile the claim that iterative retrieval improves complex tasks "
                "with the finding that retrieving when unnecessary lowers accuracy."
            ),
            reference_answer=(
                "Both hold: iteration gathers evidence single-shot retrieval misses "
                "on multi-hop tasks, while unnecessary retrieval injects distracting "
                "context on simple ones — the reconciliation is adaptivity, deciding "
                "per query whether and how much to retrieve, with complexity routing "
                "at intake."
            ),
            gold_evidence=_gold("adaptive_retrieval", "agentic_rag_survey"),
            expected_behaviors=["identify the apparent conflict", "explain the reconciliation"],
        ),
        BenchmarkTask(
            task_id="l4_judge_tension",
            level=TaskLevel.CONTRADICTION_ANALYSIS,
            question=(
                "What conflicting considerations surround using LLM-as-judge for RAG "
                "evaluation, and how should they be balanced?"
            ),
            reference_answer=(
                "LLM judges scale far beyond human annotation and correlate "
                "reasonably with human judgment, yet carry biases toward length, "
                "fluency, self-similar style, and presentation order; the reported "
                "balance is treating the judge as one evaluator among several, "
                "paired with deterministic checks and gold labels, with pinned "
                "versions and human audits of samples."
            ),
            gold_evidence=_gold("rag_evaluation"),
            expected_behaviors=["present both sides", "state the balancing practice"],
        ),
    ]

    level5 = [
        BenchmarkTask(
            task_id="l5_adaptive_review",
            level=TaskLevel.RESEARCH_SYNTHESIS,
            question=(
                "Write a structured literature review on adaptive retrieval for RAG, "
                "including major approaches, evidence, limitations, and open research gaps."
            ),
            reference_answer=(
                "A complete review covers the three approach families "
                "(self-reflective control signals, active retrieval during "
                "generation, and query transformation), the evidence that gains "
                "concentrate on multi-hop and knowledge-intensive tasks while simple "
                "questions see little benefit, limitations around unreliable "
                "self-assessment and incomparable evaluations, and open gaps in "
                "principled stopping criteria, cost-aware policies, and benchmarks "
                "that measure retrieval decisions themselves."
            ),
            gold_evidence=_gold("adaptive_retrieval", "agentic_rag_survey", "rag_evaluation"),
            expected_behaviors=[
                "organize the answer into sections",
                "identify recurring themes",
                "cite sources precisely",
                "distinguish evidence from interpretation",
            ],
        ),
        BenchmarkTask(
            task_id="l5_structured_index_review",
            level=TaskLevel.RESEARCH_SYNTHESIS,
            question=(
                "Write a structured overview of hierarchical and graph-based retrieval "
                "indexes: approaches, what they enable, and their costs."
            ),
            reference_answer=(
                "A complete overview covers RAPTOR's recursive cluster-and-summarize "
                "tree and its collapsed-tree retrieval, GraphRAG's entity graph with "
                "community summaries and local/global search, what each enables "
                "(long-document integration; corpus-wide sensemaking), shared costs "
                "(model calls at index time, rebuild burden, degradation on noisy "
                "input), and their fit with long-context generation."
            ),
            gold_evidence=_gold("raptor_overview", "graphrag_overview", "long_context_rag"),
            expected_behaviors=["organize by approach", "cover costs", "cite sources precisely"],
        ),
        BenchmarkTask(
            task_id="l5_grounding_review",
            level=TaskLevel.RESEARCH_SYNTHESIS,
            question=(
                "Write a review of techniques for reducing hallucination and improving "
                "citation quality in RAG systems."
            ),
            reference_answer=(
                "A complete review covers grounding instructions and abstention "
                "trade-offs, inline attribution requirements, temperature reduction, "
                "deterministic citation checks (existence, verbatim quotes), "
                "entailment-based support checking, agentic citation verification "
                "before finalizing answers, and measurement via groundedness and "
                "hallucination rate."
            ),
            gold_evidence=_gold("hallucination_grounding", "rag_evaluation", "agentic_rag_survey"),
            expected_behaviors=["organize by technique", "cite sources precisely"],
        ),
        BenchmarkTask(
            task_id="l5_retrieval_stack_review",
            level=TaskLevel.RESEARCH_SYNTHESIS,
            question=(
                "Write a review of first-stage retrieval and reranking choices for "
                "technical corpora, including where the evidence conflicts."
            ),
            reference_answer=(
                "A complete review covers sparse versus dense signals and their "
                "failure modes, hybrid fusion methods (weighted score fusion, RRF) "
                "and when hybrid helps, the conflicting reranking evidence and its "
                "setup-based explanation, and the recommendation to validate "
                "reranking per corpus and metric rather than enabling it by default."
            ),
            gold_evidence=_gold("hybrid_retrieval", "reranking_gains", "reranking_limits"),
            expected_behaviors=["present conflicting evidence fairly", "cite sources precisely"],
        ),
        BenchmarkTask(
            task_id="l5_open_gaps",
            level=TaskLevel.RESEARCH_SYNTHESIS,
            question=(
                "Synthesize the open research gaps identified across the corpus for "
                "next-generation RAG systems."
            ),
            reference_answer=(
                "Recurring gaps include principled stopping criteria and cost-aware "
                "policies for iterative retrieval, benchmarks that evaluate retrieval "
                "decisions rather than only final answers, evaluation practices "
                "robust to judge bias and contamination, affordable gold labels for "
                "multimodal elements with verifiable grounding to figures and "
                "tables, and layout-aware ingestion."
            ),
            gold_evidence=_gold(
                "adaptive_retrieval", "rag_evaluation", "multimodal_rag", "agentic_rag_survey"
            ),
            expected_behaviors=["identify recurring themes", "cite sources precisely"],
        ),
    ]

    return level1 + level2 + level3 + level4 + level5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = BenchmarkManifest(benchmark_version=BENCHMARK_VERSION, tasks=build_tasks())
    per_level = {level.value: len(manifest.tasks_at_level(level)) for level in list(TaskLevel)}
    if any(count < 5 for count in per_level.values()):
        print(f"benchmark incomplete: tasks per level = {per_level}", file=sys.stderr)
        return 1
    save_benchmark(manifest, args.out)
    print(
        f"Wrote {len(manifest.tasks)} tasks (per level: {per_level}) "
        f"to {args.out} [version {manifest.benchmark_version}]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
