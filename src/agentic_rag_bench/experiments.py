"""Experiment runner: systems x benchmark tasks, persisted reproducibly (brief §12).

Each run writes one directory under the results root:

    <out>/<experiment_id>/
        metadata.json   ExperimentMetadata: config snapshot, git commit,
                        benchmark version, model names, prompt versions, seed
        results.jsonl   one TaskResult per benchmark task (answer + scores)

Failures are recorded per task, never silently dropped; the run continues.
"""

import subprocess
from pathlib import Path

from agentic_rag_bench.agentic_rag import AgenticRAGPipeline
from agentic_rag_bench.config import SystemConfig, load_system_config
from agentic_rag_bench.embeddings import build_embedder
from agentic_rag_bench.evaluation import Evaluator, LLMJudge
from agentic_rag_bench.generation import build_llm_provider
from agentic_rag_bench.generation.prompts import PROMPT_VERSIONS
from agentic_rag_bench.observability import get_logger
from agentic_rag_bench.reranking import build_reranker
from agentic_rag_bench.retrieval import Retriever
from agentic_rag_bench.schemas import (
    BenchmarkManifest,
    ExperimentMetadata,
    PipelineAnswer,
    SystemType,
    TaskResult,
    utc_now,
)
from agentic_rag_bench.standard_rag import StandardRAGPipeline

logger = get_logger("experiments")

METADATA_FILE = "metadata.json"
RESULTS_FILE = "results.jsonl"

Pipeline = StandardRAGPipeline | AgenticRAGPipeline


def git_commit_hash() -> str | None:
    """Current commit hash, or None outside a git checkout (recorded either way)."""
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return output.stdout.strip() or None


def build_pipeline(config: SystemConfig, processed_dir: Path) -> Pipeline:
    """Assemble a runnable pipeline from a system config and ingested artifacts."""
    retriever = Retriever.from_processed_dir(
        processed_dir, build_embedder(config.embedding), config.retrieval
    )
    llm = build_llm_provider(config.generation)
    reranker = build_reranker(config.reranker) if config.reranker.enabled else None
    if config.system_type is SystemType.AGENTIC_RAG:
        return AgenticRAGPipeline(config, retriever, llm, reranker)
    return StandardRAGPipeline(config, retriever, llm, reranker)


def run_experiment(
    config_path: Path,
    processed_dir: Path,
    benchmark: BenchmarkManifest,
    out_root: Path,
    defaults_path: Path | None = None,
    use_llm_judge: bool = False,
    config_name: str | None = None,
) -> Path:
    """Run one system over the whole benchmark. Returns the experiment directory."""
    config = load_system_config(config_path, defaults_path=defaults_path)
    pipeline = build_pipeline(config, processed_dir)
    retriever = Retriever.from_processed_dir(
        processed_dir, build_embedder(config.embedding), config.retrieval
    )
    judge = LLMJudge(build_llm_provider(config.generation)) if use_llm_judge else None
    evaluator = Evaluator(retriever.store, k=config.retrieval.top_k, llm_judge=judge)

    started = utc_now()
    experiment_id = f"{config.name}_{started.strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = out_root / experiment_id
    out_dir.mkdir(parents=True, exist_ok=False)

    metadata = ExperimentMetadata(
        experiment_id=experiment_id,
        system_type=config.system_type,
        config_name=config_name or config_path.name,
        config_snapshot=config.model_dump(mode="json"),
        git_commit=git_commit_hash(),
        benchmark_version=benchmark.benchmark_version,
        model_names={
            "generation": config.generation.model,
            "embedding": config.embedding.model,
            "reranker": config.reranker.model if config.reranker.enabled else "disabled",
        },
        prompt_versions=dict(PROMPT_VERSIONS),
        seed=config.seed,
        started_at=started,
    )
    (out_dir / METADATA_FILE).write_text(
        metadata.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    log = logger.bind(experiment_id=experiment_id, system_type=config.system_type.value)
    log.info("experiment_started", tasks=len(benchmark.tasks), config=str(config_path))

    with (out_dir / RESULTS_FILE).open("w", encoding="utf-8") as results:
        for task in benchmark.tasks:
            try:
                answer = pipeline.run(task.task_id, task.question)
            except Exception as exc:
                log.error("task_crashed", task_id=task.task_id, error=str(exc))
                answer = PipelineAnswer(
                    query_id=task.task_id,
                    system_type=config.system_type,
                    question=task.question,
                    answer="",
                    failed=True,
                    failure_reason=f"unhandled: {exc}",
                )
            scores = evaluator.evaluate(task, answer)
            record = TaskResult(
                experiment_id=experiment_id,
                task_id=task.task_id,
                task_level=int(task.level),
                answer=answer,
                scores=scores,
            )
            results.write(record.model_dump_json() + "\n")
            log.info(
                "task_done",
                task_id=task.task_id,
                failed=answer.failed,
                latency=round(answer.latency_seconds, 3),
            )

    log.info("experiment_complete", out=str(out_dir))
    return out_dir


def load_results(experiment_dir: Path) -> tuple[ExperimentMetadata, list[TaskResult]]:
    """Load one experiment's metadata and results."""
    metadata_path = experiment_dir / METADATA_FILE
    results_path = experiment_dir / RESULTS_FILE
    if not metadata_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(f"{experiment_dir} is not a complete experiment directory")
    metadata = ExperimentMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    results = [
        TaskResult.model_validate_json(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return metadata, results


def reevaluate(experiment_dir: Path, processed_dir: Path, benchmark: BenchmarkManifest) -> None:
    """Recompute scores for stored answers (e.g. after evaluator changes)."""
    metadata, results = load_results(experiment_dir)
    config = SystemConfig.model_validate(metadata.config_snapshot)
    retriever = Retriever.from_processed_dir(
        processed_dir, build_embedder(config.embedding), config.retrieval
    )
    evaluator = Evaluator(retriever.store, k=config.retrieval.top_k)
    tasks = {task.task_id: task for task in benchmark.tasks}

    with (experiment_dir / RESULTS_FILE).open("w", encoding="utf-8") as fh:
        for record in results:
            task = tasks.get(record.task_id)
            if task is None:
                logger.warning("task_missing_from_benchmark", task_id=record.task_id)
                fh.write(record.model_dump_json() + "\n")
                continue
            updated = record.model_copy(update={"scores": evaluator.evaluate(task, record.answer)})
            fh.write(updated.model_dump_json() + "\n")
    logger.info("reevaluated", experiment=str(experiment_dir), rows=len(results))
