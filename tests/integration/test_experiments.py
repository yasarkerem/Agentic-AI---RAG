"""Experiment runner: persistence, matrix, crash isolation, re-evaluation."""

import json
from pathlib import Path

import pytest

from agentic_rag_bench.config import load_system_config
from agentic_rag_bench.evaluation import load_benchmark
from agentic_rag_bench.experiments import load_results, reevaluate, run_experiment
from agentic_rag_bench.ingestion import ingest_corpus
from agentic_rag_bench.schemas import BenchmarkManifest, BenchmarkTask, SystemType

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
BENCHMARK = REPO_ROOT / "data" / "benchmark" / "benchmark_v0.1.json"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def processed_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    config = load_system_config(CONFIGS / "default.yaml")
    ingest_corpus(
        raw_dir=REPO_ROOT / "data" / "corpus_sample",
        processed_dir=out,
        chunking=config.chunking,
        embedding=config.embedding,
    )
    return out


@pytest.fixture(scope="module")
def small_benchmark() -> BenchmarkManifest:
    manifest = load_benchmark(BENCHMARK)
    # One task per level keeps the integration test fast but representative.
    picked: dict[int, BenchmarkTask] = {}
    for task in manifest.tasks:
        picked.setdefault(task.level, task)
    return BenchmarkManifest(
        benchmark_version=manifest.benchmark_version, tasks=list(picked.values())
    )


class TestRunExperiment:
    def test_persists_metadata_and_results(self, processed_dir, small_benchmark, tmp_path):
        out_dir = run_experiment(
            config_path=CONFIGS / "standard_rag.yaml",
            processed_dir=processed_dir,
            benchmark=small_benchmark,
            out_root=tmp_path,
            defaults_path=CONFIGS / "default.yaml",
        )
        metadata, results = load_results(out_dir)
        assert metadata.system_type is SystemType.STANDARD_RAG
        assert metadata.benchmark_version == small_benchmark.benchmark_version
        assert metadata.prompt_versions  # prompts recorded (brief §12)
        retrieval_snapshot = metadata.config_snapshot["retrieval"]
        assert isinstance(retrieval_snapshot, dict)
        assert retrieval_snapshot["top_k"] == 8
        assert len(results) == len(small_benchmark.tasks)
        for record in results:
            assert record.scores is not None
            assert record.scores.system is not None
            assert not record.answer.failed

    def test_agentic_run_records_traces(self, processed_dir, small_benchmark, tmp_path):
        out_dir = run_experiment(
            config_path=CONFIGS / "agentic_rag.yaml",
            processed_dir=processed_dir,
            benchmark=small_benchmark,
            out_root=tmp_path,
            defaults_path=CONFIGS / "default.yaml",
        )
        _, results = load_results(out_dir)
        for record in results:
            assert record.answer.trace is not None
            assert record.answer.trace.stop_reason is not None

    def test_hybrid_configs_run(self, processed_dir, small_benchmark, tmp_path):
        for name in ("standard_hybrid.yaml", "agentic_hybrid.yaml"):
            out_dir = run_experiment(
                config_path=CONFIGS / name,
                processed_dir=processed_dir,
                benchmark=small_benchmark,
                out_root=tmp_path,
                defaults_path=CONFIGS / "default.yaml",
            )
            _, results = load_results(out_dir)
            assert results

    def test_results_are_valid_jsonl(self, processed_dir, small_benchmark, tmp_path):
        out_dir = run_experiment(
            config_path=CONFIGS / "standard_rag.yaml",
            processed_dir=processed_dir,
            benchmark=small_benchmark,
            out_root=tmp_path,
            defaults_path=CONFIGS / "default.yaml",
        )
        for line in (out_dir / "results.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["experiment_id"].startswith("standard-rag-dense")

    def test_reevaluate_rewrites_scores(self, processed_dir, small_benchmark, tmp_path):
        out_dir = run_experiment(
            config_path=CONFIGS / "standard_rag.yaml",
            processed_dir=processed_dir,
            benchmark=small_benchmark,
            out_root=tmp_path,
            defaults_path=CONFIGS / "default.yaml",
        )
        _, before = load_results(out_dir)
        reevaluate(out_dir, processed_dir, small_benchmark)
        _, after = load_results(out_dir)
        assert len(after) == len(before)
        for old, new in zip(before, after, strict=True):
            assert new.task_id == old.task_id
            assert new.scores is not None

    def test_missing_experiment_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_results(tmp_path / "nonexistent")
