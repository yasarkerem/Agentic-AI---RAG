"""Report generation from real (mock-provider) experiment runs."""

from pathlib import Path

import pytest

from agentic_rag_bench.config import load_system_config
from agentic_rag_bench.evaluation import load_benchmark
from agentic_rag_bench.experiments import run_experiment
from agentic_rag_bench.ingestion import ingest_corpus
from agentic_rag_bench.reporting import build_report
from agentic_rag_bench.schemas import BenchmarkManifest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def two_experiments(tmp_path_factory):
    processed = tmp_path_factory.mktemp("processed")
    out = tmp_path_factory.mktemp("results")
    config = load_system_config(CONFIGS / "default.yaml")
    ingest_corpus(
        raw_dir=REPO_ROOT / "data" / "corpus_sample",
        processed_dir=processed,
        chunking=config.chunking,
        embedding=config.embedding,
    )
    manifest = load_benchmark(REPO_ROOT / "data" / "benchmark" / "benchmark_v0.1.json")
    small = BenchmarkManifest(
        benchmark_version=manifest.benchmark_version, tasks=manifest.tasks[:6]
    )
    dirs = []
    for name in ("standard_rag.yaml", "agentic_rag.yaml"):
        dirs.append(
            run_experiment(
                config_path=CONFIGS / name,
                processed_dir=processed,
                benchmark=small,
                out_root=out,
                defaults_path=CONFIGS / "default.yaml",
            )
        )
    return dirs


class TestReport:
    def test_report_and_figures_created(self, two_experiments, tmp_path):
        report = build_report(two_experiments, tmp_path)
        assert report.is_file()
        text = report.read_text(encoding="utf-8")
        assert "Harness validation run" in text  # mock runs are labeled
        assert "standard-rag-dense" in text
        assert "agentic-rag-dense" in text
        assert "Wilcoxon" in text
        for figure in (
            "quality_by_level.png",
            "cost_vs_quality.png",
            "latency_vs_quality.png",
            "citation_correctness.png",
            "rounds_by_level.png",
        ):
            assert (tmp_path / "figures" / figure).is_file(), figure

    def test_no_experiments_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="no experiments"):
            build_report([], tmp_path)
