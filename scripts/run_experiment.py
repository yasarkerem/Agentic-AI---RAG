"""Run the experiment matrix: each system config over the full benchmark.

Defaults run the minimum comparison from the brief (§12) — standard dense,
standard hybrid, agentic dense, agentic hybrid — over the sample corpus with
the mock provider (free, reproducible). Ingests the sample corpus first if
data/processed is missing.

Usage:
    uv run python scripts/run_experiment.py                    # full matrix
    uv run python scripts/run_experiment.py --configs configs/standard_rag.yaml
    uv run python scripts/run_experiment.py --reevaluate reports/results/<id> ...
"""

import argparse
import sys
from pathlib import Path

from agentic_rag_bench.config import ConfigError, load_system_config
from agentic_rag_bench.evaluation import BenchmarkError, load_benchmark
from agentic_rag_bench.experiments import reevaluate, run_experiment
from agentic_rag_bench.ingestion import IngestionError, ingest_corpus
from agentic_rag_bench.observability import configure_logging

DEFAULT_CONFIGS = [
    Path("configs/standard_rag.yaml"),
    Path("configs/standard_hybrid.yaml"),
    Path("configs/agentic_rag.yaml"),
    Path("configs/agentic_hybrid.yaml"),
]
DEFAULTS_FILE = Path("configs/default.yaml")
DEFAULT_BENCHMARK = Path("data/benchmark/benchmark_v0.1.json")
DEFAULT_PROCESSED = Path("data/processed")
DEFAULT_OUT = Path("reports/results")
SAMPLE_CORPUS = Path("data/corpus_sample")


def ensure_processed(processed: Path) -> None:
    if (processed / "chunks.jsonl").is_file():
        return
    print(f"{processed} missing — ingesting sample corpus first")
    config = load_system_config(DEFAULTS_FILE)
    ingest_corpus(
        raw_dir=SAMPLE_CORPUS,
        processed_dir=processed,
        chunking=config.chunking,
        embedding=config.embedding,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Also run the LLM judge (uses the config's provider; mock by default)",
    )
    parser.add_argument(
        "--reevaluate",
        type=Path,
        nargs="+",
        default=None,
        metavar="EXPERIMENT_DIR",
        help="Recompute scores for stored experiment directories instead of running",
    )
    args = parser.parse_args()

    configure_logging()
    try:
        benchmark = load_benchmark(args.benchmark)
        ensure_processed(args.processed)

        if args.reevaluate is not None:
            targets = []
            for path in args.reevaluate:
                if (path / "metadata.json").is_file():
                    targets.append(path)
                else:  # allow passing the results root: walk its children
                    targets.extend(
                        child
                        for child in sorted(path.iterdir())
                        if (child / "metadata.json").is_file()
                    )
            if not targets:
                print("nothing to re-evaluate", file=sys.stderr)
                return 1
            for target in targets:
                reevaluate(target, args.processed, benchmark)
                print(f"re-evaluated {target}")
            return 0

        for config_path in args.configs:
            out_dir = run_experiment(
                config_path=config_path,
                processed_dir=args.processed,
                benchmark=benchmark,
                out_root=args.out,
                defaults_path=DEFAULTS_FILE,
                use_llm_judge=args.judge,
            )
            print(f"experiment complete: {out_dir}")
    except (ConfigError, BenchmarkError, IngestionError, FileNotFoundError) as exc:
        print(f"experiment failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
