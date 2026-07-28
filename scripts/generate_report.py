"""Generate the comparison report from stored experiment results.

Usage:
    uv run python scripts/generate_report.py                       # all runs in reports/results
    uv run python scripts/generate_report.py --results <dir> ...   # specific runs
"""

import argparse
import sys
from pathlib import Path

from agentic_rag_bench.observability import configure_logging
from agentic_rag_bench.reporting import build_report

DEFAULT_RESULTS = Path("reports/results")
DEFAULT_OUT = Path("reports")


def discover(results_root: Path) -> list[Path]:
    if not results_root.is_dir():
        return []
    return sorted(child for child in results_root.iterdir() if (child / "metadata.json").is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        default=None,
        help="Experiment directories (default: every run under reports/results)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    configure_logging()
    experiment_dirs = args.results if args.results else discover(DEFAULT_RESULTS)
    if not experiment_dirs:
        print(
            "no experiment results found — run `make experiment` first",
            file=sys.stderr,
        )
        return 1
    try:
        report = build_report(experiment_dirs, args.out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"report failed: {exc}", file=sys.stderr)
        return 1
    print(f"report written: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
