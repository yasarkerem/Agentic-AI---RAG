"""Benchmark manifest loading and validation."""

from pathlib import Path

from pydantic import ValidationError

from agentic_rag_bench.schemas import BenchmarkManifest


class BenchmarkError(Exception):
    """Raised when a benchmark file is missing or invalid."""


def load_benchmark(path: Path) -> BenchmarkManifest:
    if not path.is_file():
        raise BenchmarkError(f"Benchmark file not found: {path} — run `make benchmark`")
    try:
        manifest = BenchmarkManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as exc:
        raise BenchmarkError(f"Invalid benchmark file {path}: {exc}") from exc
    task_ids = [task.task_id for task in manifest.tasks]
    if len(task_ids) != len(set(task_ids)):
        raise BenchmarkError(f"Duplicate task IDs in {path}")
    return manifest


def save_benchmark(manifest: BenchmarkManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
