"""Comparison report: tables, statistics, and figures from experiment results.

Output: <out>/report.md plus PNG figures under <out>/figures/. When the
runs used the mock provider, the report is stamped as a harness validation
run — those numbers say nothing about real RAG quality and are labeled so.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from agentic_rag_bench.evaluation.stats import Summary, summarize, wilcoxon_signed_rank
from agentic_rag_bench.experiments import load_results
from agentic_rag_bench.observability import get_logger
from agentic_rag_bench.schemas import ExperimentMetadata, TaskResult

logger = get_logger("reporting")

# Validated categorical palette (dataviz reference, light mode) — fixed slot
# order by system position; contrast WARN on slots 3-4 is relieved by direct
# value labels on every bar.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT_PRIMARY = "#333230"
TEXT_SECONDARY = "#6f6e66"
SURFACE = "#fcfcfb"

ANSWER_METRICS = [
    "factual_correctness",
    "completeness",
    "citation_correctness",
    "citation_completeness",
    "groundedness",
    "hallucination_rate",
    "structure_readability",
]
COMPOSITE_PARTS = ["factual_correctness", "completeness", "groundedness", "citation_correctness"]


class Experiment:
    """One loaded experiment with metric access helpers."""

    def __init__(self, metadata: ExperimentMetadata, results: list[TaskResult]) -> None:
        self.metadata = metadata
        self.results = results
        self.name = str(metadata.config_snapshot.get("name", metadata.experiment_id))

    def answer_metric(self, metric: str, levels: set[int] | None = None) -> list[float]:
        values = []
        for r in self.results:
            if levels is not None and r.task_level not in levels:
                continue
            if r.scores is None or r.scores.answer is None:
                continue
            value = getattr(r.scores.answer, metric)
            if value is not None:
                values.append(float(value))
        return values

    def composite_quality(self, levels: set[int] | None = None) -> list[float]:
        """Mean of correctness, completeness, groundedness, citation correctness."""
        values = []
        for r in self.results:
            if levels is not None and r.task_level not in levels:
                continue
            if r.scores is None or r.scores.answer is None:
                continue
            parts = [float(getattr(r.scores.answer, p)) for p in COMPOSITE_PARTS]
            values.append(sum(parts) / len(parts))
        return values

    def paired_metric(self, metric: str, levels: set[int] | None = None) -> dict[str, float]:
        """task_id -> metric value, for paired comparisons."""
        pairs = {}
        for r in self.results:
            if levels is not None and r.task_level not in levels:
                continue
            if r.scores is None or r.scores.answer is None:
                continue
            value = getattr(r.scores.answer, metric)
            if value is not None:
                pairs[r.task_id] = float(value)
        return pairs

    def system_values(self, field: str) -> list[float]:
        return [
            float(getattr(r.scores.system, field))
            for r in self.results
            if r.scores is not None and r.scores.system is not None
        ]

    def rounds_by_level(self) -> dict[int, list[float]]:
        grouped: dict[int, list[float]] = defaultdict(list)
        for r in self.results:
            grouped[r.task_level].append(float(r.answer.retrieval_rounds))
        return grouped

    def quality_by_level(self) -> dict[int, list[float]]:
        grouped: dict[int, list[float]] = defaultdict(list)
        for r in self.results:
            if r.scores is None or r.scores.answer is None:
                continue
            parts = [float(getattr(r.scores.answer, p)) for p in COMPOSITE_PARTS]
            grouped[r.task_level].append(sum(parts) / len(parts))
        return grouped

    @property
    def total_cost(self) -> float:
        return sum(self.system_values("estimated_cost_usd"))

    @property
    def failure_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.answer.failed) / len(self.results)

    @property
    def is_mock(self) -> bool:
        return self.metadata.model_names.get("generation", "").startswith("mock")


def load_experiments(experiment_dirs: list[Path]) -> list[Experiment]:
    experiments = []
    for directory in experiment_dirs:
        metadata, results = load_results(directory)
        experiments.append(Experiment(metadata, results))
    return experiments


# --- figures -----------------------------------------------------------------


def _style_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color="#e4e3db", linewidth=0.6)
    ax.set_axisbelow(True)


def _new_figure(title: str) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, loc="left", pad=12)
    _style_axes(ax)
    return fig, ax


def _grouped_bars(
    ax: Axes,
    groups: list[str],
    series: dict[str, list[float]],
    label_values: bool = True,
) -> None:
    n_series = len(series)
    width = min(0.8 / max(n_series, 1), 0.22)
    for idx, (name, values) in enumerate(series.items()):
        offsets = [g + (idx - (n_series - 1) / 2) * (width + 0.02) for g in range(len(groups))]
        bars = ax.bar(offsets, values, width=width, color=PALETTE[idx % len(PALETTE)], label=name)
        if label_values:
            for bar, value in zip(bars, values, strict=True):
                ax.annotate(
                    f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=TEXT_PRIMARY,
                )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    if n_series >= 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def figure_quality_by_level(experiments: list[Experiment], out: Path) -> None:
    levels = sorted({level for e in experiments for level in e.quality_by_level()})
    fig, ax = _new_figure("Composite answer quality by task level")
    series = {}
    for e in experiments:
        by_level = e.quality_by_level()
        series[e.name] = [
            (sum(by_level[lv]) / len(by_level[lv])) if by_level.get(lv) else 0.0 for lv in levels
        ]
    _grouped_bars(ax, [f"L{lv}" for lv in levels], series)
    ax.set_ylabel("mean composite quality", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylim(0, 1.15)
    _save(fig, out)


def _scatter_vs_quality(
    experiments: list[Experiment], x_values: list[float], x_label: str, title: str, out: Path
) -> None:
    fig, ax = _new_figure(title)
    for idx, (e, x) in enumerate(zip(experiments, x_values, strict=True)):
        quality = e.composite_quality()
        y = sum(quality) / len(quality) if quality else 0.0
        ax.scatter([x], [y], s=90, color=PALETTE[idx % len(PALETTE)], zorder=3)
        ax.annotate(
            e.name,
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color=TEXT_PRIMARY,
        )
    ax.set_xlabel(x_label, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("mean composite quality", color=TEXT_SECONDARY, fontsize=9)
    _save(fig, out)


def figure_citation_correctness(experiments: list[Experiment], out: Path) -> None:
    fig, ax = _new_figure("Citation correctness by system")
    names = [e.name for e in experiments]
    values = []
    for e in experiments:
        metric = e.answer_metric("citation_correctness")
        values.append(sum(metric) / len(metric) if metric else 0.0)
    bars = ax.bar(
        range(len(names)),
        values,
        width=0.5,
        color=[PALETTE[i % len(PALETTE)] for i in range(len(names))],
    )
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.2f}",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            fontsize=8,
            color=TEXT_PRIMARY,
        )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("mean citation correctness", color=TEXT_SECONDARY, fontsize=9)
    _save(fig, out)


def figure_rounds_by_level(experiments: list[Experiment], out: Path) -> None:
    levels = sorted({level for e in experiments for level in e.rounds_by_level()})
    fig, ax = _new_figure("Retrieval rounds by task complexity")
    series = {}
    for e in experiments:
        by_level = e.rounds_by_level()
        series[e.name] = [
            (sum(by_level[lv]) / len(by_level[lv])) if by_level.get(lv) else 0.0 for lv in levels
        ]
    _grouped_bars(ax, [f"L{lv}" for lv in levels], series)
    ax.set_ylabel("mean retrieval rounds", color=TEXT_SECONDARY, fontsize=9)
    _save(fig, out)


# --- report ------------------------------------------------------------------


def _fmt(summary: Summary | None, digits: int = 3) -> str:
    if summary is None:
        return "—"
    return (
        f"{summary.mean:.{digits}f} "
        f"(median {summary.median:.{digits}f}, sd {summary.std:.{digits}f}, "
        f"95% CI {summary.ci_low:.{digits}f}-{summary.ci_high:.{digits}f})"
    )


def _paired_section(experiments: list[Experiment]) -> list[str]:
    """Paired Wilcoxon comparisons: agentic vs standard with matching retrieval."""
    lines = ["## Paired comparisons (Wilcoxon signed-rank, two-sided)", ""]
    by_name = {e.name: e for e in experiments}
    pairs = [
        ("standard-rag-dense", "agentic-rag-dense"),
        ("standard-rag-hybrid", "agentic-rag-hybrid"),
    ]
    level_groups: list[tuple[str, set[int] | None]] = [
        ("all levels", None),
        ("levels 1-2", {1, 2}),
        ("levels 4-5", {4, 5}),
    ]
    metrics = ["completeness", "citation_completeness", "groundedness"]
    tested_any = False
    for standard_name, agentic_name in pairs:
        standard = by_name.get(standard_name)
        agentic = by_name.get(agentic_name)
        if standard is None or agentic is None:
            continue
        lines.append(f"### {agentic_name} vs {standard_name}")
        lines.append("")
        lines.append("| Metric | Scope | n | W | z | p-value | note |")
        lines.append("|---|---|---|---|---|---|---|")
        for metric in metrics:
            for scope_name, levels in level_groups:
                a = agentic.paired_metric(metric, levels)
                s = standard.paired_metric(metric, levels)
                shared = sorted(set(a) & set(s))
                if not shared:
                    continue
                result = wilcoxon_signed_rank([a[t] for t in shared], [s[t] for t in shared])
                if result is None:
                    lines.append(
                        f"| {metric} | {scope_name} | {len(shared)} | — | — | — | "
                        "all paired differences are zero |"
                    )
                    continue
                tested_any = True
                note = "small sample — approximate" if result.small_sample else ""
                lines.append(
                    f"| {metric} | {scope_name} | {result.n} | {result.statistic:.1f} "
                    f"| {result.z:.2f} | {result.p_value:.4f} | {note} |"
                )
        lines.append("")
    if not tested_any:
        lines.append(
            "_No paired test produced a usable statistic (identical scores or "
            "missing matching runs). No significance claims are made._"
        )
        lines.append("")
    return lines


def build_report(experiment_dirs: list[Path], out_dir: Path) -> Path:
    """Generate report.md and figures. Returns the report path."""
    experiments = load_experiments(experiment_dirs)
    if not experiments:
        raise ValueError("no experiments to report on")
    figures_dir = out_dir / "figures"

    figure_quality_by_level(experiments, figures_dir / "quality_by_level.png")
    _scatter_vs_quality(
        experiments,
        [e.total_cost for e in experiments],
        "total estimated cost (USD)",
        "Cost vs quality",
        figures_dir / "cost_vs_quality.png",
    )
    mean_latencies = []
    for e in experiments:
        latencies = e.system_values("latency_seconds")
        mean_latencies.append(sum(latencies) / len(latencies) if latencies else 0.0)
    _scatter_vs_quality(
        experiments,
        mean_latencies,
        "mean latency per task (s)",
        "Latency vs quality",
        figures_dir / "latency_vs_quality.png",
    )
    figure_citation_correctness(experiments, figures_dir / "citation_correctness.png")
    figure_rounds_by_level(experiments, figures_dir / "rounds_by_level.png")

    lines = ["# Benchmark comparison report", ""]
    if any(e.is_mock for e in experiments):
        lines += [
            "> **Harness validation run.** One or more systems used the deterministic",
            "> mock provider. These numbers validate the pipeline, persistence, and",
            "> statistics machinery only — they say **nothing** about real RAG quality",
            "> and must not be quoted as benchmark results.",
            "",
        ]
    lines += ["## Runs", ""]
    lines.append("| System | Experiment | Benchmark | Model | Tasks | Failure rate |")
    lines.append("|---|---|---|---|---|---|")
    for e in experiments:
        lines.append(
            f"| {e.name} | {e.metadata.experiment_id} | v{e.metadata.benchmark_version} "
            f"| {e.metadata.model_names.get('generation', '?')} | {len(e.results)} "
            f"| {e.failure_rate:.0%} |"
        )
    lines.append("")

    lines += ["## Answer metrics (mean over all tasks)", ""]
    header = "| Metric | " + " | ".join(e.name for e in experiments) + " |"
    lines.append(header)
    lines.append("|---" * (len(experiments) + 1) + "|")
    for metric in ANSWER_METRICS:
        row = [metric]
        for e in experiments:
            row.append(_fmt(summarize(e.answer_metric(metric))))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines += ["## System metrics", ""]
    lines.append(header)
    lines.append("|---" * (len(experiments) + 1) + "|")
    for field, label in [
        ("latency_seconds", "latency (s)"),
        ("llm_calls", "LLM calls"),
        ("input_tokens", "input tokens"),
        ("output_tokens", "output tokens"),
        ("estimated_cost_usd", "estimated cost (USD)"),
        ("retrieval_rounds", "retrieval rounds"),
    ]:
        row = [label]
        for e in experiments:
            row.append(_fmt(summarize(e.system_values(field))))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines += _paired_section(experiments)

    lines += [
        "## Figures",
        "",
        "![Quality by task level](figures/quality_by_level.png)",
        "![Cost vs quality](figures/cost_vs_quality.png)",
        "![Latency vs quality](figures/latency_vs_quality.png)",
        "![Citation correctness](figures/citation_correctness.png)",
        "![Retrieval rounds by complexity](figures/rounds_by_level.png)",
        "",
        "_Composite quality = mean of factual correctness, completeness, "
        "groundedness, and citation correctness (deterministic proxies unless an "
        "LLM judge ran)._",
        "",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("report_written", path=str(report_path), experiments=len(experiments))
    return report_path
