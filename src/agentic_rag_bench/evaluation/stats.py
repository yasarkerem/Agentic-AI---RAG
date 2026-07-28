"""Statistical analysis for paired benchmark comparisons (brief §13).

Implemented in-repo to keep dependencies light; the Wilcoxon signed-rank
test uses the standard normal approximation with tie correction and
continuity correction, validated against hand-computed values in tests.
For very small samples (n < 10) the approximation is rough — results carry
an explicit warning flag instead of silent confidence.
"""

import math
import random
from dataclasses import dataclass
from statistics import mean, median, stdev


@dataclass(frozen=True)
class Summary:
    """Descriptive statistics for one metric sample."""

    n: int
    mean: float
    median: float
    std: float
    ci_low: float
    ci_high: float


def summarize(values: list[float], seed: int = 42, n_boot: int = 2000) -> Summary | None:
    """Mean/median/std plus a seeded bootstrap 95% CI. None for empty input."""
    if not values:
        return None
    if len(values) == 1:
        v = values[0]
        return Summary(n=1, mean=v, median=v, std=0.0, ci_low=v, ci_high=v)
    rng = random.Random(seed)
    boot_means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    low = boot_means[int(0.025 * n_boot)]
    high = boot_means[min(n_boot - 1, int(0.975 * n_boot))]
    return Summary(
        n=len(values),
        mean=mean(values),
        median=median(values),
        std=stdev(values),
        ci_low=low,
        ci_high=high,
    )


@dataclass(frozen=True)
class WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test via normal approximation."""

    n: int  # non-zero differences used
    statistic: float  # W = min(W+, W-)
    z: float
    p_value: float
    small_sample: bool  # n < 10: approximation is rough — interpret with caution

    @property
    def usable(self) -> bool:
        return self.n >= 1


def _ranks_with_ties(values: list[float]) -> list[float]:
    """Average ranks (1-based) for possibly-tied absolute differences."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def wilcoxon_signed_rank(a: list[float], b: list[float]) -> WilcoxonResult | None:
    """Paired two-sided test of a vs b. None when there are no non-zero diffs."""
    if len(a) != len(b):
        raise ValueError(f"paired samples differ in length: {len(a)} vs {len(b)}")
    diffs = [x - y for x, y in zip(a, b, strict=True) if x != y]
    n = len(diffs)
    if n == 0:
        return None

    abs_diffs = [abs(d) for d in diffs]
    ranks = _ranks_with_ties(abs_diffs)
    w_plus = sum(rank for rank, diff in zip(ranks, diffs, strict=True) if diff > 0)
    w_minus = sum(rank for rank, diff in zip(ranks, diffs, strict=True) if diff < 0)
    w = min(w_plus, w_minus)

    mu = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24
    # Tie correction: group sizes of tied absolute differences.
    seen: dict[float, int] = {}
    for value in abs_diffs:
        seen[value] = seen.get(value, 0) + 1
    variance -= sum(t**3 - t for t in seen.values() if t > 1) / 48
    if variance <= 0:
        return WilcoxonResult(n=n, statistic=w, z=0.0, p_value=1.0, small_sample=n < 10)

    sigma = math.sqrt(variance)
    z = (w - mu + 0.5) / sigma  # continuity correction toward the mean
    p = math.erfc(abs(z) / math.sqrt(2))  # two-sided
    return WilcoxonResult(n=n, statistic=w, z=z, p_value=min(1.0, p), small_sample=n < 10)
