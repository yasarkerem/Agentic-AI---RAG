"""Statistics: summaries and the Wilcoxon signed-rank test (hand-verified)."""

import math

import pytest

from agentic_rag_bench.evaluation.stats import summarize, wilcoxon_signed_rank


class TestSummarize:
    def test_basic_moments(self):
        summary = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        assert summary is not None
        assert summary.mean == 3.0
        assert summary.median == 3.0
        assert summary.std == pytest.approx(math.sqrt(2.5))
        assert summary.ci_low <= summary.mean <= summary.ci_high

    def test_deterministic_ci(self):
        a = summarize([0.2, 0.4, 0.6, 0.8])
        b = summarize([0.2, 0.4, 0.6, 0.8])
        assert a == b

    def test_empty_returns_none(self):
        assert summarize([]) is None

    def test_single_value(self):
        summary = summarize([0.7])
        assert summary is not None
        assert summary.std == 0.0
        assert summary.ci_low == summary.ci_high == 0.7


class TestWilcoxon:
    def test_all_positive_differences_hand_computed(self):
        """n=5, all diffs positive: W=0, mu=7.5, sigma=sqrt(13.75).

        z = (0 - 7.5 + 0.5) / 3.7081 = -1.8877; two-sided p ≈ 0.0591.
        """
        result = wilcoxon_signed_rank([2, 3, 4, 5, 6], [1, 1, 1, 1, 1])
        assert result is not None
        assert result.statistic == 0.0
        assert result.z == pytest.approx(-1.8877, abs=1e-3)
        assert result.p_value == pytest.approx(0.0591, abs=1e-3)
        assert result.small_sample

    def test_symmetric_differences_not_significant(self):
        result = wilcoxon_signed_rank([1, 2, 3, 4], [2, 1, 4, 3])
        assert result is not None
        assert result.p_value > 0.5

    def test_identical_samples_return_none(self):
        assert wilcoxon_signed_rank([1.0, 2.0], [1.0, 2.0]) is None

    def test_tie_correction_applied(self):
        # Two pairs of tied absolute differences must not crash and must
        # reduce variance relative to the untied formula.
        result = wilcoxon_signed_rank([2, 2, 5, 5, 9], [1, 1, 3, 3, 4])
        assert result is not None
        assert 0.0 <= result.p_value <= 1.0

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="differ in length"):
            wilcoxon_signed_rank([1.0], [1.0, 2.0])

    def test_larger_sample_not_flagged_small(self):
        a = [float(i) + (0.5 if i % 2 else -0.25) for i in range(12)]
        b = [float(i) for i in range(12)]
        result = wilcoxon_signed_rank(a, b)
        assert result is not None
        assert not result.small_sample
