"""Token accounting and cost calculation."""

import pytest
from pydantic import ValidationError

from agentic_rag_bench.schemas import ModelPricing, TokenUsage, calculate_cost


class TestTokenUsage:
    def test_total(self):
        assert TokenUsage(input_tokens=7, output_tokens=3).total_tokens == 10

    def test_addition_accumulates(self):
        total = TokenUsage(input_tokens=100, output_tokens=20) + TokenUsage(
            input_tokens=50, output_tokens=30
        )
        assert total == TokenUsage(input_tokens=150, output_tokens=50)

    def test_negative_tokens_rejected(self):
        with pytest.raises(ValidationError):
            TokenUsage(input_tokens=-1, output_tokens=0)


class TestCalculateCost:
    PRICING = ModelPricing(model="example", input_usd_per_mtok=3.0, output_usd_per_mtok=15.0)

    def test_exact_cost(self):
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=200_000)
        assert calculate_cost(usage, self.PRICING) == pytest.approx(3.0 + 0.2 * 15.0)

    def test_zero_usage_costs_nothing(self):
        assert calculate_cost(TokenUsage(), self.PRICING) == 0.0

    def test_small_usage_not_rounded_away(self):
        usage = TokenUsage(input_tokens=100, output_tokens=10)
        expected = (100 * 3.0 + 10 * 15.0) / 1_000_000
        assert calculate_cost(usage, self.PRICING) == pytest.approx(expected)

    def test_negative_pricing_rejected(self):
        with pytest.raises(ValidationError):
            ModelPricing(model="bad", input_usd_per_mtok=-1.0, output_usd_per_mtok=0.0)
