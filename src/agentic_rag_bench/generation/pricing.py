"""Model pricing for cost estimation (USD per million tokens).

Cached from Anthropic's published API pricing (2026-06). Unknown models fall
back to zero-cost with a warning, so experiments never crash on a missing
price — but the gap is visible in logs rather than silently mispriced.
"""

from agentic_rag_bench.observability import get_logger
from agentic_rag_bench.schemas import ModelPricing, TokenUsage, calculate_cost

logger = get_logger("pricing")

PRICING_VERSION = "2026-06"

_PRICES: dict[str, tuple[float, float]] = {
    # model: (input USD/MTok, output USD/MTok)
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def pricing_for(model: str) -> ModelPricing:
    """Pricing for a model; mock/unknown models are zero-cost."""
    if model in _PRICES:
        input_price, output_price = _PRICES[model]
        return ModelPricing(
            model=model, input_usd_per_mtok=input_price, output_usd_per_mtok=output_price
        )
    if not model.startswith("mock"):
        logger.warning("unknown_model_pricing", model=model, assumed="zero cost")
    return ModelPricing(model=model, input_usd_per_mtok=0.0, output_usd_per_mtok=0.0)


def estimate_cost(model: str, usage: TokenUsage) -> float:
    """Estimated USD cost of one usage record."""
    return calculate_cost(usage, pricing_for(model))
