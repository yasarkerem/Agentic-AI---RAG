"""LLM provider abstraction: mock behavior, pricing, registry, payloads."""

import json

import pytest

from agentic_rag_bench.config import AppSettings, ConfigError, GenerationConfig
from agentic_rag_bench.generation import (
    MockLLMProvider,
    build_llm_provider,
    embed_payload,
    estimate_cost,
    extract_payload,
    pricing_for,
)
from agentic_rag_bench.schemas import TokenUsage


class TestPayload:
    def test_round_trip(self):
        prompt = embed_payload("Do the thing.", {"question": "q?", "n": 3})
        assert extract_payload(prompt) == {"question": "q?", "n": 3}

    def test_missing_payload_returns_empty(self):
        assert extract_payload("no payload here") == {}

    def test_malformed_payload_returns_empty(self):
        assert extract_payload("<payload>{broken</payload>") == {}


class TestMockProvider:
    def _complete(self, purpose, payload):
        provider = MockLLMProvider()
        return provider.complete(embed_payload("x", payload), purpose=purpose)

    def test_deterministic(self):
        a = self._complete("answer", {"question": "q?", "chunks": []})
        b = self._complete("answer", {"question": "q?", "chunks": []})
        assert a.text == b.text

    def test_analyze_classifies_comparison_as_complex(self):
        response = self._complete("analyze", {"question": "Compare GraphRAG and RAPTOR"})
        assert json.loads(response.text)["complexity"] == "complex"

    def test_analyze_classifies_lookup_as_simple(self):
        response = self._complete("analyze", {"question": "What is RAPTOR?"})
        assert json.loads(response.text)["complexity"] == "simple"

    def test_plan_splits_on_and(self):
        response = self._complete(
            "plan",
            {"question": "What is RAPTOR and what is GraphRAG?", "max_subquestions": 4},
        )
        subqs = json.loads(response.text)["subquestions"]
        assert len(subqs) == 2

    def test_plan_respects_cap(self):
        response = self._complete(
            "plan",
            {"question": "a and b and c and d and e", "max_subquestions": 2},
        )
        assert len(json.loads(response.text)["subquestions"]) == 2

    def test_assess_flags_weak_subquestions(self):
        response = self._complete(
            "assess",
            {"evidence_counts": {"a?": 5, "b?": 0}, "min_evidence": 2},
        )
        parsed = json.loads(response.text)
        assert parsed["sufficient"] is False
        assert parsed["weak_subquestions"] == ["b?"]

    def test_assess_sufficient_when_all_covered(self):
        response = self._complete(
            "assess", {"evidence_counts": {"a?": 3, "b?": 2}, "min_evidence": 2}
        )
        assert json.loads(response.text)["sufficient"] is True

    def test_answer_cites_chunk_ids(self):
        response = self._complete(
            "answer",
            {
                "question": "What is RAPTOR?",
                "chunks": [{"chunk_id": "doc_p0_c0", "text": "RAPTOR builds a tree. More."}],
            },
        )
        assert "[doc_p0_c0]" in response.text

    def test_answer_admits_missing_evidence(self):
        response = self._complete("answer", {"question": "q?", "chunks": []})
        assert "does not contain evidence" in response.text

    def test_usage_reported(self):
        response = self._complete("answer", {"question": "q?", "chunks": []})
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0


class TestPricing:
    def test_known_model_priced(self):
        pricing = pricing_for("claude-opus-5")
        assert pricing.input_usd_per_mtok == 5.00
        assert pricing.output_usd_per_mtok == 25.00

    def test_mock_model_is_free(self):
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert estimate_cost("mock-llm", usage) == 0.0

    def test_estimate_matches_manual_calculation(self):
        usage = TokenUsage(input_tokens=2_000_000, output_tokens=100_000)
        assert estimate_cost("claude-sonnet-5", usage) == pytest.approx(2 * 3.00 + 0.1 * 15.00)


class TestRegistry:
    def test_mock_provider_built(self):
        provider = build_llm_provider(GenerationConfig())
        assert provider.model_name == "mock-llm"

    def test_unknown_provider_rejected(self):
        with pytest.raises(ConfigError, match="Unknown LLM provider"):
            build_llm_provider(GenerationConfig(provider="openai", model="gpt"))

    def test_anthropic_without_key_rejected(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        settings = AppSettings(_env_file=None)
        assert settings.anthropic_api_key is None
        with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
            build_llm_provider(
                GenerationConfig(provider="anthropic", model="claude-opus-5"),
                settings=settings,
            )
