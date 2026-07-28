"""Structured logging: secret redaction and output shape."""

import json

import pytest
from pydantic import SecretStr

from agentic_rag_bench.observability import REDACTED, configure_logging, redact_sensitive
from agentic_rag_bench.observability.logs import _is_sensitive_key


def _run_processor(event_dict):
    return redact_sensitive(None, "info", dict(event_dict))


class TestRedaction:
    @pytest.mark.parametrize(
        "key",
        ["api_key", "API_KEY", "anthropic_api_key", "token", "auth_token", "password", "secret"],
    )
    def test_sensitive_keys_redacted(self, key):
        result = _run_processor({"event": "call", key: "sk-super-secret-value"})
        assert result[key] == REDACTED

    def test_normal_fields_untouched(self):
        event = {"event": "retrieval", "query_id": "q1", "top_k": 8, "latency": 0.42}
        assert _run_processor(event) == event

    def test_token_count_fields_not_redacted(self):
        """'token' is sensitive, but token *counts* are core metrics (brief §15)."""
        event = {"event": "llm_call", "input_tokens": 120, "output_tokens": 30}
        result = _run_processor(event)
        assert result["input_tokens"] == 120
        assert result["output_tokens"] == 30

    def test_nested_mapping_redacted(self):
        result = _run_processor({"event": "cfg", "provider": {"name": "x", "api_key": "sk-123"}})
        assert result["provider"]["api_key"] == REDACTED
        assert result["provider"]["name"] == "x"

    def test_secretstr_values_redacted_anywhere(self):
        result = _run_processor({"event": "cfg", "credential_ref": SecretStr("sk-123")})
        assert result["credential_ref"] == REDACTED

    def test_list_of_mappings_redacted(self):
        result = _run_processor({"event": "cfg", "providers": [{"token": "t1"}, {"ok": 1}]})
        assert result["providers"][0]["token"] == REDACTED
        assert result["providers"][1]["ok"] == 1


class TestSensitiveKeyMatcher:
    def test_substring_matches(self):
        assert _is_sensitive_key("MY_SERVICE_API_KEY")
        assert _is_sensitive_key("Authorization")

    def test_token_counts_are_safe(self):
        assert not _is_sensitive_key("input_tokens")
        assert not _is_sensitive_key("max_total_tokens")


class TestConfigureLogging:
    def test_rejects_unknown_format(self):
        with pytest.raises(ValueError, match="log_format"):
            configure_logging(log_format="xml")

    def test_json_output_is_parseable_and_redacted(self, capsys):
        configure_logging(level="INFO", log_format="json")
        from agentic_rag_bench.observability import get_logger

        get_logger("test").info("provider_call", api_key="sk-123", query_id="q1")
        captured = capsys.readouterr()
        line = captured.err.strip().splitlines()[-1]
        payload = json.loads(line)
        assert payload["event"] == "provider_call"
        assert payload["api_key"] == REDACTED
        assert payload["query_id"] == "q1"
        assert "timestamp" in payload
        assert payload["level"] == "info"
