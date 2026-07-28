"""Configuration loading, merging, and failure modes."""

from pathlib import Path

import pytest

from agentic_rag_bench.config import (
    ChunkingConfig,
    ConfigError,
    RetrievalConfig,
    deep_merge,
    load_system_config,
)
from agentic_rag_bench.schemas import SystemType

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"


class TestRepositoryConfigs:
    """The configs shipped in the repository must always load."""

    def test_default_config_loads(self):
        config = load_system_config(CONFIGS / "default.yaml")
        assert config.system_type is SystemType.STANDARD_RAG
        assert config.retrieval.top_k == 8
        assert config.agent is None

    def test_standard_rag_config_loads(self):
        config = load_system_config(
            CONFIGS / "standard_rag.yaml", defaults_path=CONFIGS / "default.yaml"
        )
        assert config.name == "standard-rag-dense"
        assert config.system_type is SystemType.STANDARD_RAG
        assert config.agent is None
        # Inherited from the defaults overlay
        assert config.chunking.chunk_size == 512

    def test_agentic_rag_config_loads(self):
        config = load_system_config(
            CONFIGS / "agentic_rag.yaml", defaults_path=CONFIGS / "default.yaml"
        )
        assert config.system_type is SystemType.AGENTIC_RAG
        assert config.agent is not None
        assert config.agent.max_retrieval_rounds == 3
        assert config.agent.max_subquestions == 6

    def test_fairness_controls_identical_across_systems(self):
        """Embedding, chunking, and generation must match between systems (brief §2)."""
        standard = load_system_config(
            CONFIGS / "standard_rag.yaml", defaults_path=CONFIGS / "default.yaml"
        )
        agentic = load_system_config(
            CONFIGS / "agentic_rag.yaml", defaults_path=CONFIGS / "default.yaml"
        )
        assert standard.embedding == agentic.embedding
        assert standard.chunking == agentic.chunking
        assert standard.generation == agentic.generation
        assert standard.seed == agentic.seed


class TestFailureModes:
    def test_missing_file_raises_config_error(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_system_config(tmp_path / "nope.yaml")

    def test_invalid_yaml_raises_config_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_system_config(bad)

    def test_non_mapping_yaml_raises_config_error(self, tmp_path):
        bad = tmp_path / "list.yaml"
        bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="must be a mapping"):
            load_system_config(bad)

    def test_unknown_key_rejected(self, tmp_path):
        config = tmp_path / "extra.yaml"
        config.write_text("name: x\nsystem_type: standard_rag\ntypo_field: 1\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="typo_field"):
            load_system_config(config)

    def test_agentic_without_agent_section_rejected(self, tmp_path):
        config = tmp_path / "agentic.yaml"
        config.write_text("name: x\nsystem_type: agentic_rag\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="require an 'agent' section"):
            load_system_config(config)

    def test_standard_with_agent_section_rejected(self, tmp_path):
        config = tmp_path / "standard.yaml"
        config.write_text(
            "name: x\nsystem_type: standard_rag\nagent:\n  max_retrieval_rounds: 2\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="must not define an 'agent' section"):
            load_system_config(config)


class TestValidationRules:
    def test_chunk_overlap_must_be_smaller_than_size(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

    def test_chunk_overlap_valid(self):
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        assert config.chunk_overlap == 20

    def test_hybrid_weight_requires_bm25(self):
        with pytest.raises(ValueError, match=r"dense_weight must be 1\.0"):
            RetrievalConfig(use_bm25=False, dense_weight=0.5)

    def test_hybrid_weight_with_bm25_valid(self):
        config = RetrievalConfig(use_bm25=True, dense_weight=0.7)
        assert config.dense_weight == 0.7

    def test_dense_weight_bounds(self):
        with pytest.raises(ValueError):
            RetrievalConfig(use_bm25=True, dense_weight=1.5)


class TestDeepMerge:
    def test_override_wins_and_defaults_survive(self):
        base = {"retrieval": {"top_k": 8, "use_bm25": False}, "seed": 42}
        override = {"retrieval": {"top_k": 20}}
        merged = deep_merge(base, override)
        assert merged["retrieval"]["top_k"] == 20
        assert merged["retrieval"]["use_bm25"] is False
        assert merged["seed"] == 42

    def test_inputs_not_mutated(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        deep_merge(base, override)
        assert base == {"a": {"b": 1}}
        assert override == {"a": {"c": 2}}

    def test_override_replaces_non_dict_values(self):
        merged = deep_merge({"a": {"b": 1}}, {"a": "flat"})
        assert merged == {"a": "flat"}

    def test_yaml_override_merging_end_to_end(self, tmp_path):
        defaults = tmp_path / "default.yaml"
        defaults.write_text(
            "name: base\nsystem_type: standard_rag\nretrieval:\n  top_k: 8\n",
            encoding="utf-8",
        )
        overlay = tmp_path / "overlay.yaml"
        overlay.write_text("name: tuned\nretrieval:\n  top_k: 20\n", encoding="utf-8")
        config = load_system_config(overlay, defaults_path=defaults)
        assert config.name == "tuned"
        assert config.retrieval.top_k == 20
        assert config.chunking.chunk_size == 512  # model default still applies
