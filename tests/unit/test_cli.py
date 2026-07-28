"""CLI behavior, including the missing-configuration failure mode."""

from pathlib import Path

import pytest

from agentic_rag_bench import __version__
from agentic_rag_bench.cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestInfo:
    def test_info_succeeds(self):
        assert main(["info"]) == 0


class TestVersion:
    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        assert __version__ in capsys.readouterr().out


class TestValidateConfig:
    def test_repository_configs_are_valid(self):
        for name in ("default.yaml", "standard_rag.yaml", "agentic_rag.yaml"):
            config = REPO_ROOT / "configs" / name
            defaults = REPO_ROOT / "configs" / "default.yaml"
            assert main(["validate-config", str(config), "--defaults", str(defaults)]) == 0

    def test_missing_config_fails_nonzero(self, tmp_path):
        assert main(["validate-config", str(tmp_path / "missing.yaml")]) == 1

    def test_invalid_config_fails_nonzero(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: x\nsystem_type: not_a_system\n", encoding="utf-8")
        assert main(["validate-config", str(bad)]) == 1

    def test_no_command_exits_with_usage_error(self):
        with pytest.raises(SystemExit) as excinfo:
            main([])
        assert excinfo.value.code == 2
