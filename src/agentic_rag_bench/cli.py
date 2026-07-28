"""Command-line entry point (``arb``).

Phase 1 exposes project introspection and config validation. Pipeline
commands (ingest, run, evaluate, report) are added by their phases.
"""

import argparse
import sys
from pathlib import Path

from agentic_rag_bench import __version__
from agentic_rag_bench.config import AppSettings, ConfigError, load_system_config
from agentic_rag_bench.observability import configure_logging, get_logger

DEFAULT_CONFIG = Path("configs/default.yaml")


def _cmd_info(_: argparse.Namespace) -> int:
    settings = AppSettings()
    logger = get_logger("cli")
    logger.info(
        "agentic-rag-bench",
        version=__version__,
        data_dir=str(settings.data_dir),
        reports_dir=str(settings.reports_dir),
        log_level=settings.log_level,
        anthropic_key_configured=settings.anthropic_api_key is not None,
        openai_key_configured=settings.openai_api_key is not None,
    )
    return 0


def _cmd_validate_config(args: argparse.Namespace) -> int:
    logger = get_logger("cli")
    defaults = args.defaults if args.defaults is not None else DEFAULT_CONFIG
    defaults_path = defaults if defaults.is_file() else None
    try:
        config = load_system_config(args.config, defaults_path=defaults_path)
    except ConfigError as exc:
        logger.error("config_invalid", config=str(args.config), error=str(exc))
        return 1
    logger.info(
        "config_valid",
        config=str(args.config),
        name=config.name,
        system_type=config.system_type.value,
        top_k=config.retrieval.top_k,
        reranker_enabled=config.reranker.enabled,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arb",
        description="Agentic-RAG-Bench: compare standard and agentic RAG pipelines.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--log-level", default=None, help="Override ARB_LOG_LEVEL")
    parser.add_argument(
        "--log-format", default=None, choices=("console", "json"), help="Override ARB_LOG_FORMAT"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="Show version and resolved settings")
    info.set_defaults(func=_cmd_info)

    validate = subparsers.add_parser("validate-config", help="Validate a system config file")
    validate.add_argument("config", type=Path, help="Path to the system YAML config")
    validate.add_argument(
        "--defaults",
        type=Path,
        default=None,
        help=f"Shared defaults overlay (default: {DEFAULT_CONFIG} when present)",
    )
    validate.set_defaults(func=_cmd_validate_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = AppSettings()
    configure_logging(
        level=args.log_level or settings.log_level,
        log_format=args.log_format or settings.log_format,
    )
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
