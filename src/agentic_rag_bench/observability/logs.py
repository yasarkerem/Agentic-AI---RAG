"""Structured logging built on structlog.

Every log line is a structured event; callers bind context such as
``query_id``, ``experiment_id``, ``system_type``, or ``retrieval_round``.
A redaction processor guarantees that secret-like fields never reach the
output stream (brief §15: do not log secrets or full API keys).
"""

import logging
import sys
from collections.abc import Mapping
from typing import Any

import structlog
from pydantic import SecretStr

REDACTED = "***REDACTED***"

_SENSITIVE_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)

_SAFE_TOKEN_KEYS = frozenset(
    {"input_tokens", "output_tokens", "total_tokens", "max_total_tokens", "context_token_limit"}
)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in _SAFE_TOKEN_KEYS:
        return False
    return any(marker in lowered for marker in _SENSITIVE_MARKERS)


def _redact_value(value: Any) -> Any:
    if isinstance(value, SecretStr):
        return REDACTED
    if isinstance(value, Mapping):
        return {
            key: REDACTED if _is_sensitive_key(str(key)) else _redact_value(inner)
            for key, inner in value.items()
        }
    if isinstance(value, list | tuple):
        return [_redact_value(item) for item in value]
    return value


def redact_sensitive(
    logger: structlog.typing.WrappedLogger,
    method_name: str,
    event_dict: structlog.typing.EventDict,
) -> structlog.typing.EventDict:
    """structlog processor: mask values under secret-like keys, recursively."""
    for key in list(event_dict):
        if _is_sensitive_key(key):
            event_dict[key] = REDACTED
        else:
            event_dict[key] = _redact_value(event_dict[key])
    return event_dict


def configure_logging(level: str = "INFO", log_format: str = "console") -> None:
    """Configure structlog + stdlib logging. ``log_format`` is 'console' or 'json'."""
    if log_format not in ("console", "json"):
        raise ValueError(f"log_format must be 'console' or 'json', got {log_format!r}")

    renderer: structlog.typing.Processor
    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_sensitive,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """Return a named structured logger."""
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(name)
    return logger
