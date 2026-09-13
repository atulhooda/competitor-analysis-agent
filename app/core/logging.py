"""Structured logging (structlog) with secret redaction.

Logs go to stderr so CLI commands can print machine-readable output on stdout.
"""

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

_SENSITIVE_FRAGMENTS = ("token", "secret", "password", "authorization")
_NOISY_LOGGERS = ("httpx", "httpcore", "trafilatura", "htmldate", "charset_normalizer", "google")


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return lowered.endswith(("_key", "apikey")) or any(f in lowered for f in _SENSITIVE_FRAGMENTS)


def _redact_secrets(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key in list(event_dict):
        if _is_sensitive(key):
            event_dict[key] = "***"
    return event_dict


def configure_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    """Configure structlog and the stdlib root logger. Safe to call more than once."""
    numeric_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact_secrets,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )
    logging.basicConfig(
        level=numeric_level, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s"
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(numeric_level, logging.WARNING))
