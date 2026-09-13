"""Structured logging (structlog) with secret redaction.

Logs go to stderr so CLI commands can print machine-readable output on stdout. The
stream is looked up at write time rather than captured at configuration time, so logging
keeps working when stderr is swapped (test runners, daemonizers, output redirection).
"""

import logging
import sys
from collections.abc import MutableMapping
from typing import Any, TextIO

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


class _CurrentStderrLogger:
    """structlog output that writes each line to whatever ``sys.stderr`` is right now."""

    def msg(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    debug = info = warning = warn = error = critical = exception = fatal = log = msg


def _stderr_logger_factory(*_args: Any) -> _CurrentStderrLogger:
    return _CurrentStderrLogger()


class _CurrentStderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """stdlib handler bound to the current ``sys.stderr`` at emit time."""

    @property
    def stream(self) -> TextIO:
        return sys.stderr

    @stream.setter
    def stream(self, _value: TextIO) -> None:
        pass


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
        logger_factory=_stderr_logger_factory,
        cache_logger_on_first_use=False,
    )
    handler = _CurrentStderrHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=numeric_level, handlers=[handler])
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(numeric_level, logging.WARNING))
