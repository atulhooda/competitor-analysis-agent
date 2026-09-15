"""Which job failures are retried, and when (Phase 8).

- **Transient** (network, a dropped database connection, Gemini unavailable or rate limited,
  WordPress 429/5xx): the job is requeued with exponential backoff, JOB_RETRY_BASE_SECONDS
  doubling per attempt up to JOB_RETRY_MAX_SECONDS, JOB_MAX_ATTEMPTS attempts in all. The
  retry continues from the job's checkpoints: finished stages aren't run again.
- **Permanent** (configuration, credentials, invalid input, a failed quality gate, a
  programming error): never retried. A person fixes the cause, then retries the job.
- **Budget** (an LLM token budget is spent): stopped and not retried; the next scheduled run
  continues once the budget allows it.

The clients keep their own short retries (the Gemini SDK, the WordPress client, the
fetcher). A job retry is one delayed requeue, never a loop around a call that already
retries, and never a blind repeat of a CMS write: Phase 7 reconciles with the CMS first.

Failures are classified by exception type, never by message wording. For failures a service
reports in its outcome instead of raising, the outcome's ``ExceptionName: message`` prefix
names the type.
"""

from collections.abc import Iterator

import httpx
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

from app.core.errors import AppError, PermanentError, TransientError
from app.domain.jobs import ErrorKind
from app.llm.errors import LLMBudgetExceededError
from app.services.llm_usage import BUDGET_REACHED

_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
    OperationalError,
    InterfaceError,
    TimeoutError,
    ConnectionError,
    httpx.TransportError,
)


def classify_type(kind: type[BaseException]) -> ErrorKind:
    if issubclass(kind, LLMBudgetExceededError):
        return ErrorKind.BUDGET
    if issubclass(kind, PermanentError):
        return ErrorKind.PERMANENT
    if issubclass(kind, TransientError) or issubclass(kind, _TRANSIENT_TYPES):
        return ErrorKind.TRANSIENT
    return ErrorKind.PERMANENT  # a bug or invalid data: retrying won't help


def classify(exc: BaseException) -> ErrorKind:
    if isinstance(exc, DBAPIError) and exc.connection_invalidated:
        return ErrorKind.TRANSIENT
    return classify_type(type(exc))


def _subclasses(kind: type[BaseException]) -> Iterator[type[BaseException]]:
    for sub in kind.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def _known_types() -> dict[str, type[BaseException]]:
    known: dict[str, type[BaseException]] = {}
    roots: tuple[type[BaseException], ...] = (AppError, *_TRANSIENT_TYPES)
    for root in roots:
        known.setdefault(root.__name__, root)
        for sub in _subclasses(root):
            known.setdefault(sub.__name__, sub)
    return known


def named_type(error: str | None) -> type[BaseException] | None:
    """The exception type an ``ExceptionName: message`` error names, if it is a known one."""
    if not error:
        return None
    return _known_types().get(error.split(":", 1)[0].strip())


def classify_text(error: str | None) -> ErrorKind | None:
    """The kind named by an outcome's ``ExceptionName: message`` error, or None when it
    names no known type."""
    if error and BUDGET_REACHED in error:
        return ErrorKind.BUDGET
    kind = named_type(error)
    return classify_type(kind) if kind is not None else None


def backoff_seconds(attempt: int, *, base: int, cap: int) -> int:
    """The wait after failed attempt number ``attempt`` (1, 2, …): base, 2 x base, 4 x base…,
    never more than ``cap``."""
    return int(min(base * 2 ** max(attempt - 1, 0), cap))


def retryable(kind: ErrorKind | None) -> bool:
    return kind in (ErrorKind.TRANSIENT, ErrorKind.INTERRUPTED)


__all__ = [
    "backoff_seconds",
    "classify",
    "classify_text",
    "classify_type",
    "named_type",
    "retryable",
]
