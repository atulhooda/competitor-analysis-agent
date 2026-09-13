"""Error taxonomy shared across the application.

Callers decide whether to retry from the exception *type*, never from message text:
transient errors may succeed later (timeouts, 429, 5xx); permanent errors will not
succeed without a change (4xx, validation, robots.txt disallow).
"""


class AppError(Exception):
    """Base class for all application errors."""


class TransientError(AppError):
    """A failure that may succeed if retried later."""


class PermanentError(AppError):
    """A failure that will not succeed on retry without a change."""


class ConfigurationError(PermanentError):
    """Required configuration is missing or invalid."""
