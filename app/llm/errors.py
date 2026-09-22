"""Provider-independent LLM errors. Providers map their SDK errors onto these."""

from app.core.errors import AppError, ConfigurationError, PermanentError, TransientError
from app.llm.base import LLMUsage


class LLMError(AppError):
    """Base class for LLM failures."""


class LLMConfigurationError(LLMError, ConfigurationError):
    """LLM configuration is missing or invalid (e.g. GEMINI_API_KEY is not set)."""


class LLMAuthenticationError(LLMError, PermanentError):
    """The provider rejected the credentials (HTTP 401/403)."""


class LLMInvalidRequestError(LLMError, PermanentError):
    """The provider rejected the request (HTTP 400/404, e.g. an unknown model)."""


class LLMBillingError(LLMError, PermanentError):
    """The account cannot pay for the call (HTTP 402: no credits, billing suspended).

    Separate from :class:`LLMInvalidRequestError` because the request itself was fine: only
    another provider (or a human topping the account up) can make the call succeed.
    """


class LLMRequestRejectedError(LLMError, PermanentError):
    """The provider rejected one particular request (HTTP 400) over what was in it.

    Deliberately **not** an ``LLMInvalidRequestError``: that one means the setup is wrong
    (an unknown model, a malformed configuration) and every later call fails the same way.
    This one is about this call's content - the web pages a built-in tool pulled into it,
    or output it could not parse - and Gemini answers it non-deterministically: the same
    URLs are accepted on one attempt and rejected ("Request contains an invalid argument",
    "Request blocked due to copyright/recitation content") on the next. The caller drops
    what it sent and carries on.
    """


class LLMRateLimitError(LLMError, TransientError):
    """Rate limited (HTTP 429) after the SDK's own retries."""


class LLMUnavailableError(LLMError, TransientError):
    """5xx, timeout or connection failure after the SDK's own retries."""


class LLMResponseError(LLMError, PermanentError):
    """The response was unusable: failed or blocked, empty, or invalid structured output.

    ``usage`` is set when the provider billed the call anyway (e.g. invalid JSON), so cost
    tracking stays accurate.
    """

    def __init__(self, message: str, *, usage: LLMUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class LLMBudgetExceededError(LLMError, TransientError):
    """A configured token budget (per run or per day) would be exceeded; no call was made."""
