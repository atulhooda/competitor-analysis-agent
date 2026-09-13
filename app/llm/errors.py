"""Provider-independent LLM errors. Providers map their SDK errors onto these."""

from app.core.errors import AppError, ConfigurationError, PermanentError, TransientError


class LLMError(AppError):
    """Base class for LLM failures."""


class LLMConfigurationError(LLMError, ConfigurationError):
    """LLM configuration is missing or invalid (e.g. GEMINI_API_KEY is not set)."""


class LLMAuthenticationError(LLMError, PermanentError):
    """The provider rejected the credentials (HTTP 401/403)."""


class LLMInvalidRequestError(LLMError, PermanentError):
    """The provider rejected the request (HTTP 400/404, e.g. an unknown model)."""


class LLMRateLimitError(LLMError, TransientError):
    """Rate limited (HTTP 429) after the SDK's own retries."""


class LLMUnavailableError(LLMError, TransientError):
    """5xx, timeout or connection failure after the SDK's own retries."""


class LLMResponseError(LLMError, PermanentError):
    """The response was unusable: failed or blocked, empty, or invalid structured output."""
