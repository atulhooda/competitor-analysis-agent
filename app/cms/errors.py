"""CMS errors (Phase 7). Callers decide from the type: transient errors (timeouts, network
failures, rate limits, server errors) may be retried; permanent ones (credentials,
permissions, invalid requests) never are. ``outcome_unknown`` marks a change request that may
have taken effect although no answer came back: the post is looked up before any retry.

Messages never contain credentials: they are built here from the method, path, status and
the CMS's own error code and message.
"""

from typing import Any

from app.core.errors import AppError, ConfigurationError, PermanentError, TransientError


class CMSError(AppError):
    def __init__(self, message: str, *, status: int | None = None, code: str | None = None, outcome_unknown: bool = False, retry_after: float | None = None, data: dict[str, Any] | None = None) -> None:  # fmt: skip
        super().__init__(message)
        self.status = status
        self.code = code
        self.outcome_unknown = outcome_unknown
        self.retry_after = retry_after
        self.data = data or {}


class CMSTransientError(CMSError, TransientError):
    pass


class CMSPermanentError(CMSError, PermanentError):
    pass


class CMSConfigurationError(CMSPermanentError, ConfigurationError):
    """The CMS isn't configured (URL, user, application password)."""


class CMSReadOnlyError(CMSPermanentError):
    """A change was attempted through a read-only (dry-run) client."""


class CMSAuthError(CMSPermanentError):
    pass


class CMSPermissionError(CMSPermanentError):
    pass


class CMSValidationError(CMSPermanentError):
    pass


class CMSNotFoundError(CMSPermanentError):
    pass


class CMSConflictError(CMSPermanentError):
    pass


class CMSResponseError(CMSPermanentError):
    """A response that isn't what the API promises (not JSON, missing fields, a redirect)."""


class CMSDeploymentError(CMSPermanentError):
    """The site's build or deployment of the change failed (GitHub publishing)."""


class CMSProtectedError(CMSPermanentError):
    """A deployment can't be verified because access to it is protected (Vercel
    Deployment Protection): a person must decide, nothing is bypassed."""


class CMSRateLimitError(CMSTransientError):
    pass


class CMSTimeoutError(CMSTransientError):
    pass


class CMSNetworkError(CMSTransientError):
    pass


class CMSServerError(CMSTransientError):
    pass
