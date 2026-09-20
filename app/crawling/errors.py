"""Fetch and parse errors. Each carries a stable ``code`` used in scan reports."""

from app.core.errors import AppError, PermanentError, TransientError


class FetchError(AppError):
    code = "fetch_error"

    def __init__(self, url: str, detail: str = "") -> None:
        self.url = url
        self.detail = detail
        super().__init__(f"{self.code}: {url}" + (f" ({detail})" if detail else ""))


class InvalidUrlError(FetchError, PermanentError):
    code = "invalid_url"


class OutOfScopeError(FetchError, PermanentError):
    code = "out_of_scope"


class UnsafeDestinationError(FetchError, PermanentError):
    code = "unsafe_destination"


class RobotsDisallowedError(FetchError, PermanentError):
    code = "robots_disallowed"


class BlockedError(FetchError, PermanentError):
    """401/403 or a bot challenge. We never try to get around it."""

    code = "blocked"


class HTTPStatusError(FetchError):
    code = "http_error"

    def __init__(self, url: str, status: int, detail: str = "") -> None:
        self.status = status
        super().__init__(url, f"HTTP {status}" + (f"; {detail}" if detail else ""))


class ClientStatusError(HTTPStatusError, PermanentError):
    code = "http_client_error"


class ServerStatusError(HTTPStatusError, TransientError):
    code = "http_server_error"


class TooManyRedirectsError(FetchError, PermanentError):
    code = "too_many_redirects"


class ResponseTooLargeError(FetchError, PermanentError):
    code = "response_too_large"


class UnsupportedContentTypeError(FetchError, PermanentError):
    code = "unsupported_content_type"


class NetworkFetchError(FetchError, TransientError):
    code = "network_error"


class ContentParseError(PermanentError):
    """A sitemap, feed or page could not be parsed."""

    code = "parse_error"

    def __init__(self, url: str, detail: str) -> None:
        self.url = url
        self.detail = detail
        super().__init__(f"{self.code}: {url} ({detail})")
