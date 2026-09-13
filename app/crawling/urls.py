"""URL normalization and competitor scope checks (deterministic, no network)."""

import ipaddress
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

_TRACKING_PARAMS = frozenset(
    {"gclid", "fbclid", "msclkid", "dclid", "yclid", "igshid", "mkt_tok", "hsctatracking"}
)
_TRACKING_PREFIXES = ("utm_", "mc_", "_hs", "hsa_")
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _is_tracking_param(name: str) -> bool:
    lowered = name.lower()
    return lowered in _TRACKING_PARAMS or lowered.startswith(_TRACKING_PREFIXES)


def normalize_url(url: str, base: str | None = None) -> str | None:
    """Return a canonical absolute http(s) URL, or ``None`` if the URL is not crawlable.

    Lowercases scheme and host, IDNA-encodes the host, drops credentials, default
    ports, fragments and tracking parameters. The path is left as-is: a trailing
    slash can be meaningful.
    """
    raw = url.strip()
    if not raw:
        return None
    if base:
        raw = urljoin(base, raw)
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS:
        return None
    host = (parts.hostname or "").rstrip(".")
    if not host:
        return None
    try:
        host = host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return None
    netloc = f"[{host}]" if _is_ipv6(host) else host
    if port is not None and port != _DEFAULT_PORTS[scheme]:
        netloc = f"{netloc}:{port}"
    query = parts.query
    if query:
        pairs = parse_qsl(query, keep_blank_values=True)
        kept = [(k, v) for k, v in pairs if not _is_tracking_param(k)]
        if len(kept) != len(pairs):
            query = urlencode(kept)
    return urlunsplit((scheme, netloc, parts.path or "/", query, ""))


def _is_ipv6(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).version == 6
    except ValueError:
        return False


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def origin_of(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _base_domain(host: str) -> str:
    return host.lower().rstrip(".").removeprefix("www.")


@dataclass(frozen=True)
class SiteScope:
    """Hosts a competitor scan may touch: each base domain and its subdomains."""

    base_domains: frozenset[str]

    @classmethod
    def from_urls(cls, urls: Iterable[str], extra_domains: Iterable[str] = ()) -> "SiteScope":
        domains = {_base_domain(host_of(u)) for u in urls if host_of(u)}
        domains.update(_base_domain(d) for d in extra_domains if d)
        return cls(frozenset(domains))

    def contains(self, url: str) -> bool:
        host = host_of(url)
        return any(host == d or host.endswith("." + d) for d in self.base_domains)
