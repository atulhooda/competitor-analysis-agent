"""SSRF guard: refuse to fetch hosts that resolve to non-public addresses.

Known limitation: the check and the connection resolve DNS separately, so a DNS
rebinding attack could slip between them. Acceptable for fetching configured
competitor sites; revisit before accepting arbitrary user-supplied URLs.
"""

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable

from app.crawling.errors import NetworkFetchError, UnsafeDestinationError

Resolver = Callable[[str], Awaitable[list[str]]]

_BLOCKED_SUFFIXES = (".localhost", ".local", ".internal")


async def system_resolver(host: str) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]) for info in infos})


async def ensure_public_destination(url: str, host: str, resolver: Resolver) -> None:
    """Raise ``UnsafeDestinationError`` unless every address for ``host`` is public."""
    if host == "localhost" or host.endswith(_BLOCKED_SUFFIXES):
        raise UnsafeDestinationError(url, f"{host} is a local hostname")
    try:
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        try:
            addresses = await resolver(host)
        except OSError as exc:
            raise NetworkFetchError(url, f"DNS lookup failed for {host}: {exc}") from exc
    if not addresses:
        raise NetworkFetchError(url, f"DNS lookup returned no addresses for {host}")
    for address in addresses:
        if not ipaddress.ip_address(address.split("%", 1)[0]).is_global:
            raise UnsafeDestinationError(url, f"{host} resolves to non-public address {address}")
