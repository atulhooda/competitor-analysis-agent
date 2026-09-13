import pytest

from app.crawling.errors import NetworkFetchError, UnsafeDestinationError
from app.crawling.netguard import ensure_public_destination


def resolver_for(*addresses: str):  # type: ignore[no-untyped-def]
    async def resolve(host: str) -> list[str]:
        return list(addresses)

    return resolve


async def test_public_address_allowed() -> None:
    await ensure_public_destination(
        "https://acme.test/", "acme.test", resolver_for("93.184.216.34")
    )


@pytest.mark.parametrize(
    "address", ["10.0.0.5", "192.168.1.10", "127.0.0.1", "169.254.169.254", "::1", "fd00::1"]
)
async def test_private_addresses_blocked(address: str) -> None:
    with pytest.raises(UnsafeDestinationError):
        await ensure_public_destination("https://acme.test/", "acme.test", resolver_for(address))


async def test_mixed_resolution_blocked_if_any_address_is_private() -> None:
    with pytest.raises(UnsafeDestinationError):
        await ensure_public_destination(
            "https://acme.test/", "acme.test", resolver_for("93.184.216.34", "10.0.0.1")
        )


@pytest.mark.parametrize("host", ["localhost", "db.internal", "printer.local", "127.0.0.1"])
async def test_local_hostnames_and_ip_literals_blocked(host: str) -> None:
    with pytest.raises(UnsafeDestinationError):
        await ensure_public_destination(f"http://{host}/", host, resolver_for("93.184.216.34"))


async def test_dns_failure_is_a_network_error() -> None:
    async def failing(host: str) -> list[str]:
        raise OSError("Name or service not known")

    with pytest.raises(NetworkFetchError):
        await ensure_public_destination("https://nope.test/", "nope.test", failing)
