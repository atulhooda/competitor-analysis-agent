"""Signing in as a GitHub App: the JWT, the installation token (cached, refreshed), and the
client sending the app's token instead of a person's. respx; no network."""

import base64
import json

import httpx
import pytest
import respx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from pydantic import SecretStr

from app.cms.errors import CMSAuthError, CMSConfigurationError
from app.cms.github import GitHubClient
from app.cms.github.app_auth import GitHubAppAuth, app_jwt, load_key

API = "https://api.github.com"
REPO = "clinic/site"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PEM = KEY.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()  # fmt: skip


def _part(segment: str) -> dict[str, object]:
    data: dict[str, object] = json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))  # fmt: skip
    return data


def test_the_jwt_is_signed_by_the_apps_key_and_short_lived() -> None:
    token = app_jwt("12345", KEY, now=1_000_000.0)
    header, claims, signature = token.split(".")
    assert _part(header) == {"alg": "RS256", "typ": "JWT"}
    assert _part(claims) == {"iat": 999_940, "exp": 1_000_540, "iss": "12345"}
    KEY.public_key().verify(base64.urlsafe_b64decode(signature + "=="), f"{header}.{claims}".encode(), padding.PKCS1v15(), hashes.SHA256())  # fmt: skip


def test_a_key_pasted_with_literal_newlines_loads_and_a_wrong_one_is_refused() -> None:
    assert isinstance(load_key(SecretStr(PEM.replace("\n", "\\n"))), rsa.RSAPrivateKey)
    with pytest.raises(CMSConfigurationError):
        load_key(SecretStr("not a key"))


@pytest.fixture
def github() -> respx.MockRouter:
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{API}/repos/{REPO}/installation").mock(return_value=httpx.Response(200, json={"id": 77}))  # fmt: skip
        router.post(f"{API}/app/installations/77/access_tokens").mock(side_effect=[httpx.Response(201, json={"token": f"ghs_token{i}", "expires_at": "2026-10-24T13:00:00Z"}) for i in range(1, 4)])  # fmt: skip
        router.get(f"{API}/repos/{REPO}").mock(return_value=httpx.Response(200, json={"full_name": REPO}))  # fmt: skip
        yield router


async def test_the_installation_token_is_cached_until_shortly_before_it_expires(github: respx.MockRouter) -> None:  # fmt: skip
    clock = [1_792_846_800.0 - 3_600]  # an hour before expiry
    auth = GitHubAppAuth(API, REPO, "12345", SecretStr(PEM), clock=lambda: clock[0])
    assert await auth.token() == "ghs_token1"
    assert await auth.token() == "ghs_token1"  # cached
    clock[0] += 3_600 - 200  # within five minutes of expiry
    assert await auth.token() == "ghs_token2"
    jwt_sent = github.calls[0].request.headers["authorization"]
    assert jwt_sent.startswith("Bearer ey")
    assert len([c for c in github.calls if c.request.url.path.endswith("/installation")]) == 1
    await auth.aclose()


async def test_the_client_sends_the_apps_token_never_a_persons(github: respx.MockRouter) -> None:
    auth = GitHubAppAuth(API, REPO, "12345", SecretStr(PEM), clock=lambda: 1_792_843_200.0)
    client = GitHubClient(API, REPO, None, app=auth)
    await client.get(f"repos/{REPO}")
    [call] = [c for c in github.calls if c.request.url.path == f"/repos/{REPO}"]
    assert call.request.headers["authorization"] == "Bearer ghs_token1"
    await client.aclose()


async def test_an_app_not_installed_on_the_repository_says_so() -> None:
    with respx.mock() as router:
        router.get(f"{API}/repos/{REPO}/installation").mock(return_value=httpx.Response(404, json={"message": "Not Found"}))  # fmt: skip
        auth = GitHubAppAuth(API, REPO, "12345", SecretStr(PEM))
        with pytest.raises(CMSAuthError, match="isn't installed on clinic/site"):
            await auth.token()
        await auth.aclose()
