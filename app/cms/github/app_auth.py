"""Signing in as a GitHub App, so the agent's commits, pull requests and merges are the
app's (``<app>[bot]``) and never a person's: nothing it does appears on anyone's profile or
contribution graph.

    the app's private key → a 9-minute JWT (RS256) → the app's installation on the repo
    → an installation token (an hour), refreshed five minutes before it expires

Neither the key nor a token is ever logged or put in an error message.
"""

import base64
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from pydantic import SecretStr

from app.cms.errors import CMSAuthError, CMSConfigurationError, CMSNetworkError

API_VERSION = "2022-11-28"
REFRESH_BEFORE = 300.0  # seconds before expiry a token is replaced


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def load_key(pem: SecretStr) -> rsa.RSAPrivateKey:
    """The app's private key (PEM). An environment variable may hold it with literal \\n."""
    text = pem.get_secret_value().strip().replace("\\n", "\n")
    try:
        key = serialization.load_pem_private_key(text.encode(), password=None)
    except (ValueError, TypeError) as exc:
        raise CMSConfigurationError("GITHUB_APP_PRIVATE_KEY isn't a PEM private key (paste the whole .pem file the app gave you)") from exc  # fmt: skip
    if not isinstance(key, rsa.RSAPrivateKey):
        raise CMSConfigurationError("GITHUB_APP_PRIVATE_KEY must be the app's RSA key")
    return key


def app_jwt(app_id: str, key: rsa.RSAPrivateKey, now: float) -> str:
    """The app's own credential: valid 9 minutes, issued a minute early for clock skew."""
    header = _b64(json.dumps({"alg": "RS256", "typ": "JWT"}, separators=(",", ":")).encode())
    claims = _b64(json.dumps({"iat": int(now) - 60, "exp": int(now) + 540, "iss": str(app_id)}, separators=(",", ":")).encode())  # fmt: skip
    signing_input = f"{header}.{claims}".encode()
    signature = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{header}.{claims}.{_b64(signature)}"


@dataclass
class _Token:
    value: str
    expires: float


class GitHubAppAuth:
    """Installation tokens for the app's installation on one repository, cached."""

    def __init__(
        self,
        api_url: str,
        repo: str,
        app_id: str,
        private_key: SecretStr,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.time,
        timeout: float = 30.0,
        user_agent: str = "competitor-analysis-agent",
    ) -> None:
        self._api = api_url.rstrip("/")
        self._repo = repo
        self._app_id = app_id
        self._key = load_key(private_key)
        self._clock = clock
        self._http = httpx.AsyncClient(timeout=timeout, transport=transport, follow_redirects=False, headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": API_VERSION, "User-Agent": user_agent})  # fmt: skip
        self._installation: int | None = None
        self._token: _Token | None = None

    async def token(self) -> str:
        now = self._clock()
        if self._token is not None and now < self._token.expires - REFRESH_BEFORE:
            return self._token.value
        bearer = {"Authorization": f"Bearer {app_jwt(self._app_id, self._key, now)}"}
        if self._installation is None:
            found = await self._call("GET", f"/repos/{self._repo}/installation", bearer, "the app isn't installed on " + self._repo + ": install it on that repository (Settings → GitHub Apps)")  # fmt: skip
            self._installation = int(found["id"])
        made = await self._call("POST", f"/app/installations/{self._installation}/access_tokens", bearer, "GitHub refused an installation token for the app")  # fmt: skip
        try:
            until = datetime.fromisoformat(
                str(made.get("expires_at")).replace("Z", "+00:00")
            ).timestamp()
        except ValueError:
            until = now + 3_600  # GitHub's installation tokens last an hour
        self._token = _Token(str(made["token"]), until)
        return self._token.value

    async def _call(self, method: str, path: str, headers: dict[str, str], refused: str) -> dict[str, Any]:  # fmt: skip
        try:
            response = await self._http.request(method, self._api + path, headers=headers)
        except httpx.HTTPError as exc:
            raise CMSNetworkError(f"{method} {path}: couldn't reach GitHub ({type(exc).__name__})") from exc  # fmt: skip
        if response.status_code in (401, 403, 404):
            raise CMSAuthError(f"{refused} (GitHub answered {response.status_code}; check GITHUB_APP_ID and GITHUB_APP_PRIVATE_KEY)")  # fmt: skip
        if response.status_code >= 400:
            raise CMSNetworkError(f"{method} {path}: GitHub answered {response.status_code}")
        data = response.json()
        return data if isinstance(data, dict) else {}

    async def aclose(self) -> None:
        await self._http.aclose()


__all__ = ["GitHubAppAuth", "app_jwt", "load_key"]
