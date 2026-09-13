from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import respx

from app.config import Settings
from app.crawling.fetcher import PoliteFetcher
from app.main import create_app
from tests.fakesite import BASE, FakeClock, make_settings, mount_site, public_resolver

COMPETITORS_YAML = f"""
competitors:
  - slug: acme
    name: Acme
    website: {BASE}/
    tracked_pages: [{BASE}/pricing]
"""


@pytest.fixture
def competitors_file(tmp_path: Path) -> Path:
    path = tmp_path / "competitors.yaml"
    path.write_text(COMPETITORS_YAML, encoding="utf-8")
    return path


async def client_for(settings: Settings, clock: FakeClock) -> AsyncIterator[httpx.AsyncClient]:
    fetcher = PoliteFetcher(settings, resolver=public_resolver, clock=clock, sleep=clock.sleep)
    app = create_app(settings, fetcher=fetcher)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
            yield client
    await fetcher.aclose()


@pytest.fixture
async def dev_client(competitors_file: Path, clock: FakeClock) -> AsyncIterator[httpx.AsyncClient]:
    settings = make_settings(app_env="development", competitors_file=competitors_file)
    async for client in client_for(settings, clock):
        yield client


async def test_health_reports_gemini_without_requiring_it(dev_client: httpx.AsyncClient) -> None:
    response = await dev_client.get("/health")
    assert response.status_code == 200
    llm = response.json()["llm"]
    assert llm == {
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "configured": False,
        "required_from_phase": 3,
    }


async def test_health_never_exposes_the_key(competitors_file: Path, clock: FakeClock) -> None:
    settings = make_settings(competitors_file=competitors_file, gemini_api_key="super-secret-value")
    async for client in client_for(settings, clock):
        response = await client.get("/health")
        assert response.json()["llm"]["configured"] is True
        assert "super-secret-value" not in response.text


async def test_list_competitors(dev_client: httpx.AsyncClient) -> None:
    response = await dev_client.get("/api/v1/competitors")
    assert response.status_code == 200
    assert response.json()[0]["slug"] == "acme"


async def test_scan_endpoint_runs_a_compliant_scan(dev_client: httpx.AsyncClient) -> None:
    with respx.mock(assert_all_called=False) as router:
        routes = mount_site(router)
        response = await dev_client.post(
            "/api/v1/competitors/acme/scan", json={"since": "2026-09-01", "limit": 10}
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert f"{BASE}/blog/ai-support-agents" in [i["final_url"] for i in body["items"]]
    assert not routes["/private/internal-post"].called


async def test_scan_validation_errors(dev_client: httpx.AsyncClient) -> None:
    assert (await dev_client.post("/api/v1/competitors/nope/scan")).status_code == 404
    bad_since = await dev_client.post("/api/v1/competitors/acme/scan", json={"since": "whenever"})
    assert bad_since.status_code == 422
    bad_limit = await dev_client.post("/api/v1/competitors/acme/scan", json={"limit": 0})
    assert bad_limit.status_code == 422


async def test_api_key_is_enforced_when_configured(
    competitors_file: Path, clock: FakeClock
) -> None:
    settings = make_settings(competitors_file=competitors_file, api_key="k-123")
    async for client in client_for(settings, clock):
        assert (await client.get("/api/v1/competitors")).status_code == 401
        wrong = await client.get("/api/v1/competitors", headers={"X-API-Key": "nope"})
        assert wrong.status_code == 401
        right = await client.get("/api/v1/competitors", headers={"X-API-Key": "k-123"})
        assert right.status_code == 200
        assert (await client.get("/health")).status_code == 200  # health stays public


async def test_production_without_api_key_fails_closed(
    competitors_file: Path, clock: FakeClock
) -> None:
    settings = make_settings(app_env="production", competitors_file=competitors_file)
    async for client in client_for(settings, clock):
        assert (await client.get("/api/v1/competitors")).status_code == 503
