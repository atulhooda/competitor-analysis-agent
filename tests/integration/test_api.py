import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
import pytest
import respx

from app.config import Settings
from app.crawling.fetcher import PoliteFetcher
from app.main import create_app
from tests.fakesite import BASE, FakeClock, make_settings, mount_site, public_resolver

ACME = {"slug": "acme", "name": "Acme", "website": f"{BASE}/", "tracked_pages": [f"{BASE}/pricing"]}


async def client_for(settings: Settings, clock: FakeClock) -> AsyncIterator[httpx.AsyncClient]:
    fetcher = PoliteFetcher(settings, resolver=public_resolver, clock=clock, sleep=clock.sleep)
    app = create_app(settings, fetcher=fetcher)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
            client.app = app  # type: ignore[attr-defined]
            yield client
    await fetcher.aclose()


@pytest.fixture
async def client(db_url: str, clock: FakeClock) -> AsyncIterator[httpx.AsyncClient]:
    async for c in client_for(make_settings(app_env="development", database_url=db_url), clock):
        yield c


async def test_health(client: httpx.AsyncClient) -> None:
    body = (await client.get("/health")).json()
    assert body["status"] == "ok"
    assert body["database"]["reachable"] is True
    assert body["database"]["up_to_date"] is True
    assert body["llm"] == {
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "configured": False,
        "required_from_phase": 3,
    }


async def test_health_reports_an_unreachable_database(clock: FakeClock) -> None:
    settings = make_settings(database_url="postgresql+psycopg://postgres@127.0.0.1:1/nothing")
    async for client in client_for(settings, clock):
        body = (await client.get("/health")).json()
        assert body["status"] == "degraded"
        assert body["database"]["reachable"] is False


async def test_health_never_exposes_secrets(db_url: str, clock: FakeClock) -> None:
    settings = make_settings(database_url=db_url, gemini_api_key="super-secret-value")
    async for client in client_for(settings, clock):
        response = await client.get("/health")
        assert response.json()["llm"]["configured"] is True
        assert "super-secret-value" not in response.text


async def test_competitor_management(client: httpx.AsyncClient) -> None:
    created = await client.post("/api/v1/competitors", json=ACME)
    assert created.status_code == 201, created.text
    assert created.json()["active"] is True
    assert (await client.post("/api/v1/competitors", json=ACME)).status_code == 409

    patched = await client.patch(
        "/api/v1/competitors/acme",
        json={"name": "Acme Inc.", "exclude_patterns": ["/tag/"], "active": False},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["name"] == "Acme Inc."
    assert patched.json()["config"]["exclude_patterns"] == ["/tag/"]
    assert patched.json()["config"]["tracked_pages"] == [f"{BASE}/pricing"]  # kept
    assert (await client.get("/api/v1/competitors")).json() == []  # inactive hidden
    listed = await client.get("/api/v1/competitors", params={"include_inactive": True})
    assert [c["slug"] for c in listed.json()] == ["acme"]
    bad = await client.patch("/api/v1/competitors/acme", json={"exclude_patterns": ["("]})
    assert bad.status_code == 422
    assert "invalid regex" in bad.json()["detail"][0]["msg"]
    assert (await client.get("/api/v1/competitors/nope")).status_code == 404


async def test_synchronous_scan_and_history_endpoints(client: httpx.AsyncClient) -> None:
    await client.post("/api/v1/competitors", json=ACME)
    with respx.mock(assert_all_called=False) as router:
        routes = mount_site(router)
        response = await client.post("/api/v1/competitors/acme/scans?wait=true", json={"limit": 10})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["run"]["status"] == "succeeded"
    assert body["run"]["summary"]["changes"]["baseline"] is True
    assert response.headers["location"] == f"/api/v1/runs/{body['run']['id']}"
    assert body["result"]["items"]
    assert "discovered" not in body["result"]  # internal-only field is never serialized
    assert not routes["/private/internal-post"].called

    run = (await client.get(f"/api/v1/runs/{body['run']['id']}")).json()
    assert any(e["event"] == "robots_disallowed" for e in run["events"])
    assert [r["id"] for r in (await client.get("/api/v1/runs")).json()] == [run["id"]]

    week = datetime(2026, 9, 7, tzinfo=UTC).isoformat()
    content = await client.get(
        "/api/v1/content", params={"competitor": "acme", "published_since": week}
    )
    titles = [c["title"] for c in content.json()]
    assert titles == ["Launch Week Recap", "AI Support Agents: A Practical Guide"]

    item_id = content.json()[1]["id"]
    detail = (await client.get(f"/api/v1/content/{item_id}", params={"include_text": True})).json()
    assert detail["published_at_source"] == "structured_data"
    assert detail["current_version"]["text"]
    versions = (await client.get(f"/api/v1/content/{item_id}/versions")).json()
    assert [v["version_no"] for v in versions] == [1]
    assert versions[0]["text"] is None  # only with include_text
    assert (await client.get("/api/v1/content/999999")).status_code == 404

    assert (await client.get("/api/v1/changes")).json() == []  # baseline: nothing is "new"
    activity = (await client.get("/api/v1/competitors/acme/activity", params={"weeks": 4})).json()
    assert len(activity["weeks"]) == 4


async def test_background_scan_returns_a_run_to_poll(client: httpx.AsyncClient) -> None:
    await client.post("/api/v1/competitors", json=ACME)
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        response = await client.post("/api/v1/competitors/acme/scans")
        assert response.status_code == 202, response.text
        run_id = response.json()["run"]["id"]
        assert response.json()["run"]["status"] in ("queued", "running")
        await asyncio.gather(*client.app.state.background_tasks)  # type: ignore[attr-defined]
    run = (await client.get(f"/api/v1/runs/{run_id}")).json()
    assert run["status"] == "succeeded"
    assert run["trigger"] == "api"


async def test_scan_request_errors(client: httpx.AsyncClient) -> None:
    assert (await client.post("/api/v1/competitors/nope/scans")).status_code == 404
    await client.post("/api/v1/competitors", json=ACME)
    bad_since = await client.post("/api/v1/competitors/acme/scans", json={"since": "whenever"})
    assert bad_since.status_code == 422
    bad_limit = await client.post("/api/v1/competitors/acme/scans", json={"limit": 0})
    assert bad_limit.status_code == 422
    await client.patch("/api/v1/competitors/acme", json={"active": False})
    assert (await client.post("/api/v1/competitors/acme/scans")).status_code == 409


async def test_api_key_is_enforced_when_configured(db_url: str, clock: FakeClock) -> None:
    settings = make_settings(database_url=db_url, api_key="k-123")
    async for client in client_for(settings, clock):
        assert (await client.get("/api/v1/competitors")).status_code == 401
        wrong = await client.get("/api/v1/competitors", headers={"X-API-Key": "nope"})
        assert wrong.status_code == 401
        right = await client.get("/api/v1/competitors", headers={"X-API-Key": "k-123"})
        assert right.status_code == 200
        assert (await client.get("/api/v1/changes")).status_code == 401
        assert (await client.get("/health")).status_code == 200  # health stays public


async def test_production_without_api_key_fails_closed(db_url: str, clock: FakeClock) -> None:
    settings = make_settings(app_env="production", database_url=db_url)
    async for client in client_for(settings, clock):
        assert (await client.get("/api/v1/competitors")).status_code == 503
