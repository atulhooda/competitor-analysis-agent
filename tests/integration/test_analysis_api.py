"""Phase 3 HTTP API, end to end: scan (fake site) → analyze (fake Gemini) → intelligence."""

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from app.config import Settings
from app.crawling.fetcher import PoliteFetcher
from app.main import create_app
from tests.fakellm import FakeLLM
from tests.fakesite import BASE, FakeClock, make_settings, mount_site, public_resolver

ACME = {"slug": "acme", "name": "Acme", "website": f"{BASE}/", "tracked_pages": [f"{BASE}/pricing"]}


async def client_for(
    settings: Settings, clock: FakeClock, llm: FakeLLM | None
) -> AsyncIterator[httpx.AsyncClient]:
    fetcher = PoliteFetcher(settings, resolver=public_resolver, clock=clock, sleep=clock.sleep)
    app = create_app(settings, fetcher=fetcher, llm=llm)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
            client.app = app  # type: ignore[attr-defined]
            yield client
    await fetcher.aclose()


@pytest.fixture
def fake() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
async def client(db_url: str, clock: FakeClock, fake: FakeLLM) -> AsyncIterator[httpx.AsyncClient]:
    settings = make_settings(app_env="development", database_url=db_url)
    async for c in client_for(settings, clock, fake):
        yield c


async def scanned(client: httpx.AsyncClient) -> None:
    await client.post("/api/v1/competitors", json=ACME)
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        response = await client.post("/api/v1/competitors/acme/scans?wait=true")
    assert response.status_code == 200, response.text


async def test_analysis_and_intelligence_endpoints(client: httpx.AsyncClient, fake: FakeLLM) -> None:  # fmt: skip
    await scanned(client)
    plan = (await client.get("/api/v1/competitors/acme/analysis-plan")).json()
    assert plan["coverage"]["pending"] == 8
    assert plan["batches"] == 2
    assert fake.requests == []  # planning never calls the model

    response = await client.post("/api/v1/competitors/acme/analyses?wait=true", json={"limit": 50})
    assert response.status_code == 200, response.text
    run = response.json()["run"]
    assert (run["kind"], run["status"], run["summary"]["analyzed"]) == ("analysis", "succeeded", 8)
    assert run["stats"]["total_tokens"] > 0
    assert response.headers["location"] == f"/api/v1/runs/{run['id']}"

    analyses = (await client.get("/api/v1/analyses", params={"competitor": "acme"})).json()
    assert len(analyses) == 8
    on_topic = (await client.get("/api/v1/analyses", params={"topic": "ai-agents"})).json()
    assert on_topic
    assert all(any(t["slug"] == "ai-agents" for t in a["topics"]) for a in on_topic)
    pricing = (await client.get("/api/v1/analyses", params={"content_format": "pricing_page"})).json()  # fmt: skip
    assert [a["url"] for a in pricing] == [f"{BASE}/pricing"]
    history = (await client.get(f"/api/v1/content/{pricing[0]['content_item_id']}/analyses")).json()
    assert len(history) == 1
    assert history[0]["is_current"]
    assert (await client.get("/api/v1/content/999999/analyses")).status_code == 404

    topics = (await client.get("/api/v1/topics")).json()
    agents = next(t for t in topics if t["slug"] == "ai-agents")
    assert agents["items"] >= 3
    assert agents["competitors"] == 1
    assert agents["subtopics"] >= 1
    subtopics = (await client.get("/api/v1/topics", params={"parent": "ai-agents"})).json()
    assert {t["slug"] for t in subtopics} >= {"ai-agents--ticket-automation"}
    detail = (await client.get("/api/v1/topics/ai-agents")).json()
    assert detail["trend"]["items"] == agents["items"]
    assert detail["recent_items"]
    assert (await client.get("/api/v1/topics/nope")).status_code == 404
    assert (await client.get("/api/v1/topics", params={"parent": "nope"})).status_code == 404

    intel = (await client.get("/api/v1/competitors/acme/intelligence", params={"days": 30})).json()
    assert intel["coverage"]["pending"] == 0
    assert intel["topics"]
    assert intel["formats"]
    assert intel["recent_items"]
    assert intel["profile"]["version"] == 1
    assert (await client.get("/api/v1/competitors/nope/intelligence")).status_code == 404
    profile = (await client.get("/api/v1/competitors/acme/profile")).json()
    assert profile["profile"]["tagline"]["evidence"][0]["url"] == f"{BASE}/"
    assert len((await client.get("/api/v1/competitors/acme/profiles")).json()) == 1

    landscape = (await client.get("/api/v1/intelligence/landscape")).json()
    assert landscape["report"] is None
    assert landscape["metrics"]["competitors"][0]["competitor"] == "acme"
    generated = await client.post("/api/v1/intelligence/landscape?wait=true", json={"window_days": 30})  # fmt: skip
    assert generated.status_code == 200, generated.text
    assert generated.json()["run"]["status"] == "succeeded"
    report = (await client.get("/api/v1/intelligence/landscape")).json()["report"]
    assert report["narrative"]["summary"]
    assert report["metrics"]["topics"]

    usage = (await client.get("/api/v1/llm/usage", params={"days": 1})).json()
    purposes = {row["purpose"] for row in usage["rows"]}
    assert purposes == {"content_analysis", "competitor_profile", "landscape"}
    assert usage["today_tokens"] == usage["total_tokens"] > 0


async def test_topic_admin_endpoints(client: httpx.AsyncClient) -> None:
    await scanned(client)
    await client.post("/api/v1/competitors/acme/analyses?wait=true")
    merged = await client.post("/api/v1/topics/merge", json={"source": "automation", "target": "ai-agents"})  # fmt: skip
    assert merged.status_code == 200, merged.text
    body = merged.json()
    assert body["target"] == "ai-agents"
    assert (body["links_moved"], body["links_combined"]) == (0, 1)  # the page had both topics
    bad = await client.post("/api/v1/topics/merge", json={"source": "ai-agents", "target": "ai-agents--ticket-automation"})  # fmt: skip
    assert bad.status_code == 422
    missing = await client.post("/api/v1/topics/merge", json={"source": "nope", "target": "ai-agents"})  # fmt: skip
    assert missing.status_code == 422
    consolidated = (await client.post("/api/v1/topics/consolidate")).json()
    assert consolidated["applied"] is False
    assert consolidated["proposals"] == []


async def test_background_analysis_returns_a_run_to_poll(client: httpx.AsyncClient) -> None:
    await scanned(client)
    response = await client.post("/api/v1/competitors/acme/analyses")
    assert response.status_code == 202, response.text
    run_id = response.json()["run"]["id"]
    conflict = await client.post("/api/v1/competitors/acme/analyses")
    assert conflict.status_code in (202, 409)  # 409 while the first is still running
    await asyncio.gather(*client.app.state.background_tasks)  # type: ignore[attr-defined]
    run = (await client.get(f"/api/v1/runs/{run_id}")).json()
    assert run["status"] == "succeeded"
    assert run["trigger"] == "api"
    assert (await client.post("/api/v1/competitors/nope/analyses")).status_code == 404


async def test_ai_endpoints_need_a_gemini_key_but_reads_do_not(db_url: str, clock: FakeClock) -> None:  # fmt: skip
    settings = make_settings(app_env="development", database_url=db_url)
    async for client in client_for(settings, clock, llm=None):
        await scanned(client)
        refused = await client.post("/api/v1/competitors/acme/analyses")
        assert refused.status_code == 503
        assert "GEMINI_API_KEY" in refused.json()["detail"]
        assert (await client.post("/api/v1/intelligence/landscape")).status_code == 503
        assert (await client.post("/api/v1/topics/consolidate")).status_code == 503
        assert (await client.get("/api/v1/competitors/acme/intelligence")).status_code == 200
        assert (await client.get("/api/v1/competitors/acme/analysis-plan")).status_code == 200
        assert (await client.get("/api/v1/competitors/acme/profile")).status_code == 404
        health = (await client.get("/health")).json()
        assert health["llm"]["configured"] is False


async def test_intelligence_endpoints_require_the_api_key(db_url: str, clock: FakeClock) -> None:
    settings = make_settings(database_url=db_url, api_key="k-123")
    async for client in client_for(settings, clock, llm=FakeLLM()):
        for path in ("/api/v1/topics", "/api/v1/analyses", "/api/v1/llm/usage", "/api/v1/intelligence/landscape"):  # fmt: skip
            assert (await client.get(path)).status_code == 401
        assert (await client.post("/api/v1/competitors/acme/analyses")).status_code == 401
        ok = await client.get("/api/v1/topics", headers={"X-API-Key": "k-123"})
        assert ok.status_code == 200
