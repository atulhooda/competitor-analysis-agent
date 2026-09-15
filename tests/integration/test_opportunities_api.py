"""Phase 4 HTTP API: company profile, generation (background and synchronous), listing,
filtering, details, evidence, history and status changes."""

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

COMPANY = {
    "name": "Example Startup",
    "description": "Helps founders deploy AI agents for customer support.",
    "target_audiences": ["founders"],
    "core_topics": ["AI agents"],
    "adjacent_topics": ["Automation"],
    "excluded_topics": ["Pricing"],
}


async def client_for(settings: Settings, clock: FakeClock, llm: FakeLLM) -> AsyncIterator[httpx.AsyncClient]:  # fmt: skip
    fetcher = PoliteFetcher(settings, resolver=public_resolver, clock=clock, sleep=clock.sleep)
    app = create_app(settings, fetcher=fetcher, llm=llm)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
            client.app = app  # type: ignore[attr-defined]
            yield client
    await fetcher.aclose()


@pytest.fixture
async def client(db_url: str, clock: FakeClock) -> AsyncIterator[httpx.AsyncClient]:
    settings = make_settings(app_env="development", database_url=db_url)
    async for c in client_for(settings, clock, FakeLLM()):
        for slug in ("acme", "acme-eu"):
            await c.post("/api/v1/competitors", json={"slug": slug, "name": slug, "website": f"{BASE}/", "tracked_pages": [f"{BASE}/pricing"]})  # fmt: skip
            with respx.mock(assert_all_called=False) as router:
                mount_site(router)
                assert (await c.post(f"/api/v1/competitors/{slug}/scans?wait=true")).status_code == 200  # fmt: skip
            analyzed = await c.post(f"/api/v1/competitors/{slug}/analyses?wait=true", json={"profile": False})  # fmt: skip
            assert analyzed.json()["run"]["status"] == "succeeded"
        yield c


async def test_company_profile_versions(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/v1/company-profile")).status_code == 404
    assert (
        await client.post("/api/v1/opportunities/generate")
    ).status_code == 409  # no profile yet
    created = await client.put("/api/v1/company-profile", json=COMPANY)
    assert created.status_code == 200, created.text
    assert created.json()["created"] is True
    assert created.json()["version"]["version"] == 1
    same = await client.put("/api/v1/company-profile", json=COMPANY)
    assert same.json()["created"] is False
    assert same.json()["version"]["version"] == 1
    changed = await client.put("/api/v1/company-profile", json={**COMPANY, "tone": "plain"})
    assert changed.json()["version"]["version"] == 2
    assert [v["version"] for v in (await client.get("/api/v1/company-profile/versions")).json()] == [2, 1]  # fmt: skip
    assert (await client.get("/api/v1/company-profile")).json()["profile"]["tone"] == "plain"
    bad = await client.put("/api/v1/company-profile", json={**COMPANY, "preferred_formats": ["poem"]})  # fmt: skip
    assert bad.status_code == 422


async def test_generate_list_inspect_and_decide(client: httpx.AsyncClient) -> None:
    await client.put("/api/v1/company-profile", json=COMPANY)
    response = await client.post("/api/v1/opportunities/generate", json={"interpret": True})
    assert response.status_code == 202, response.text
    run = response.json()["run"]
    assert run["status"] in ("queued", "running")
    assert response.headers["location"] == f"/api/v1/runs/{run['id']}"
    await asyncio.gather(*client.app.state.background_tasks)  # type: ignore[attr-defined]
    finished = (await client.get(f"/api/v1/runs/{run['id']}")).json()
    assert (finished["kind"], finished["status"]) == ("opportunities", "succeeded")
    assert finished["summary"]["qualified"] >= 2

    ranked = (await client.get("/api/v1/opportunities")).json()
    assert [o["rank"] for o in ranked] == list(range(1, len(ranked) + 1))
    assert [o["score"] for o in ranked] == sorted((o["score"] for o in ranked), reverse=True)
    agents = next(o for o in ranked if o["topic_label"].casefold() == "ai agents")
    assert agents["interpretation_status"] == "ok"
    assert agents["title"].endswith("the practical guide for founders")

    top = ranked[0]["score"]
    assert all(o["score"] >= top for o in (await client.get("/api/v1/opportunities", params={"min_score": top})).json())  # fmt: skip
    by_topic = (await client.get("/api/v1/opportunities", params={"topic": "ai-agents"})).json()
    assert [o["id"] for o in by_topic] == [agents["id"]]
    assert (await client.get("/api/v1/opportunities", params={"topic": "agents"})).json()
    assert (await client.get("/api/v1/opportunities", params={"competitor": "acme-eu"})).json()
    assert (await client.get("/api/v1/opportunities", params={"competitor": "nobody"})).json() == []
    assert (await client.get("/api/v1/opportunities", params={"status": "rejected"})).json() == []

    detail = (await client.get(f"/api/v1/opportunities/{agents['id']}")).json()
    assessment = detail["assessment"]
    assert {c["dimension"] for c in assessment["breakdown"]} == {"momentum", "strategic_fit", "audience_fit", "content_gap", "recency", "saturation"}  # fmt: skip
    assert assessment["company_profile_version"] == 1
    assert assessment["interpretation"]["recommended_format"] == "comparison"
    assert assessment["suggestion"]["reasons"]
    assert detail["events"][0]["kind"] == "created"
    assert (await client.get("/api/v1/opportunities/999999")).status_code == 404

    evidence = (await client.get(f"/api/v1/opportunities/{agents['id']}/evidence")).json()
    kinds = {e["kind"] for e in evidence}
    assert {"topic_metrics", "topic_trend", "content", "company_profile"} <= kinds
    page = next(e for e in evidence if e["kind"] == "content")
    assert page["competitor"] in ("acme", "acme-eu")
    assert page["data"]["url"].startswith(BASE)
    assert (await client.get("/api/v1/opportunities/999999/evidence")).status_code == 404
    history = (await client.get(f"/api/v1/opportunities/{agents['id']}/history")).json()
    assert len(history) == 1
    assert history[0]["change"] is None

    approved = await client.patch(f"/api/v1/opportunities/{agents['id']}", json={"status": "approved", "note": "go"})  # fmt: skip
    assert approved.status_code == 200
    assert (approved.json()["status"], approved.json()["status_note"]) == ("approved", "go")
    assert approved.json()["events"][-1]["actor"] == "api"
    invalid = await client.patch(f"/api/v1/opportunities/{agents['id']}", json={"status": "new"})
    assert invalid.status_code == 409
    assert (await client.patch(f"/api/v1/opportunities/{agents['id']}", json={"status": "maybe"})).status_code == 422  # fmt: skip
    assert (await client.patch("/api/v1/opportunities/999999", json={"status": "reviewed"})).status_code == 404  # fmt: skip
    listed = (await client.get("/api/v1/opportunities", params={"status": ["approved", "new"]})).json()  # fmt: skip
    assert agents["id"] in [o["id"] for o in listed]


async def test_synchronous_generation_and_conflicts(client: httpx.AsyncClient) -> None:
    await client.put("/api/v1/company-profile", json=COMPANY)
    done = await client.post("/api/v1/opportunities/generate?wait=true", json={"interpret": False, "window_days": 90})  # fmt: skip
    assert done.status_code == 200
    assert done.json()["run"]["status"] == "succeeded"
    assert done.json()["run"]["params"]["window_days"] == 90
    queued = await client.post("/api/v1/opportunities/generate")
    conflict = await client.post("/api/v1/opportunities/generate")
    assert queued.status_code == 202
    assert conflict.status_code == 409  # one generation at a time
    await asyncio.gather(*client.app.state.background_tasks)  # type: ignore[attr-defined]
    assert (await client.post("/api/v1/opportunities/generate", json={"window_days": 3})).status_code == 422  # fmt: skip
