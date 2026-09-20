"""Phase 5 HTTP API: article drafts from approved opportunities, generated in the background.
There is no publishing endpoint."""

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from app.config import Settings
from app.crawling.fetcher import PoliteFetcher
from app.llm import LLMResponseError
from app.main import create_app
from app.prompts.article_edit import EditOut
from tests.fakellm import FakeLLM
from tests.fakesite import BASE, FakeClock, make_settings, mount_site, public_resolver
from tests.pipeline import ARTICLE_COMPANY


async def client_for(settings: Settings, clock: FakeClock, llm: FakeLLM | None) -> AsyncIterator[httpx.AsyncClient]:  # fmt: skip
    fetcher = PoliteFetcher(settings, resolver=public_resolver, clock=clock, sleep=clock.sleep)
    app = create_app(settings, fetcher=fetcher, llm=llm, resolver=public_resolver)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
            client.app = app  # type: ignore[attr-defined]
            yield client
    await fetcher.aclose()


async def settle(client: httpx.AsyncClient) -> None:
    await asyncio.gather(*client.app.state.background_tasks)  # type: ignore[attr-defined]


@pytest.fixture
async def client(db_url: str, clock: FakeClock) -> AsyncIterator[httpx.AsyncClient]:
    settings = make_settings(app_env="development", database_url=db_url)
    fake = FakeLLM()
    async for c in client_for(settings, clock, fake):
        for slug in ("acme", "acme-eu"):
            await c.post("/api/v1/competitors", json={"slug": slug, "name": slug, "website": f"{BASE}/", "tracked_pages": [f"{BASE}/pricing"]})  # fmt: skip
            with respx.mock(assert_all_called=False) as router:
                mount_site(router)
                assert (await c.post(f"/api/v1/competitors/{slug}/scans?wait=true")).status_code == 200  # fmt: skip
            assert (await c.post(f"/api/v1/competitors/{slug}/analyses?wait=true", json={"profile": False})).status_code == 200  # fmt: skip
        assert (await c.put("/api/v1/company-profile", json=ARTICLE_COMPANY)).status_code == 200
        assert (await c.post("/api/v1/opportunities/generate?wait=true")).status_code == 200
        c.fake = fake  # type: ignore[attr-defined]
        yield c


async def agents_opportunity(client: httpx.AsyncClient, *, approve: bool = True) -> int:
    ranked = (await client.get("/api/v1/opportunities")).json()
    opportunity_id: int = next(
        o["id"] for o in ranked if o["topic_label"].casefold() == "ai agents"
    )
    if approve:
        assert (await client.patch(f"/api/v1/opportunities/{opportunity_id}", json={"status": "approved"})).status_code == 200  # fmt: skip
    return opportunity_id


async def test_generate_in_the_background_then_inspect(client: httpx.AsyncClient) -> None:
    opportunity_id = await agents_opportunity(client)
    brief = await client.get(f"/api/v1/opportunities/{opportunity_id}/brief")
    assert brief.status_code == 200
    assert brief.json()["opportunity_id"] == opportunity_id

    created = await client.post("/api/v1/articles", json={"opportunity_id": opportunity_id})

    assert created.status_code == 202, created.text
    body = created.json()
    article_id = body["article"]["id"]
    assert created.headers["location"] == f"/api/v1/articles/{article_id}"
    assert body["created"] is True
    assert body["article"]["status"] == "queued"
    assert body["run"]["status"] in ("queued", "running")
    assert body["run"]["article_id"] == article_id
    early = (await client.get(f"/api/v1/articles/{article_id}")).json()
    assert early["brief"] == brief.json()  # inspectable right away
    await settle(client)

    detail = (await client.get(f"/api/v1/articles/{article_id}", params={"include_markdown": True})).json()  # fmt: skip
    assert (detail["status"], detail["progress"]["percent"], detail["current_step"]) == ("completed", 100, None)  # fmt: skip
    assert [s["step"] for s in detail["steps"]] == ["brief", "research", "outline", "draft", "edit"]
    assert all(s["prompt_version"] for s in detail["steps"])
    assert detail["content"]["title"] == "AI agents for founders: the practical guide"
    assert detail["content_version"]["kind"] == "final"
    assert detail["markdown"].startswith("# AI agents for founders")
    assert detail["tokens_used"] > 0
    assert detail["token_budget"] == 400_000
    assert detail["runs"][0]["status"] == "succeeded"
    assert detail["opportunity_status"] == "approved"

    sources = (await client.get(f"/api/v1/articles/{article_id}/sources")).json()
    assert {s["label"] for s in sources} == {"S1", "S2", "S3", "S4", "S5"}
    assert all(s["url"].startswith("https://") and s["facts"] for s in sources)
    versions = (await client.get(f"/api/v1/articles/{article_id}/versions")).json()
    assert [(v["kind"], v["number"], v["current"]) for v in versions] == [("outline", 1, True), ("draft", 1, True), ("final", 1, True)]  # fmt: skip
    final = (await client.get(f"/api/v1/articles/{article_id}/versions/{versions[-1]['id']}", params={"include_markdown": True})).json()  # fmt: skip
    assert final["citations"]
    assert all(c["url"].startswith("https://") and c["claim"] for c in final["citations"])
    assert final["changes"]
    assert "## Sources" in final["markdown"]
    steps = (await client.get(f"/api/v1/articles/{article_id}/steps")).json()
    assert [(s["step"], s["status"]) for s in steps][-1] == ("edit", "succeeded")

    listed = (await client.get("/api/v1/articles", params={"status": "completed", "opportunity_id": opportunity_id})).json()  # fmt: skip
    assert [a["id"] for a in listed] == [article_id]
    assert (await client.get("/api/v1/articles", params={"status": "failed"})).json() == []
    assert (await client.get("/api/v1/articles", params={"created_since": "2099-01-01T00:00:00Z"})).json() == []  # fmt: skip

    again = await client.post("/api/v1/articles", json={"opportunity_id": opportunity_id})
    assert again.status_code == 200
    assert (again.json()["created"], again.json()["article"]["id"]) == (False, article_id)
    noop = await client.post(f"/api/v1/articles/{article_id}/resume")
    assert noop.status_code == 200
    assert (noop.json()["created"], noop.json()["run"]) == (False, None)


async def test_failure_resume_cancel_and_regenerate(client: httpx.AsyncClient) -> None:
    opportunity_id = await agents_opportunity(client)
    client.fake.fail_schema[EditOut] = [LLMResponseError("edit JSON is malformed (fake)")]  # type: ignore[attr-defined]  # fmt: skip

    failed = (await client.post("/api/v1/articles?wait=true", json={"opportunity_id": opportunity_id})).json()  # fmt: skip

    article_id = failed["article"]["id"]
    assert (failed["article"]["status"], failed["article"]["failed_step"]) == ("failed", "edit")
    assert failed["run"]["status"] == "partial"
    assert "malformed" in failed["article"]["error"]
    resumed = await client.post(f"/api/v1/articles/{article_id}/resume")
    assert resumed.status_code == 202
    await settle(client)
    detail = (await client.get(f"/api/v1/articles/{article_id}")).json()
    assert detail["status"] == "completed"
    assert detail["runs"][-1]["summary"]["steps"]["research"] == "reused"

    assert (await client.post("/api/v1/articles", json={"opportunity_id": opportunity_id, "regenerate": True})).status_code == 409  # fmt: skip
    cancelled = await client.post(f"/api/v1/articles/{article_id}/cancel", json={"note": "rewrite"})
    assert cancelled.json()["status"] == "cancelled"
    assert (await client.post(f"/api/v1/articles/{article_id}/resume")).status_code == 409
    second = await client.post("/api/v1/articles", json={"opportunity_id": opportunity_id, "regenerate": True})  # fmt: skip
    assert second.status_code == 202
    assert second.json()["article"]["attempt"] == 2
    await settle(client)


async def test_errors(client: httpx.AsyncClient) -> None:
    unapproved = await agents_opportunity(client, approve=False)
    assert (await client.post("/api/v1/articles", json={"opportunity_id": unapproved})).status_code == 409  # fmt: skip
    assert (await client.post("/api/v1/articles", json={"opportunity_id": 999_999})).status_code == 404  # fmt: skip
    assert (await client.post("/api/v1/articles", json={"opportunity_id": "x"})).status_code == 422
    assert (await client.post("/api/v1/articles", json={"opportunity_id": 1, "publish": True})).status_code == 422  # fmt: skip
    assert (await client.get("/api/v1/opportunities/999999/brief")).status_code == 404
    for path in ("/api/v1/articles/999999", "/api/v1/articles/999999/sources", "/api/v1/articles/999999/versions", "/api/v1/articles/999999/steps", "/api/v1/articles/999999/versions/1"):  # fmt: skip
        assert (await client.get(path)).status_code == 404, path
    assert (await client.post("/api/v1/articles/999999/resume")).status_code == 404
    assert (await client.post("/api/v1/articles/999999/cancel")).status_code == 404
    # Nothing here can publish: the only article routes are generation, reading and cancelling.
    routes = {r.path for r in client.app.routes if "/articles" in getattr(r, "path", "")}  # type: ignore[attr-defined]  # fmt: skip
    assert not any("publish" in path or "schedule" in path for path in routes)


async def test_generation_needs_gemini_but_the_brief_preview_does_not(db_url: str, clock: FakeClock) -> None:  # fmt: skip
    settings = make_settings(app_env="development", database_url=db_url)
    async for client in client_for(settings, clock, None):  # no GEMINI_API_KEY, no provider
        response = await client.post("/api/v1/articles", json={"opportunity_id": 1})
        assert response.status_code == 503
        assert "GEMINI_API_KEY" in response.json()["detail"]
