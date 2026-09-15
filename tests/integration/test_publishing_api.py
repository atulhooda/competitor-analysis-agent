"""Phase 7 HTTP API: approve or reject, preflight, dry run, publish in the background, and
the publication history, against a fake WordPress. Nothing reaches a real site."""

from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from app.config import Settings
from tests.fakellm import FakeLLM
from tests.fakesite import BASE as SITE
from tests.fakesite import FakeClock, make_settings, mount_site
from tests.fakewordpress import BASE, PASSWORD, USERNAME, FakeWordPress
from tests.integration.test_articles_api import agents_opportunity, client_for, settle
from tests.pipeline import ARTICLE_COMPANY


@pytest.fixture
async def client(db_url: str, clock: FakeClock) -> AsyncIterator[httpx.AsyncClient]:
    settings: Settings = make_settings(app_env="development", database_url=db_url, wordpress_base_url=BASE, wordpress_username=USERNAME, wordpress_application_password=PASSWORD, cms_max_retries=0)  # fmt: skip
    fake = FakeLLM()
    async for c in client_for(settings, clock, fake):
        assert (await c.post("/api/v1/competitors", json={"slug": "acme", "name": "acme", "website": f"{SITE}/", "tracked_pages": [f"{SITE}/pricing"]})).status_code == 201  # fmt: skip
        with respx.mock(assert_all_called=False) as router:
            mount_site(router)
            assert (await c.post("/api/v1/competitors/acme/scans?wait=true")).status_code == 200
        assert (await c.post("/api/v1/competitors/acme/analyses?wait=true", json={"profile": False})).status_code == 200  # fmt: skip
        assert (await c.put("/api/v1/company-profile", json=ARTICLE_COMPANY)).status_code == 200
        assert (await c.post("/api/v1/opportunities/generate?wait=true")).status_code == 200
        c.fake = fake  # type: ignore[attr-defined]
        yield c


async def ready_article(client: httpx.AsyncClient) -> int:
    opportunity_id = await agents_opportunity(client)
    created = await client.post("/api/v1/articles?wait=true", json={"opportunity_id": opportunity_id})  # fmt: skip
    article_id: int = created.json()["article"]["id"]
    validated = await client.post(f"/api/v1/articles/{article_id}/validate?wait=true")
    assert validated.json()["article"]["status"] == "ready", validated.text
    return article_id


async def test_approve_preflight_and_publish_in_the_background(client: httpx.AsyncClient) -> None:
    article_id = await ready_article(client)
    wp = FakeWordPress()
    with respx.mock(assert_all_called=False) as router:
        wp.mount(router)

        pending = (await client.get(f"/api/v1/articles/{article_id}/approval")).json()
        assert pending["state"] == "pending"
        assert pending["gates_passed"] is True
        refused = await client.post(f"/api/v1/articles/{article_id}/publish")
        assert refused.status_code == 409
        assert "approve it first" in refused.json()["detail"]

        approved = await client.post(f"/api/v1/articles/{article_id}/approve", json={"note": "Reviewed and approved for publication.", "approver": "atul"})  # fmt: skip
        assert approved.status_code == 200, approved.text
        body = approved.json()
        assert body["created"] is True
        assert body["approval"]["approver"] == "atul"
        assert body["approval"]["channel"] == "api"
        assert body["state"]["state"] == "approved"
        assert body["state"]["can_publish"] is True

        preflight = (await client.post(f"/api/v1/articles/{article_id}/preflight")).json()
        assert preflight["ready"] is True
        assert preflight["action"] == "create"
        dry = await client.post(f"/api/v1/articles/{article_id}/publish?dry_run=true")
        assert dry.status_code == 200
        assert dry.json()["payload"]["status"] == "draft"
        assert wp.mutations == []

        queued = await client.post(f"/api/v1/articles/{article_id}/publish")

        assert queued.status_code == 202, queued.text
        assert queued.headers["location"] == f"/api/v1/articles/{article_id}/publication"
        assert queued.json()["status"] == "queued"
        assert queued.json()["publication_id"]
        assert queued.json()["run"]["kind"] == "article_publish"
        await settle(client)
        publication = (await client.get(f"/api/v1/articles/{article_id}/publication")).json()
        assert publication["status"] == "draft_created"
        assert publication["external_id"] == str(next(iter(wp.posts)))
        assert publication["url"] is None  # a draft isn't public
        assert publication["details"]["meta_description"]
        assert [a["action"] for a in publication["attempts"]] == ["create"]
        again = await client.post(f"/api/v1/articles/{article_id}/publish?wait=true")
        assert again.status_code == 200
        assert again.json()["created"] is False
        assert again.json()["status"] == "draft_created"
        assert len(wp.posts) == 1
        history = (await client.get(f"/api/v1/articles/{article_id}/publications")).json()
        assert len(history) == 1
        approvals = (await client.get(f"/api/v1/articles/{article_id}/approvals")).json()
        assert [a["decision"] for a in approvals] == ["approved"]
        live = await client.post(f"/api/v1/articles/{article_id}/publish", json={"status": "publish"})  # fmt: skip
        assert live.status_code == 409
        assert "WORDPRESS_ALLOW_DIRECT_PUBLISH" in live.json()["detail"]


async def test_rejection_and_refusals(client: httpx.AsyncClient) -> None:
    assert (await client.post("/api/v1/articles/999999/approve")).status_code == 404
    assert (await client.get("/api/v1/articles/999999/approval")).status_code == 404
    assert (await client.get("/api/v1/articles/999999/publications")).status_code == 404
    article_id = await ready_article(client)
    assert (await client.get(f"/api/v1/articles/{article_id}/publication")).status_code == 404
    assert (await client.post(f"/api/v1/articles/{article_id}/reject", json={})).status_code == 422  # a reason is required  # fmt: skip
    rejected = await client.post(f"/api/v1/articles/{article_id}/reject", json={"note": "Needs another review"})  # fmt: skip
    assert rejected.status_code == 200
    assert rejected.json()["state"]["state"] == "rejected"
    blocked = await client.post(f"/api/v1/articles/{article_id}/publish")
    assert blocked.status_code == 409
    assert "rejected" in blocked.json()["detail"]
    await client.post(f"/api/v1/articles/{article_id}/cancel", json={"note": "no"})
    refused = await client.post(f"/api/v1/articles/{article_id}/approve")
    assert refused.status_code == 409
    assert "cancelled" in refused.json()["detail"]


@pytest.fixture
async def unconfigured(db_url: str, clock: FakeClock) -> AsyncIterator[httpx.AsyncClient]:
    async for c in client_for(make_settings(app_env="development", database_url=db_url), clock, FakeLLM()):  # fmt: skip
        yield c


async def test_publishing_needs_the_cms_configured(unconfigured: httpx.AsyncClient) -> None:
    refused = await unconfigured.post("/api/v1/articles/1/publish")
    assert refused.status_code == 503
    assert "WORDPRESS_BASE_URL" in refused.json()["detail"]
    assert PASSWORD not in refused.text
    health = (await unconfigured.get("/health")).json()
    assert health["cms"] == {"provider": "wordpress", "configured": False}


@pytest.fixture
async def secured(db_url: str, clock: FakeClock) -> AsyncIterator[httpx.AsyncClient]:
    settings = make_settings(app_env="development", database_url=db_url, api_key="test-key")
    async for c in client_for(settings, clock, FakeLLM()):
        yield c


async def test_the_endpoints_require_the_api_key(secured: httpx.AsyncClient) -> None:
    for method, path in (("post", "approve"), ("post", "reject"), ("get", "approval"), ("get", "approvals"), ("post", "preflight"), ("post", "publish"), ("get", "publication"), ("get", "publications")):  # fmt: skip
        assert (await getattr(secured, method)(f"/api/v1/articles/1/{path}")).status_code == 401
