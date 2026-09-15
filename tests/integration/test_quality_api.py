"""Phase 6 HTTP API: validate a completed article in the background and read the fact-check,
originality, SEO, quality and revision results. There is no publishing endpoint."""

from collections.abc import AsyncIterator

import httpx
import pytest

from app.config import Settings
from tests.fakellm import FakeLLM
from tests.fakesite import FakeClock, make_settings
from tests.integration.test_articles_api import agents_opportunity, client_for, settle
from tests.integration.test_articles_api import client as client

HANDOFF = "Designing the human handoff is where"


async def completed_article(client: httpx.AsyncClient) -> int:
    opportunity_id = await agents_opportunity(client)
    created = await client.post("/api/v1/articles?wait=true", json={"opportunity_id": opportunity_id})  # fmt: skip
    assert created.status_code == 200, created.text
    assert created.json()["article"]["status"] == "completed"
    article_id: int = created.json()["article"]["id"]
    return article_id


async def test_validate_in_the_background_then_read_the_results(client: httpx.AsyncClient) -> None:
    article_id = await completed_article(client)

    queued = await client.post(f"/api/v1/articles/{article_id}/validate")

    assert queued.status_code == 202, queued.text
    assert queued.headers["location"] == f"/api/v1/articles/{article_id}/quality"
    body = queued.json()
    assert body["created"] is True
    assert body["article"]["status"] == "validating"
    assert body["run"]["kind"] == "article_quality"
    assert body["run"]["article_id"] == article_id
    await settle(client)

    quality = (await client.get(f"/api/v1/articles/{article_id}/quality")).json()
    assert quality["status"] == "ready"
    assert quality["current_step"] is None
    assert quality["quality_score"] >= 70
    assert quality["revision_count"] == 0
    assert quality["recommended_version_id"] == quality["report"]["version_id"]
    assert quality["report"]["passed"] is True
    assert {g["name"] for g in quality["report"]["gates"]} >= {"no_contradicted_claims", "originality", "seo_fields", "minimum_score"}  # fmt: skip
    assert [c["dimension"] for c in quality["report"]["breakdown"]][-1] == "gemini_judgment"
    assert quality["metrics"]["readability"]["flesch_reading_ease"] > 0
    assert len(quality["judge"]["dimensions"]) == 8
    assert [v["recommended"] for v in quality["versions"]] == [True]
    assert quality["quality_tokens_used"] > 0

    fact_check = (await client.get(f"/api/v1/articles/{article_id}/fact-check")).json()
    assert fact_check["metrics"]["cited_claims"] > 0
    assert {c["verdict"] for c in fact_check["checks"]} == {"supported"}
    assert all(c["evidence_verified"] and c["prompt_version"] == "fact-check/1" for c in fact_check["checks"])  # fmt: skip
    none = (await client.get(f"/api/v1/articles/{article_id}/fact-check?verdict=contradicted")).json()  # fmt: skip
    assert none["checks"] == []

    originality = (await client.get(f"/api/v1/articles/{article_id}/originality")).json()
    assert originality["report"]["documents"] > 0
    assert originality["report"]["severe"] is False

    seo = (await client.get(f"/api/v1/articles/{article_id}/seo")).json()
    assert seo["report"]["package"]["slug"] == "ai-agents"
    assert seo["report"]["mandatory_missing"] == []

    revisions = (await client.get(f"/api/v1/articles/{article_id}/revisions")).json()
    assert [(r["kind"], r["recommended"]) for r in revisions] == [("final", True)]

    detail = (await client.get(f"/api/v1/articles/{article_id}")).json()
    assert detail["status"] == "ready"
    assert detail["quality_score"] == quality["quality_score"]
    assert detail["recommended_version_id"] == quality["recommended_version_id"]
    assert detail["slug"] == "ai-agents"
    listed = (await client.get("/api/v1/articles?status=ready")).json()
    assert [a["id"] for a in listed] == [article_id]


async def test_validate_and_revise_synchronously(client: httpx.AsyncClient) -> None:
    article_id = await completed_article(client)
    fake: FakeLLM = client.fake  # type: ignore[attr-defined]
    fake.verdicts = {HANDOFF: "contradicted"}

    validated = await client.post(f"/api/v1/articles/{article_id}/validate?wait=true")

    assert validated.status_code == 200, validated.text
    assert validated.json()["article"]["status"] == "ready"
    assert validated.json()["article"]["revision_count"] == 1
    assert validated.json()["run"]["status"] == "succeeded"
    revisions = (await client.get(f"/api/v1/articles/{article_id}/revisions")).json()
    assert [(r["kind"], r["recommended"]) for r in revisions] == [("final", False), ("revision", True)]  # fmt: skip
    first = revisions[0]["version_id"]
    rejected = (await client.get(f"/api/v1/articles/{article_id}/fact-check?version_id={first}&verdict=contradicted")).json()  # fmt: skip
    assert [c["claim"][:36] for c in rejected["checks"]] == [HANDOFF]
    assert (await client.get(f"/api/v1/articles/{article_id}/quality?version_id={first}")).json()["report"]["passed"] is False  # fmt: skip

    revised = await client.post(f"/api/v1/articles/{article_id}/revise?wait=true", json={"note": "Add a short example"})  # fmt: skip

    assert revised.status_code == 200, revised.text
    revisions = (await client.get(f"/api/v1/articles/{article_id}/revisions")).json()
    assert len(revisions) == 3
    assert "editor's request: Add a short example" in revisions[2]["reason"]
    assert (await client.get(f"/api/v1/articles/{article_id}/versions/{revisions[2]['version_id']}")).json()["reason"] == revisions[2]["reason"]  # fmt: skip


async def test_refusals(client: httpx.AsyncClient) -> None:
    assert (await client.post("/api/v1/articles/999999/validate")).status_code == 404
    for path in ("quality", "fact-check", "originality", "seo", "revisions"):
        assert (await client.get(f"/api/v1/articles/999999/{path}")).status_code == 404
    article_id = await completed_article(client)
    assert (await client.post(f"/api/v1/articles/{article_id}/revise")).status_code == 409  # not validated yet  # fmt: skip
    missing = await client.get(f"/api/v1/articles/{article_id}/fact-check")
    assert missing.status_code == 404
    assert "no fact-check yet" in missing.json()["detail"]
    assert (await client.get(f"/api/v1/articles/{article_id}/quality?version_id=999999")).status_code == 404  # fmt: skip
    assert (await client.post(f"/api/v1/articles/{article_id}/validate?wait=true")).status_code == 200  # fmt: skip
    await client.post(f"/api/v1/articles/{article_id}/cancel", json={"note": "no"})
    refused = await client.post(f"/api/v1/articles/{article_id}/validate")
    assert refused.status_code == 409
    assert "cancelled" in refused.json()["detail"]


async def test_articles_still_being_written_cannot_be_validated(client: httpx.AsyncClient) -> None:
    opportunity_id = await agents_opportunity(client)
    created = (await client.post("/api/v1/articles", json={"opportunity_id": opportunity_id})).json()  # fmt: skip
    refused = await client.post(f"/api/v1/articles/{created['article']['id']}/validate")
    assert refused.status_code == 409
    await settle(client)


@pytest.fixture
async def unconfigured(db_url: str, clock: FakeClock) -> AsyncIterator[httpx.AsyncClient]:
    settings: Settings = make_settings(app_env="development", database_url=db_url)
    async for c in client_for(settings, clock, None):
        yield c


async def test_validation_needs_gemini(unconfigured: httpx.AsyncClient) -> None:
    refused = await unconfigured.post("/api/v1/articles/1/validate")
    assert refused.status_code == 503
    assert "GEMINI_API_KEY" in refused.json()["detail"]


@pytest.fixture
async def secured(db_url: str, clock: FakeClock) -> AsyncIterator[httpx.AsyncClient]:
    settings: Settings = make_settings(app_env="development", database_url=db_url, api_key="test-key")  # fmt: skip
    async for c in client_for(settings, clock, FakeLLM()):
        yield c


async def test_the_endpoints_require_the_api_key(secured: httpx.AsyncClient) -> None:
    for method, path in (("post", "validate"), ("post", "revise"), ("get", "quality"), ("get", "fact-check"), ("get", "originality"), ("get", "seo"), ("get", "revisions")):  # fmt: skip
        assert (await getattr(secured, method)(f"/api/v1/articles/1/{path}")).status_code == 401
