"""Editorial topics over HTTP: proposing ideas (synchronously or in the background), reading a
proposal run, and listing the resulting opportunities by origin. Fake Gemini; no network."""

from collections.abc import AsyncIterator

import httpx
import pytest

from tests.fakellm import FakeLLM
from tests.fakesite import FakeClock, make_settings
from tests.integration.test_articles_api import settle
from tests.integration.test_opportunities_api import client_for
from tests.pipeline import ARTICLE_COMPANY


@pytest.fixture
async def client(db_url: str, clock: FakeClock) -> AsyncIterator[httpx.AsyncClient]:
    settings = make_settings(app_env="development", database_url=db_url)
    async for c in client_for(settings, clock, FakeLLM()):
        yield c


async def test_ideas_are_proposed_and_listed_by_origin(client: httpx.AsyncClient) -> None:
    refused = await client.post("/api/v1/editorial/propose?wait=true", json={"count": 2})
    assert refused.status_code == 409  # no company profile yet
    assert (await client.put("/api/v1/company-profile", json=ARTICLE_COMPANY)).status_code == 200

    response = await client.post("/api/v1/editorial/propose?wait=true", json={"count": 2})
    assert response.status_code == 200, response.text
    body = response.json()
    assert response.headers["Location"] == f"/api/v1/editorial/runs/{body['run_id']}"
    assert body["status"] == "succeeded"
    assert body["summary"]["created"] == 2
    created = [i for i in body["ideas"] if i["opportunity_id"]]
    assert [i["topic"] for i in created] == ["AI agent handoff", "Evaluating AI agents"]
    rejected = {i["topic"]: i["rejected"] for i in body["ideas"] if i["rejected"]}
    assert rejected["AI agent pricing"] == "excluded by your company profile ('Pricing')"

    run = await client.get(f"/api/v1/editorial/runs/{body['run_id']}")
    assert run.status_code == 200
    assert run.json() == body
    editorial = (await client.get("/api/v1/opportunities", params={"origin": "editorial"})).json()
    assert sorted(o["id"] for o in editorial) == sorted(i["opportunity_id"] for i in created)
    assert {o["origin"] for o in editorial} == {"editorial"}
    assert (await client.get("/api/v1/opportunities", params={"origin": "competitors"})).json() == []  # fmt: skip


async def test_a_background_proposal_and_a_dry_run(client: httpx.AsyncClient) -> None:
    assert (await client.put("/api/v1/company-profile", json=ARTICLE_COMPANY)).status_code == 200
    dry = await client.post("/api/v1/editorial/propose?wait=true", json={"count": 1, "dry_run": True})  # fmt: skip
    assert dry.status_code == 200
    assert dry.json()["summary"]["dry_run"] is True
    assert (await client.get("/api/v1/opportunities", params={"origin": "editorial"})).json() == []  # fmt: skip

    queued = await client.post("/api/v1/editorial/propose", json={"count": 1})
    assert queued.status_code == 202
    assert queued.json()["status"] in ("queued", "running", "succeeded")
    await settle(client)
    done = (await client.get(queued.headers["Location"])).json()
    assert done["status"] == "succeeded"
    assert len((await client.get("/api/v1/opportunities", params={"origin": "editorial"})).json()) == 1  # fmt: skip


async def test_bad_requests_and_unknown_runs(client: httpx.AsyncClient) -> None:
    assert (await client.post("/api/v1/editorial/propose", json={"count": 26})).status_code == 422
    assert (await client.post("/api/v1/editorial/propose", json={"topics": ["x"]})).status_code == 422  # fmt: skip
    assert (await client.get("/api/v1/editorial/runs/999999")).status_code == 404
