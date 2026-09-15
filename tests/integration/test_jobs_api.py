"""Phase 8 HTTP API: jobs, the schedule and starting the pipeline. The API goes through the
same job machinery as the scheduler: it is never a way around a lock, a limit or a switch."""

from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy import update

from app.config import Settings
from app.db.locks import job_type_lock
from app.db.models import Job
from app.domain.jobs import JobTrigger, JobType
from tests.fakellm import FakeLLM
from tests.fakesite import FakeClock, make_settings
from tests.integration.test_articles_api import client_for, settle

SECRET = "api-key-that-must-never-leak"


@pytest.fixture
async def client(db_url: str, clock: FakeClock) -> AsyncIterator[httpx.AsyncClient]:
    settings: Settings = make_settings(app_env="development", database_url=db_url, full_pipeline_schedule="0 6 * * *")  # fmt: skip
    async for c in client_for(settings, clock, FakeLLM()):
        yield c


async def _set(client: httpx.AsyncClient, job_id: int, **values: object) -> None:
    sessions = client.app.state.sessions  # type: ignore[attr-defined]
    async with sessions() as session, session.begin():
        await session.execute(update(Job).where(Job.id == job_id).values(**values))


async def test_a_dry_run_returns_the_plan_and_changes_nothing(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/pipeline/run", json={"dry_run": True})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["job"]["dry_run"] is True
    assert body["job"]["status"] == "completed"
    assert body["plan"]["stages"] == ["scan", "analyze", "opportunities", "generate", "quality", "approval", "publish"]  # fmt: skip
    assert "planning mode" in body["plan"]["notes"][0]
    assert response.headers["location"] == f"/api/v1/jobs/{body['job']['id']}"


async def test_a_pipeline_run_is_queued_and_runs_in_the_background(client: httpx.AsyncClient) -> None:  # fmt: skip
    response = await client.post("/api/v1/pipeline/run", json={"job_type": "publish"})
    assert response.status_code == 202, response.text
    job = response.json()["job"]
    assert (job["status"], job["trigger"], job["job_type"]) == ("queued", "api", "publish")
    await settle(client)
    done = (await client.get(f"/api/v1/jobs/{job['id']}")).json()
    assert done["status"] == "completed"
    assert [s["stage"] for s in done["stages"]] == ["approval", "publish"]
    assert done["stages"][1]["status"] == "skipped"  # AUTOMATED_PUBLISHING_ENABLED=false
    assert done["checkpoint"] == "publishing_complete"
    listing = (await client.get("/api/v1/jobs", params={"job_type": "publish", "status": "completed"})).json()  # fmt: skip
    assert [j["id"] for j in listing] == [job["id"]]


async def test_a_second_run_of_a_running_job_type_is_skipped(client: httpx.AsyncClient) -> None:
    jobs = client.app.state.jobs  # type: ignore[attr-defined]
    running, _ = await jobs.enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.SCHEDULE)
    async with job_type_lock(client.app.state.engine, 7) as held:  # type: ignore[attr-defined]
        assert held  # as if another process were running a full pipeline
        response = await client.post("/api/v1/pipeline/run", json={})
        assert response.status_code == 202
        await settle(client)
    skipped = (await client.get(f"/api/v1/jobs/{response.json()['job']['id']}")).json()
    assert skipped["status"] == "skipped"
    assert "another full_pipeline job is running" in skipped["last_error"]
    assert running.status.value == "queued"


async def test_requests_are_validated(client: httpx.AsyncClient) -> None:
    assert (await client.post("/api/v1/pipeline/run", json={"job_type": "social_media"})).status_code == 422  # fmt: skip
    assert (await client.post("/api/v1/pipeline/run", json={"dry_run": True, "script": "rm -rf /"})).status_code == 422  # fmt: skip
    assert (await client.get("/api/v1/jobs/0")).status_code == 422
    assert (await client.get("/api/v1/jobs/abc")).status_code == 422
    assert (await client.get("/api/v1/jobs", params={"status": "exploded"})).status_code == 422
    assert (await client.get("/api/v1/jobs", params={"limit": 10_000})).status_code == 422
    assert (await client.post("/api/v1/schedule/pause", json={"reason": "x" * 501})).status_code == 422  # fmt: skip
    missing = await client.get("/api/v1/jobs/999")
    assert missing.status_code == 404
    assert "Unknown job 999" in missing.json()["detail"]


async def test_cancel_and_retry(client: httpx.AsyncClient) -> None:
    jobs = client.app.state.jobs  # type: ignore[attr-defined]
    queued, _ = await jobs.enqueue(JobType.SCAN, trigger=JobTrigger.API)
    cancelled = await client.post(f"/api/v1/jobs/{queued.id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert (await client.post(f"/api/v1/jobs/{queued.id}/cancel")).status_code == 409
    assert (await client.post(f"/api/v1/jobs/{queued.id}/retry")).status_code == 409  # not failed
    failed, _ = await jobs.enqueue(JobType.PUBLISH, trigger=JobTrigger.SCHEDULE)
    await _set(client, failed.id, status="failed", error_kind="transient", last_error="publish: CMS down", details={"params": {}, "stages": {"approval": {"status": "completed", "summary": {"ready": 0}}}, "checkpoint": "approval_complete"})  # fmt: skip
    retried = await client.post(f"/api/v1/jobs/{failed.id}/retry")
    assert retried.status_code == 202, retried.text
    body = retried.json()
    assert (body["parent_id"], body["trigger"], body["checkpoint"]) == (failed.id, "retry", "approval_complete")  # fmt: skip
    assert retried.headers["location"] == f"/api/v1/jobs/{body['id']}"
    await settle(client)
    done = (await client.get(f"/api/v1/jobs/{body['id']}")).json()
    assert done["status"] == "completed"
    assert [s["stage"] for s in done["stages"]] == ["approval", "publish"]
    assert (await client.post("/api/v1/jobs/999/retry")).status_code == 404


async def test_the_schedule_and_its_status(client: httpx.AsyncClient) -> None:
    schedule = (await client.get("/api/v1/schedule")).json()
    assert [(s["job_type"], s["setting"], s["expression"], s["timezone"]) for s in schedule] == [("full_pipeline", "FULL_PIPELINE_SCHEDULE", "0 6 * * *", "Asia/Kolkata")]  # fmt: skip
    assert len(schedule[0]["next_runs"]) == 3
    status = (await client.get("/api/v1/schedule/status")).json()
    assert status["enabled"] is False
    assert status["automated_publishing"] is False
    assert status["today"]["publication_limit"] == 1
    assert status["today"]["remaining"] == 1
    assert any("SCHEDULER_ENABLED=false" in w for w in status["warnings"])
    paused = (await client.post("/api/v1/schedule/pause", json={"reason": "migration"})).json()
    assert paused["paused"] is True
    assert paused["paused_reason"] == "migration"
    assert (await client.get("/api/v1/schedule/status")).json()["paused"] is True
    resumed = (await client.post("/api/v1/schedule/resume")).json()
    assert resumed["paused"] is False
    health = (await client.get("/health")).json()
    assert health["scheduler"] == {"enabled": False, "automated_publishing": False}


async def test_the_endpoints_need_the_api_key_and_never_echo_it(db_url: str, clock: FakeClock) -> None:  # fmt: skip
    settings = make_settings(app_env="production", database_url=db_url, api_key=SECRET)
    async for c in client_for(settings, clock, FakeLLM()):
        for method, path in (("GET", "/api/v1/jobs"), ("POST", "/api/v1/pipeline/run"), ("GET", "/api/v1/schedule/status"), ("POST", "/api/v1/schedule/pause")):  # fmt: skip
            assert (await c.request(method, path)).status_code == 401
        allowed = await c.post("/api/v1/pipeline/run", json={"dry_run": True}, headers={"X-API-Key": SECRET})  # fmt: skip
        assert allowed.status_code == 200
        assert SECRET not in allowed.text
        status = await c.get("/api/v1/schedule/status", headers={"X-API-Key": SECRET})
        assert SECRET not in status.text
