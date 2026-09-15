"""Phase 8 CLI: jobs, the schedule, the pipeline. The commands run jobs through the same
machinery as the scheduler (locks, limits, gates, switches), against the test database."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import cli
from tests.fakesite import make_settings

runner = CliRunner()


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch, db_url: str, tmp_path: Path) -> None:
    settings = make_settings(database_url=db_url, log_level="WARNING", full_pipeline_schedule="0 6 * * *")  # fmt: skip
    monkeypatch.setattr("app.cli.get_settings", lambda: settings)
    monkeypatch.setenv("COLUMNS", "250")


def test_plan_run_and_inspect_jobs(configured: None) -> None:
    planned = runner.invoke(cli, ["pipeline", "run", "--dry-run", "--json"])
    assert planned.exit_code == 0, planned.output
    payload = json.loads(planned.stdout)
    assert payload["job"]["dry_run"] is True
    assert payload["job"]["status"] == "completed"
    assert payload["plan"]["generation_limit"] == 3
    assert payload["plan"]["publication_limit"] == 1
    human = runner.invoke(cli, ["pipeline", "run", "--dry-run", "--job", "publish"])
    assert human.exit_code == 0
    assert "Plan for publish" in human.output
    ran = runner.invoke(cli, ["schedule", "run", "publish"])
    assert ran.exit_code == 0, ran.output
    assert "completed" in ran.output
    assert "AUTOMATED_PUBLISHING_ENABLED=false" in ran.output
    listing = runner.invoke(cli, ["jobs", "list"])
    assert listing.exit_code == 0
    assert listing.output.count("completed") >= 3
    shown = json.loads(runner.invoke(cli, ["jobs", "show", "3", "--json"]).stdout)
    assert shown["job_type"] == "publish"
    assert shown["checkpoint"] == "publishing_complete"
    filtered = json.loads(runner.invoke(cli, ["jobs", "list", "--type", "publish", "--status", "completed", "--json"]).stdout)  # fmt: skip
    assert [(j["id"], j["dry_run"]) for j in filtered] == [(3, False), (2, True)]  # newest first


def test_job_errors_are_clear(configured: None) -> None:
    unknown = runner.invoke(cli, ["jobs", "show", "999"])
    assert unknown.exit_code == 2
    assert "Unknown job 999" in unknown.output
    assert runner.invoke(cli, ["jobs", "retry", "999"]).exit_code == 2
    runner.invoke(cli, ["schedule", "run", "publish"])
    finished = runner.invoke(cli, ["jobs", "cancel", "1"])
    assert finished.exit_code == 2
    assert "only queued or running jobs" in finished.output
    not_failed = runner.invoke(cli, ["jobs", "retry", "1"])
    assert not_failed.exit_code == 2
    assert "only failed jobs" in not_failed.output
    assert runner.invoke(cli, ["schedule", "run", "social_media"]).exit_code == 2
    assert runner.invoke(cli, ["jobs", "show", "0"]).exit_code == 2


def test_status_list_pause_and_resume(configured: None) -> None:
    status = runner.invoke(cli, ["schedule", "status"])
    assert status.exit_code == 0, status.output
    assert "disabled (SCHEDULER_ENABLED=false)" in status.output
    assert "today 20" in status.output
    assert "published 0/1" in status.output
    listing = runner.invoke(cli, ["schedule", "list"])
    assert "FULL_PIPELINE_SCHEDULE" in listing.output
    assert "0 6 * * *" in listing.output
    assert runner.invoke(cli, ["schedule", "pause", "--reason", "migration"]).exit_code == 0
    paused = json.loads(runner.invoke(cli, ["schedule", "status", "--json"]).stdout)
    assert paused["paused"] is True
    assert paused["paused_reason"] == "migration"
    resumed = runner.invoke(cli, ["schedule", "resume"])
    assert "SCHEDULER_ENABLED=false" in resumed.output
    assert json.loads(runner.invoke(cli, ["schedule", "status", "--json"]).stdout)["paused"] is False  # fmt: skip
