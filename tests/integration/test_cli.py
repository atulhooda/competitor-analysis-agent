import functools
import json
from pathlib import Path

import pytest
import respx
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import NullPool
from typer.testing import CliRunner

from app.cli import cli
from app.crawling.fetcher import PoliteFetcher
from app.db.models import ContentItem
from tests.fakesite import BASE, FakeClock, make_settings, mount_site, public_resolver

runner = CliRunner()


def count_items(db_url: str) -> int:
    engine = create_engine(db_url, poolclass=NullPool)
    with engine.connect() as conn:
        total = conn.execute(select(func.count()).select_from(ContentItem)).scalar_one()
    engine.dispose()
    return int(total)


@pytest.fixture
def configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, db_url: str
) -> Path:
    path = tmp_path / "competitors.yaml"
    path.write_text(
        f"competitors:\n  - {{slug: acme, name: Acme, website: '{BASE}/', tracked_pages: ['{BASE}/pricing']}}\n",
        encoding="utf-8",
    )
    settings = make_settings(competitors_file=path, log_level="WARNING", database_url=db_url)
    monkeypatch.setattr("app.cli.get_settings", lambda: settings)
    monkeypatch.setenv("COLUMNS", "250")  # wide enough that Rich doesn't truncate table cells
    monkeypatch.setattr(
        "app.cli.PoliteFetcher",
        functools.partial(PoliteFetcher, resolver=public_resolver, clock=clock, sleep=clock.sleep),
    )
    return path


def test_check(configured: Path) -> None:
    result = runner.invoke(cli, ["check"])
    assert result.exit_code == 0, result.output
    assert "up to date" in result.output
    assert "not needed for Phases 1 and 2" in result.output
    assert "gemini-3.8-flash" in result.output


def test_check_never_prints_secret_values(
    configured: Path, monkeypatch: pytest.MonkeyPatch, db_url: str
) -> None:
    settings = make_settings(
        competitors_file=configured, gemini_api_key="super-secret-value", database_url=db_url
    )
    monkeypatch.setattr("app.cli.get_settings", lambda: settings)
    result = runner.invoke(cli, ["check"])
    assert "super-secret-value" not in result.output


def test_import_scan_and_query_history(configured: Path, db_url: str) -> None:
    imported = runner.invoke(cli, ["competitors", "import"])
    assert imported.exit_code == 0, imported.output
    assert "created: acme" in imported.output
    assert "updated: acme" in runner.invoke(cli, ["competitors", "import"]).output

    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        scanned = runner.invoke(cli, ["scan", "acme", "--json"])
    assert scanned.exit_code == 0, scanned.output
    payload = json.loads(scanned.stdout)
    assert payload["status"] == "succeeded"
    assert payload["changes"]["baseline"] is True
    assert count_items(db_url) >= 9

    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        rescanned = runner.invoke(cli, ["scan", "--all"])
    assert rescanned.exit_code == 0, rescanned.output
    assert "new=0" in rescanned.output
    assert "updated=0" in rescanned.output

    listing = runner.invoke(cli, ["competitors"])
    assert "acme" in listing.output
    content = runner.invoke(cli, ["content", "acme", "--type", "blog_post"])
    assert content.exit_code == 0, content.output
    assert "AI Support Agents" in content.output
    assert "structured_data" in content.output
    assert "No changes" in runner.invoke(cli, ["changes", "acme"]).output
    runs = runner.invoke(cli, ["runs", "acme"])
    assert runs.output.count("succeeded") == 2
    run = json.loads(runner.invoke(cli, ["run", str(payload["run_id"])]).stdout)
    assert run["competitor"] == "acme"
    activity = runner.invoke(cli, ["activity", "acme", "--weeks", "2"])
    assert activity.exit_code == 0, activity.output
    assert "reliable dates only" in activity.output

    assert runner.invoke(cli, ["competitors", "deactivate", "acme"]).exit_code == 0
    assert runner.invoke(cli, ["scan", "acme"]).exit_code == 2  # inactive


def test_dry_run_saves_nothing(configured: Path, db_url: str) -> None:
    runner.invoke(cli, ["competitors", "import"])
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        result = runner.invoke(cli, ["scan", "acme", "--dry-run", "--since", "2026-09-01"])
    assert result.exit_code == 0, result.output
    assert "AI Support Agents" in result.output
    assert count_items(db_url) == 0


def test_errors(configured: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert runner.invoke(cli, ["scan", "nope"]).exit_code == 2
    assert runner.invoke(cli, ["scan", "acme", "--since", "whenever"]).exit_code == 2
    assert runner.invoke(cli, ["scan"]).exit_code == 2  # neither slug nor --all

    unreachable = make_settings(database_url="postgresql+psycopg://postgres@127.0.0.1:1/nothing")
    monkeypatch.setattr("app.cli.get_settings", lambda: unreachable)
    result = runner.invoke(cli, ["competitors", "list"])
    assert result.exit_code == 2
    assert "Cannot reach the database" in result.output
