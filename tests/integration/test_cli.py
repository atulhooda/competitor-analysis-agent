import functools
import json
from pathlib import Path

import pytest
import respx
from typer.testing import CliRunner

from app.cli import cli
from app.crawling.fetcher import PoliteFetcher
from tests.fakesite import BASE, FakeClock, make_settings, mount_site, public_resolver

runner = CliRunner()


@pytest.fixture
def configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> Path:
    path = tmp_path / "competitors.yaml"
    path.write_text(
        f"competitors:\n  - {{slug: acme, name: Acme, website: '{BASE}/', tracked_pages: ['{BASE}/pricing']}}\n",
        encoding="utf-8",
    )
    settings = make_settings(competitors_file=path, log_level="WARNING")
    monkeypatch.setattr("app.cli.get_settings", lambda: settings)
    monkeypatch.setattr(
        "app.cli.PoliteFetcher",
        functools.partial(PoliteFetcher, resolver=public_resolver, clock=clock, sleep=clock.sleep),
    )
    return path


def test_check_works_without_gemini(configured: Path) -> None:
    result = runner.invoke(cli, ["check"])
    assert result.exit_code == 0, result.output
    assert "not needed for Phase 1" in result.output
    assert "gemini-3.8-flash" in result.output


def test_check_never_prints_secret_values(
    configured: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(competitors_file=configured, gemini_api_key="super-secret-value")
    monkeypatch.setattr("app.cli.get_settings", lambda: settings)
    result = runner.invoke(cli, ["check"])
    assert result.exit_code == 0
    assert "super-secret-value" not in result.output


def test_scan_json_output(configured: Path) -> None:
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        result = runner.invoke(cli, ["scan", "acme", "--since", "2026-09-01", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["competitor"] == "acme"
    assert f"{BASE}/blog/ai-support-agents" in [i["final_url"] for i in payload["items"]]


def test_scan_table_output(configured: Path) -> None:
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        result = runner.invoke(cli, ["scan", "acme", "--since", "2026-09-01"])
    assert result.exit_code == 0, result.output
    assert "AI Support Agents" in result.output
    assert "blog_post" in result.output


def test_unknown_competitor_and_bad_since(configured: Path) -> None:
    assert runner.invoke(cli, ["scan", "nope"]).exit_code == 2
    assert runner.invoke(cli, ["scan", "acme", "--since", "whenever"]).exit_code == 2
