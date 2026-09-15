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
    assert "scanning works without it" in result.output
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


# ── Phase 3 ──────────────────────────────────────────────────────────────────


def _scan_acme() -> None:
    assert runner.invoke(cli, ["competitors", "import"]).exit_code == 0
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        assert runner.invoke(cli, ["scan", "acme"]).exit_code == 0


def test_analyze_dry_run_needs_no_key_and_real_runs_do(configured: Path) -> None:
    _scan_acme()
    dry = runner.invoke(cli, ["analyze", "acme", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "dry run" in dry.output
    assert "pending 8" in dry.output
    plan = json.loads(runner.invoke(cli, ["analyze", "acme", "--dry-run", "--json"]).stdout)
    assert plan["batches"] == 2
    assert len(plan["items"]) == 8

    refused = runner.invoke(cli, ["analyze", "acme"])
    assert refused.exit_code == 2
    assert "GEMINI_API_KEY" in refused.output
    assert runner.invoke(cli, ["analyze"]).exit_code == 2  # neither slug nor --all


def test_analysis_commands(configured: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:  # fmt: skip
    from app.llm import LazyLLM
    from tests.fakellm import FakeLLM

    fake = FakeLLM()
    monkeypatch.setattr("app.cli.LazyLLM", functools.partial(LazyLLM, provider=fake))
    _scan_acme()

    analyzed = runner.invoke(cli, ["analyze", "--all"])
    assert analyzed.exit_code == 0, analyzed.output
    assert "succeeded" in analyzed.output
    assert "analyzed=8" in analyzed.output
    assert "gemini: 3 call(s)" in analyzed.output  # 2 analysis batches + 1 profile

    payload = json.loads(runner.invoke(cli, ["analyze", "acme", "--json"]).stdout)
    assert payload["status"] == "succeeded"
    assert payload["summary"]["analyzed"] == 0

    topics = runner.invoke(cli, ["topics"])
    assert topics.exit_code == 0
    assert "ai-agents" in topics.output
    shown = runner.invoke(cli, ["topics", "show", "ai-agents"])
    assert shown.exit_code == 0, shown.output
    assert "AI Agents" in shown.output or "AI agents" in shown.output
    assert "Ticket automation" in shown.output
    assert runner.invoke(cli, ["topics", "list", "--parent", "ai-agents"]).exit_code == 0

    trends = runner.invoke(cli, ["trends", "acme"])
    assert trends.exit_code == 0, trends.output
    assert "published: 3 in the last 30 days" in trends.output
    assert runner.invoke(cli, ["trends"]).exit_code == 0
    assert runner.invoke(cli, ["trends", "nope"]).exit_code == 2

    profile = runner.invoke(cli, ["profile", "acme"])
    assert profile.exit_code == 0, profile.output
    assert "Resolve support tickets with AI agents" in profile.output
    assert "2 statement(s) without evidence were dropped" in profile.output

    assert runner.invoke(cli, ["landscape"]).exit_code == 2  # none generated yet
    landscape = runner.invoke(cli, ["landscape", "--refresh"])
    assert landscape.exit_code == 0, landscape.output
    assert "AI agents dominate." in landscape.output

    item_id = json.loads(runner.invoke(cli, ["content", "acme", "--json", "--search", "Support Agents"]).stdout)[0]["id"]  # fmt: skip
    analysis = runner.invoke(cli, ["analysis", str(item_id)])
    assert analysis.exit_code == 0, analysis.output
    assert "content-analysis/2" in analysis.output
    assert "Human handoff" in analysis.output
    assert runner.invoke(cli, ["analysis", "999999"]).exit_code == 2

    usage = runner.invoke(cli, ["usage"])
    assert usage.exit_code == 0
    assert "content_analysis" in usage.output
    runs = runner.invoke(cli, ["runs"])
    assert "analysis" in runs.output
    assert "landscape" in runs.output

    merged = runner.invoke(cli, ["topics", "merge", "automation", "ai-agents"])
    assert merged.exit_code == 0, merged.output
    assert runner.invoke(cli, ["topics", "merge", "nope", "ai-agents"]).exit_code == 2

    seeds = tmp_path / "topics.yaml"
    seeds.write_text("topics:\n  - {name: Data privacy, aliases: [GDPR compliance]}\n  - {name: Pricing, aliases: [Automation]}\n", encoding="utf-8")  # fmt: skip
    imported = runner.invoke(cli, ["topics", "import", "--file", str(seeds)])
    assert imported.exit_code == 0, imported.output
    assert "topics created: 1" in imported.output
    assert "already means" in imported.output  # "Automation" was merged into AI agents


def test_topics_example_file_is_valid() -> None:
    from app.config import load_topic_seeds

    seeds = load_topic_seeds(Path(__file__).parents[2] / "config" / "topics.example.yaml")
    assert any(seed.name == "AI agents" for seed in seeds)


# ── Phase 4 ──────────────────────────────────────────────────────────────────


def test_company_and_opportunity_commands(configured: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:  # fmt: skip
    from app.llm import LazyLLM
    from tests.fakellm import FakeLLM

    fake = FakeLLM()
    monkeypatch.setattr("app.cli.LazyLLM", functools.partial(LazyLLM, provider=fake))
    _scan_acme()
    assert runner.invoke(cli, ["analyze", "acme", "--no-profile"]).exit_code == 0

    assert runner.invoke(cli, ["opportunities", "generate"]).exit_code == 2  # no company profile
    company = tmp_path / "company.yaml"
    company.write_text(
        "company:\n  name: Example Startup\n  description: Helps founders deploy AI agents.\n"
        "  target_audiences: [founders]\n  core_topics: [AI agents]\n  excluded_topics: [Pricing]\n",
        encoding="utf-8",
    )
    imported = runner.invoke(cli, ["company", "import", "--file", str(company)])
    assert imported.exit_code == 0, imported.output
    assert "company profile v1 created" in imported.output
    assert "unchanged" in runner.invoke(cli, ["company", "import", "--file", str(company)]).output
    assert "Example Startup" in runner.invoke(cli, ["company", "show"]).output
    versions = runner.invoke(cli, ["company", "versions"]).output
    assert "Example Startup" in versions
    assert "file" in versions

    generated = runner.invoke(cli, ["opportunities", "generate"])
    assert generated.exit_code == 0, generated.output
    assert "succeeded" in generated.output
    assert "AI agents" in generated.output
    payload = json.loads(runner.invoke(cli, ["opportunities", "list", "--json"]).stdout)
    agents = next(o for o in payload if o["topic_label"].casefold() == "ai agents")
    shown = runner.invoke(cli, ["opportunities", "show", str(agents["id"])])
    assert shown.exit_code == 0, shown.output
    assert "strategic fit" in shown.output
    assert "Why" in shown.output
    assert "Gemini" in shown.output
    evidence = runner.invoke(cli, ["opportunities", "evidence", str(agents["id"])])
    assert "topic_metrics" in evidence.output
    assert "content" in evidence.output
    assert "first assessment" in runner.invoke(cli, ["opportunities", "history", str(agents["id"])]).output  # fmt: skip
    assert runner.invoke(cli, ["opportunities", "approve", str(agents["id"]), "--note", "go"]).exit_code == 0  # fmt: skip
    assert runner.invoke(cli, ["opportunities", "reopen", str(agents["id"])]).exit_code == 2  # approved → new: invalid  # fmt: skip
    assert runner.invoke(cli, ["opportunities", "show", "999999"]).exit_code == 2
    listing = runner.invoke(cli, ["opportunities"])
    assert listing.exit_code == 0
    assert "approved" in listing.output
    assert "opportunities" in runner.invoke(cli, ["runs"]).output


# ── Phase 5 ──────────────────────────────────────────────────────────────────


def test_article_commands(configured: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:  # fmt: skip
    import yaml

    from app.llm import LazyLLM
    from app.services.articles import ArticleService
    from tests.fakellm import FakeLLM
    from tests.pipeline import ARTICLE_COMPANY

    fake = FakeLLM()
    monkeypatch.setattr("app.cli.LazyLLM", functools.partial(LazyLLM, provider=fake))
    monkeypatch.setattr("app.cli.ArticleService", functools.partial(ArticleService, resolver=public_resolver))  # fmt: skip
    _scan_acme()
    assert runner.invoke(cli, ["analyze", "acme", "--no-profile"]).exit_code == 0
    company = tmp_path / "company.yaml"
    company.write_text(yaml.safe_dump({"company": ARTICLE_COMPANY}), encoding="utf-8")
    assert runner.invoke(cli, ["company", "import", "--file", str(company)]).exit_code == 0
    assert runner.invoke(cli, ["opportunities", "generate"]).exit_code == 0
    payload = json.loads(runner.invoke(cli, ["opportunities", "list", "--json"]).stdout)
    opportunity = str(next(o["id"] for o in payload if o["topic_label"].casefold() == "ai agents"))  # fmt: skip

    refused = runner.invoke(cli, ["articles", "generate", opportunity])
    assert refused.exit_code == 2  # not approved yet
    assert "only approved opportunities" in refused.output
    brief = runner.invoke(cli, ["articles", "brief", opportunity])
    assert brief.exit_code == 0, brief.output
    assert "Angle:" in brief.output
    assert "provenance" in brief.output
    assert json.loads(runner.invoke(cli, ["articles", "brief", opportunity, "--json"]).stdout)["opportunity_id"] == int(opportunity)  # fmt: skip
    assert runner.invoke(cli, ["opportunities", "approve", opportunity]).exit_code == 0

    generated = runner.invoke(cli, ["articles", "generate", opportunity])
    assert generated.exit_code == 0, generated.output
    assert "article completed" in generated.output
    articles = json.loads(runner.invoke(cli, ["articles", "list", "--json"]).stdout)
    article = str(articles[0]["id"])
    assert articles[0]["status"] == "completed"
    shown = runner.invoke(cli, ["articles", "show", article])
    assert shown.exit_code == 0, shown.output
    assert "edited article (preview; not published)" in shown.output
    assert "## Sources" in shown.output
    assert "article-edit/1" in shown.output
    sources = runner.invoke(cli, ["articles", "sources", article])
    assert "S1" in sources.output
    assert "competitor or company source" in sources.output
    versions = json.loads(runner.invoke(cli, ["articles", "versions", article, "--json"]).stdout)
    assert [v["kind"] for v in versions] == ["outline", "draft", "final"]
    draft = runner.invoke(cli, ["articles", "versions", article, "--show", str(versions[1]["id"])])
    assert draft.exit_code == 0
    assert "draft v1" in draft.output
    assert "edit" in runner.invoke(cli, ["articles", "steps", article]).output
    assert "nothing to do" in runner.invoke(cli, ["articles", "resume", article]).output
    again = runner.invoke(cli, ["articles", "generate", opportunity])
    assert "already exists" in again.output
    assert "cancelled" in runner.invoke(cli, ["articles", "cancel", article]).output
    assert "cancelled" in runner.invoke(cli, ["articles"]).output
    assert runner.invoke(cli, ["articles", "show", "999999"]).exit_code == 2
    assert "article" in runner.invoke(cli, ["runs"]).output
