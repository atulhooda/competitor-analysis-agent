"""Command-line interface: ``uv run python -m app --help``."""

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.cms import LazyCMS
from app.cms.errors import CMSConfigurationError
from app.config import get_settings, load_company_profile, load_competitors, load_topic_seeds
from app.core.errors import ConfigurationError
from app.core.logging import configure_logging
from app.core.timeutils import parse_since, utcnow
from app.crawling.fetcher import PoliteFetcher
from app.db import (
    analysis_queries,
    article_queries,
    migrate,
    opportunity_queries,
    publishing_queries,
    quality_queries,
    queries,
)
from app.db.session import SessionFactory, create_engine, create_session_factory
from app.domain.analysis import MixShift, Share, TopicTrend
from app.domain.articles import ArticleBrief, ArticleOrigin, ArticleRunView, ArticleStatus
from app.domain.competitor_profile import Claim
from app.domain.content import ContentType
from app.domain.history import ChangeType, RunStatus, RunTrigger
from app.domain.jobs import (
    JobStatus,
    JobTrigger,
    JobType,
    JobView,
    PipelinePlan,
    SchedulerStatus,
    ScheduleView,
)
from app.domain.opportunities import OpportunityOrigin, OpportunityStatus, OpportunitySummary
from app.domain.publishing import (
    ApprovalChannel,
    ApprovalRecord,
    ApprovalView,
    DryRunReport,
    PreflightReport,
    PublicationView,
    TargetStatus,
)
from app.domain.quality import ClaimVerdict, GateStatus
from app.domain.scan import ScanResult
from app.llm import LazyLLM, LLMConfigurationError
from app.scheduling.runtime import Scheduling, standalone
from app.scheduling.worker import run_worker
from app.services.analysis import (
    AnalysisAlreadyRunningError,
    AnalysisOptions,
    AnalysisOutcome,
    AnalysisService,
)
from app.services.approvals import ApprovalService
from app.services.article_file import ArticleFileError
from app.services.article_import import ArticleImportService, ImportOutcome
from app.services.articles import (
    ArticleBudgetExhaustedError,
    ArticleConflictError,
    ArticleNotFoundError,
    ArticleOutcome,
    ArticleRequestResult,
    ArticleRunActiveError,
    ArticleService,
    OpportunityNotApprovedError,
)
from app.services.company import (
    company_view,
    latest_company_profile,
    list_company_profiles,
    save_company_profile,
)
from app.services.covers import CoverService
from app.services.editorial import EditorialRunAlreadyActiveError, EditorialService, ProposalOutcome
from app.services.intelligence import IntelligenceService
from app.services.jobs import JobConflictError, JobNotFoundError
from app.services.landscape import LandscapeAlreadyRunningError, LandscapeService
from app.services.llm_usage import usage_window_start, utc_day_start
from app.services.monitoring import MonitoringService
from app.services.opportunities import (
    GenerationOptions,
    GenerationOutcome,
    InvalidStatusTransitionError,
    NoCompanyProfileError,
    OpportunityNotFoundError,
    OpportunityRunAlreadyActiveError,
    OpportunityService,
)
from app.services.publishing import PublishingService, PublishOutcome, PublishRequestResult
from app.services.quality import QualityOutcome, QualityService
from app.services.scans import CompetitorNotFoundError, ScanError, ScanOutcome, ScanService
from app.services.topic_admin import TopicAdminService
from app.services.topics import TopicMergeError

cli = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Competitor intelligence agent: compliant website monitoring, persisted history, "
    "AI analysis (Gemini), content opportunities, article drafts, their validation, "
    "publishing approved, ready versions (WordPress drafts by default), and the scheduled "
    "pipeline that runs them (off by default).",
)
db_cli = typer.Typer(no_args_is_help=True, help="Database migrations.")
competitors_cli = typer.Typer(help="Manage monitored competitors (stored in the database).")
topics_cli = typer.Typer(help="The topic taxonomy: list, inspect, seed, merge, consolidate.")
company_cli = typer.Typer(no_args_is_help=True, help="Your company profile (what opportunities are scored against).")  # fmt: skip
opportunities_cli = typer.Typer(help="Content opportunities: generate, rank, inspect, decide.")
editorial_cli = typer.Typer(help="Editorial topics: article ideas from your company profile alone (no competitor content).")  # fmt: skip
articles_cli = typer.Typer(
    help="Article drafts from approved opportunities. Drafts only: nothing is published."
)
cli.add_typer(db_cli, name="db")
cli.add_typer(competitors_cli, name="competitors")
cli.add_typer(topics_cli, name="topics")
cli.add_typer(company_cli, name="company")
cli.add_typer(opportunities_cli, name="opportunities")
cli.add_typer(editorial_cli, name="editorial")
cli.add_typer(articles_cli, name="articles")
console = Console()
err = Console(stderr=True)


@cli.callback()
def _main(
    log_level: str | None = typer.Option(None, help="Override LOG_LEVEL (DEBUG, INFO, WARNING)."),
) -> None:
    settings = get_settings()
    configure_logging(log_level or settings.log_level, json_output=settings.log_json)


# ── plumbing ─────────────────────────────────────────────────────────────────


def _run_db[T](work: Callable[[AsyncEngine, SessionFactory], Awaitable[T]]) -> T:
    """Run ``work`` with a short-lived engine; turn database problems into clear messages."""
    settings = get_settings()

    async def runner() -> T:
        engine = create_engine(settings, pooled=False)
        try:
            return await work(engine, create_session_factory(engine))
        finally:
            await engine.dispose()

    try:
        return asyncio.run(runner())
    except ProgrammingError as exc:
        err.print(f"[red]Database schema problem:[/red] {exc.orig}")
        err.print("Run [bold]uv run python -m app db upgrade[/bold] to apply migrations.")
        raise typer.Exit(code=2) from exc
    except (OperationalError, DBAPIError, OSError) as exc:
        err.print(f"[red]Cannot reach the database[/red] at {settings.database_url_display}")
        err.print("Start it with [bold]docker compose up -d db[/bold] or set DATABASE_URL.")
        raise typer.Exit(code=2) from exc


def _since(value: str | None) -> datetime | None:
    try:
        return parse_since(value) if value else None
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--since") from exc


def _date(value: datetime | None) -> str:
    return f"{value:%Y-%m-%d}" if value else "—"


# ── check / db ───────────────────────────────────────────────────────────────


@cli.command()
def check() -> None:
    """Validate configuration and database. Secret values are never printed."""
    settings = get_settings()
    table = Table(title="Configuration", show_header=False)
    table.add_column("Setting")
    table.add_column("Value")
    ok = True
    table.add_row("Environment", settings.app_env)
    table.add_row("Database", settings.database_url_display)
    try:
        revision = migrate.current_revision(settings.database_url.get_secret_value())
        head = migrate.head_revision()
        state = "up to date" if revision == head else f"[yellow]needs `db upgrade`[/yellow] ({revision} → {head})"  # fmt: skip
        table.add_row("Schema", f"revision {revision} ({state})")
        ok = ok and revision == head
    except Exception as exc:  # report, don't crash: this command diagnoses problems
        ok = False
        table.add_row("Schema", f"[red]unreachable[/red] ({type(exc).__name__})")
    table.add_row("Crawler user agent", settings.crawler_user_agent)
    table.add_row("Crawler minimum delay", f"{settings.crawler_min_delay_seconds}s per host")
    table.add_row("LLM provider", "Google Gemini")
    table.add_row("GEMINI_MODEL", settings.gemini_model)
    table.add_row(
        "GEMINI_API_KEY",
        "set"
        if settings.llm_configured
        else "[yellow]not set[/yellow] (scanning works without it; AI analysis requires it)",
    )
    table.add_row("Analysis model", f"{settings.analysis_model} (reasoning: {settings.analysis_reasoning_effort})")  # fmt: skip
    table.add_row("Synthesis model", f"{settings.synthesis_model} (reasoning: {settings.synthesis_reasoning_effort})")  # fmt: skip
    table.add_row("Writing model", f"{settings.writing_model} (writing: {settings.writing_reasoning_effort}; research: {settings.research_reasoning_effort})")  # fmt: skip
    table.add_row("Article budget", f"{settings.article_max_tokens:,} tokens per article ({settings.article_research_max_tokens:,} for research)")  # fmt: skip
    budget = settings.llm_daily_token_budget
    table.add_row("LLM token budgets", f"{settings.llm_max_tokens_per_run:,} per run; {f'{budget:,}' if budget else 'unlimited'} per day")  # fmt: skip
    table.add_row("API_KEY", "set" if settings.api_key else "not set (allowed in development only)")
    console.print(table)
    raise typer.Exit(code=0 if ok else 1)


@db_cli.command("upgrade")
def db_upgrade(revision: str = typer.Argument("head")) -> None:
    """Apply database migrations (creates the schema on a fresh database)."""
    settings = get_settings()
    try:
        migrate.upgrade(settings.database_url.get_secret_value(), revision)
    except (OperationalError, DBAPIError, OSError) as exc:
        err.print(f"[red]Cannot reach the database[/red] at {settings.database_url_display}")
        raise typer.Exit(code=2) from exc
    current = migrate.current_revision(settings.database_url.get_secret_value())
    console.print(f"Database at revision [bold]{current}[/bold] ({settings.database_url_display})")


@db_cli.command("current")
def db_current() -> None:
    """Show the database's migration revision."""
    settings = get_settings()
    current = migrate.current_revision(settings.database_url.get_secret_value())
    console.print(f"current: {current}  head: {migrate.head_revision()}")


# ── competitors ──────────────────────────────────────────────────────────────


@competitors_cli.callback(invoke_without_command=True)
def competitors_main(ctx: typer.Context) -> None:
    """List competitors when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        list_competitors(include_inactive=True)


@competitors_cli.command("list")
def list_competitors(include_inactive: bool = typer.Option(True, "--all/--active")) -> None:
    """List monitored competitors with their content counts and last scan."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            views = await queries.list_competitors(session, include_inactive=include_inactive)
        table = Table("Slug", "Name", "Website", "Active", "Items", "Last scan")
        for v in views:
            last = f"{_date(v.last_run_at)} {v.last_run_status or ''}".strip()
            table.add_row(v.slug, v.name, v.website, "yes" if v.active else "no", str(v.content_items), last)  # fmt: skip
        console.print(table if views else "No competitors yet: run `competitors import`.")

    _run_db(work)


@competitors_cli.command("import")
def import_competitors(
    file: Annotated[
        Path | None, typer.Option(help="YAML file (default: COMPETITORS_FILE).")
    ] = None,
) -> None:
    """Create or update competitors from a YAML file (matched by slug)."""
    settings = get_settings()
    try:
        configs = load_competitors(file or settings.competitors_file)
    except ConfigurationError as exc:
        err.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    async def work(_engine: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session, session.begin():
            for config in configs:
                _competitor, created = await queries.upsert_competitor(session, config)
                console.print(f"{'created' if created else 'updated'}: {config.slug}")

    _run_db(work)


def _set_active(slug: str, active: bool) -> None:
    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session, session.begin():
            competitor = await queries.get_competitor(session, slug)
            if competitor is None:
                err.print(f"[red]Unknown competitor {slug!r}[/red]")
                raise typer.Exit(code=2)
            competitor.active = active
        console.print(f"{slug}: {'active' if active else 'inactive'}")

    _run_db(work)


@competitors_cli.command("activate")
def activate(slug: str) -> None:
    """Resume monitoring a competitor."""
    _set_active(slug, True)


@competitors_cli.command("deactivate")
def deactivate(slug: str) -> None:
    """Stop monitoring a competitor (its history is kept)."""
    _set_active(slug, False)


# ── scanning ─────────────────────────────────────────────────────────────────


@cli.command()
def scan(
    slug: str | None = typer.Argument(None, help="Competitor slug (omit with --all)."),
    scan_all: bool = typer.Option(False, "--all", help="Scan every active competitor."),
    since: str | None = typer.Option(None, help="Window start: 7d, 24h, 2w or an ISO date."),
    limit: int | None = typer.Option(None, min=1, max=500, help="Max new/changed pages to fetch."),
    dry_run: bool = typer.Option(False, help="Scan without saving anything (Phase 1 behavior)."),
    json_output: bool = typer.Option(False, "--json", help="Print results as JSON."),
    include_text: bool = typer.Option(False, help="Include extracted main text (JSON output)."),
) -> None:
    """Scan competitor websites (robots-compliant, incremental, no LLM) and record history."""
    if bool(slug) == scan_all:
        raise typer.BadParameter("give a competitor slug or --all (not both)")
    since_at = _since(since)
    settings = get_settings()

    async def work(engine: AsyncEngine, sessions: SessionFactory) -> list[ScanOutcome]:
        async with sessions() as session:
            views = await queries.list_competitors(session)
            slugs = [v.slug for v in views] if scan_all else [str(slug)]
            if dry_run:
                competitor = await queries.get_competitor(session, slugs[0])
                if competitor is None:
                    raise ScanError(f"Unknown competitor {slugs[0]!r}")
                config = competitor.to_config()
        async with PoliteFetcher(settings) as fetcher:
            if dry_run:
                result = await MonitoringService(fetcher, settings).scan(
                    config, since=since_at, limit=limit, include_text=include_text
                )
                _print_result(result, json_output)
                return []
            service = ScanService(engine, sessions, fetcher, settings)
            outcomes = []
            for name in slugs:
                outcome = await service.run(
                    name,
                    trigger=RunTrigger.CLI,
                    since=since_at,
                    limit=limit,
                    include_text=include_text,
                )
                outcomes.append(outcome)
                _print_outcome(name, outcome, json_output)
            return outcomes

    try:
        outcomes = _run_db(work)
    except ScanError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    if any(o.status == "failed" for o in outcomes):
        raise typer.Exit(code=1)


def _print_outcome(slug: str, outcome: ScanOutcome, json_output: bool) -> None:
    if json_output:
        payload = {
            "run_id": outcome.run_id,
            "status": outcome.status.value,
            "error": outcome.error,
            "changes": outcome.summary.as_dict() if outcome.summary else None,
            "result": outcome.result.model_dump(mode="json") if outcome.result else None,
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return
    if outcome.result is None:
        err.print(f"[red]{slug}: run {outcome.run_id} failed:[/red] {outcome.error}")
        return
    _render_result(outcome.result)
    s = outcome.summary
    if s is not None:
        label = "baseline recorded" if s.baseline else "changes"
        console.print(
            f"[bold]run {outcome.run_id}[/bold] {outcome.status.value} · {label}: "
            f"new={s.new_urls} first_captures={s.first_captures} updated={s.updated} "
            f"minor={s.minor_updates} pricing_changed={s.pricing_changed} removed={s.removed} "
            f"restored={s.restored} unchanged={s.unchanged} not_modified={s.not_modified}"
        )


def _print_result(result: ScanResult, json_output: bool) -> None:
    if json_output:
        sys.stdout.write(result.model_dump_json(indent=2) + "\n")
    else:
        _render_result(result)


def _render_result(result: ScanResult) -> None:
    color = {"ok": "green", "partial": "yellow", "failed": "red"}[result.status]
    robots = result.robots
    console.print(
        f"[bold]{result.competitor}[/bold]  status=[{color}]{result.status}[/{color}]  "
        f"duration={result.duration_seconds:.1f}s  requests={result.stats.http_requests}  "
        f"robots={robots.status if robots else 'n/a'}"
        + (f" (crawl-delay {robots.crawl_delay}s)" if robots and robots.crawl_delay else "")
    )
    if result.since:
        console.print(f"window: since {result.since:%Y-%m-%d %H:%M} UTC")
    console.print(f"feeds: {', '.join(result.feeds) or 'none found'} · sitemaps read: {len(result.sitemaps)}")  # fmt: skip
    table = Table("Published", "Type", "Title", "URL", "Words")
    for item in result.items:
        table.add_row(_date(item.published_at), item.content_type.value, (item.title or "")[:70], item.final_url, str(item.word_count))  # fmt: skip
    console.print(table)
    s = result.stats
    console.print(
        f"discovered={s.discovered} fetched={s.fetched} known_unchanged={s.known_unchanged} "
        f"revisited={s.revisited} not_modified={s.not_modified} outside_window={s.outside_window} "
        f"over_limit={s.over_limit} robots_disallowed={s.robots_disallowed} errors={s.errors}"
    )
    for issue in result.errors[:10]:
        console.print(f"[yellow]error[/yellow] {issue.reason}: {issue.url} {issue.detail or ''}")


# ── history queries ──────────────────────────────────────────────────────────


@cli.command()
def content(
    slug: str | None = typer.Argument(None, help="Competitor slug (all if omitted)."),
    content_type: Annotated[ContentType | None, typer.Option("--type")] = None,
    published_since: str | None = typer.Option(None, help="Reliably dated since: 7d, ISO date…"),
    new_only: bool = typer.Option(False, help="Exclude the competitor's baseline scan."),
    search: str | None = typer.Option(None, help="Search titles and URLs."),
    limit: int = typer.Option(30, min=1, max=200),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List captured competitor content, newest publication first."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            items = await queries.list_content(
                session,
                competitor=slug,
                content_type=content_type,
                published_since=_since(published_since),
                include_baseline=not new_only,
                search=search,
                limit=limit,
            )
        if json_output:
            sys.stdout.write("[" + ",".join(i.model_dump_json() for i in items) + "]\n")
            return
        table = Table("ID", "Published", "Source", "First seen", "Type", "Status", "Title", "URL")
        for i in items:
            table.add_row(
                str(i.id), _date(i.published_at), i.published_at_source or "—",
                _date(i.first_seen_at), i.content_type.value, i.status.value,
                (i.title or "")[:60], i.url,
            )  # fmt: skip
        console.print(table)

    _run_db(work)


@cli.command()
def changes(
    slug: str | None = typer.Argument(None, help="Competitor slug (all if omitted)."),
    since: str | None = typer.Option("30d", help="Window start: 7d, ISO date…"),
    change_type: Annotated[ChangeType | None, typer.Option("--type")] = None,
    include_minor: bool = typer.Option(False, help="Include minor edits."),
    limit: int = typer.Option(50, min=1, max=200),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List detected changes (new, updated, pricing_changed, removed, restored)."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            events = await queries.list_changes(
                session,
                competitor=slug,
                change_type=change_type,
                since=_since(since),
                include_minor=include_minor,
                limit=limit,
            )
        if json_output:
            sys.stdout.write("[" + ",".join(e.model_dump_json() for e in events) + "]\n")
            return
        table = Table("Detected", "Competitor", "Change", "Type", "Title", "Details")
        for e in events:
            detail = ", ".join(f"{k}={v}" for k, v in e.details.items() if k in ("words_added", "words_removed", "added", "removed", "http_status"))  # fmt: skip
            label = e.change_type.value + (" (minor)" if e.is_minor else "")
            table.add_row(f"{e.detected_at:%Y-%m-%d %H:%M}", e.competitor, label, e.content_type.value, (e.title or e.url)[:60], detail)  # fmt: skip
        console.print(table if events else "No changes in this window.")

    _run_db(work)


@cli.command()
def runs(
    slug: str | None = typer.Argument(None, help="Competitor slug (all if omitted)."),
    limit: int = typer.Option(20, min=1, max=200),
) -> None:
    """List recent runs (scans, analyses, reports)."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            views = await queries.list_runs(session, competitor=slug, limit=limit)
        table = Table("Run", "Kind", "Competitor", "Trigger", "Status", "Started", "Result", "Error")  # fmt: skip
        for r in views:
            if r.kind == "scan":
                changes_ = r.summary.get("changes", {})
                result = f"fetched {r.stats.get('fetched', '—')}, new {changes_.get('new_urls', '—')}, updated {changes_.get('updated', '—')}"  # fmt: skip
            elif r.kind == "analysis":
                result = f"analyzed {r.summary.get('analyzed', '—')}, pending {r.summary.get('pending_after', '—')}, {r.stats.get('total_tokens', 0):,} tokens"  # fmt: skip
            else:
                result = f"{r.stats.get('total_tokens', 0):,} tokens" if r.stats else ""
            table.add_row(
                str(r.id), r.kind, r.competitor or "—", r.trigger.value, r.status.value,
                f"{r.started_at:%Y-%m-%d %H:%M}" if r.started_at else "—", result, (r.error or "")[:50],
            )  # fmt: skip
        console.print(table)

    _run_db(work)


@cli.command()
def run(run_id: int) -> None:
    """Show one run with its statistics and events."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            view = await queries.get_run(session, run_id)
        if view is None:
            err.print(f"[red]Unknown run {run_id}[/red]")
            raise typer.Exit(code=2)
        sys.stdout.write(view.model_dump_json(indent=2) + "\n")

    _run_db(work)


@cli.command()
def activity(slug: str, weeks: int = typer.Option(8, min=1, max=104)) -> None:
    """Weekly publication and change counts for a competitor."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            competitor = await queries.get_competitor(session, slug)
            if competitor is None:
                err.print(f"[red]Unknown competitor {slug!r}[/red]")
                raise typer.Exit(code=2)
            report = await queries.activity(session, competitor, weeks=weeks, now=utcnow())
        table = Table("Week of", "Published", "By type", "Newly discovered", "Updated", "Pricing", "Removed")  # fmt: skip
        for w in report.weeks:
            by_type = ", ".join(f"{k}:{v}" for k, v in w.published_by_type.items())
            table.add_row(_date(w.week_start), str(w.published), by_type, str(w.newly_discovered), str(w.updated), str(w.pricing_changed), str(w.removed))  # fmt: skip
        console.print(table)
        console.print(
            f"published {report.published_total} in {weeks} weeks "
            f"({report.published_per_week}/week, reliable dates only); "
            f"{report.undated_items} captured items have no reliable publication date; "
            f"first scan: {_date(report.first_scan_at)}"
        )

    _run_db(work)


# ── AI analysis (Phase 3) ────────────────────────────────────────────────────


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _shares(shares: list[Share], limit: int = 6) -> str:
    return ", ".join(f"{s.value} {_pct(s.share)}" for s in shares[:limit]) or "—"


def _llm_error(exc: Exception) -> typer.Exit:
    err.print(f"[red]{exc}[/red]")
    if isinstance(exc, LLMConfigurationError):
        err.print("Set GEMINI_API_KEY in .env (see .env.example). `analyze --dry-run` works without it.")  # fmt: skip
    return typer.Exit(code=2)


@cli.command()
def analyze(
    slug: str | None = typer.Argument(None, help="Competitor slug (omit with --all)."),
    analyze_all: bool = typer.Option(False, "--all", help="Analyze every active competitor."),
    limit: int | None = typer.Option(None, min=1, max=500, help="Max pages this run."),
    reanalyze: bool = typer.Option(False, help="Redo already-analyzed pages (oldest first)."),
    change_summaries: bool = typer.Option(True, help="Summarize significant page changes."),
    profile: bool = typer.Option(True, help="Refresh the competitor profile."),
    force_profile: bool = typer.Option(False, help="Regenerate the profile even if unchanged."),
    dry_run: bool = typer.Option(
        False, help="Show what would be sent (no Gemini call, no writes)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print results as JSON."),
) -> None:
    """Analyze new and changed competitor pages with Gemini (topics, format, audience,
    intent, angle, themes, entities), summarize changes, and refresh profiles."""
    if bool(slug) == analyze_all:
        raise typer.BadParameter("give a competitor slug or --all (not both)")
    settings = get_settings()
    options = AnalysisOptions(
        limit=limit,
        reanalyze=reanalyze,
        change_summaries=change_summaries,
        profile=profile,
        force_profile=force_profile,
    )

    async def work(engine: AsyncEngine, sessions: SessionFactory) -> list[AnalysisOutcome]:
        async with sessions() as session:
            views = await queries.list_competitors(session)
        slugs = [v.slug for v in views] if analyze_all else [str(slug)]
        llm = LazyLLM(settings)
        service = AnalysisService(engine, sessions, llm, settings)
        outcomes = []
        try:
            for name in slugs:
                if dry_run:
                    plan = await service.plan(name, options)
                    if json_output:
                        sys.stdout.write(plan.model_dump_json(indent=2) + "\n")
                        continue
                    table = Table("ID", "Type", "Words", "Action", "Chars sent", "Title")
                    for item in plan.items:
                        chars = f"{item.digest_chars:,}" + (
                            " (condensed)" if item.truncated else ""
                        )
                        table.add_row(str(item.content_item_id), item.content_type.value, str(item.word_count), item.action, chars, (item.title or item.url)[:60])  # fmt: skip
                    console.print(table)
                    c = plan.coverage
                    console.print(
                        f"[bold]{name}[/bold] dry run · model {plan.model} · {plan.batches} batch(es) · "
                        f"≈{plan.estimated_input_tokens:,} input tokens (output ≤{plan.estimated_max_output_tokens:,}) · "
                        f"captured {c.captured}, analyzed {c.current}, pending {c.pending}, ineligible {c.ineligible}"
                    )  # fmt: skip
                    continue
                outcome = await service.run(name, trigger=RunTrigger.CLI, options=options)
                outcomes.append(outcome)
                _print_analysis(name, outcome, json_output)
        finally:
            await llm.aclose()
        return outcomes

    try:
        outcomes = _run_db(work)
    except (LLMConfigurationError, AnalysisAlreadyRunningError) as exc:
        raise _llm_error(exc) from exc
    except CompetitorNotFoundError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    if any(o.status == RunStatus.FAILED for o in outcomes):
        raise typer.Exit(code=1)


def _print_analysis(slug: str, outcome: AnalysisOutcome, json_output: bool) -> None:
    if json_output:
        payload = {
            "run_id": outcome.run_id,
            "status": outcome.status.value,
            "error": outcome.error,
            "summary": outcome.summary.as_dict() if outcome.summary else None,
            "usage": outcome.usage.as_dict() if outcome.usage else None,
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return
    s, u = outcome.summary, outcome.usage
    color = {"succeeded": "green", "partial": "yellow"}.get(outcome.status.value, "red")
    console.print(f"[bold]{slug}[/bold] run {outcome.run_id} [{color}]{outcome.status.value}[/{color}]")  # fmt: skip
    if s is not None:
        console.print(
            f"analyzed={s.analyzed} carried_forward={s.carried_forward} failed={s.failed} "
            f"pending={s.pending_after} ineligible={s.ineligible} batches={s.batches} "
            f"new_topics={s.topics_created} change_summaries={s.change_summaries} "
            f"profile={s.profile or '—'}{f' v{s.profile_version}' if s.profile_version else ''}"
        )
    if u is not None:
        console.print(f"gemini: {u.calls} call(s), {u.input_tokens:,} input + {u.output_tokens:,} output tokens ({u.total_tokens:,} total)")  # fmt: skip
    if outcome.error:
        err.print(f"[yellow]{outcome.error}[/yellow]")


@cli.command()
def analysis(item_id: int, json_output: bool = typer.Option(False, "--json")) -> None:
    """Show the AI analysis of one content item (latest first)."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            views = await analysis_queries.item_analyses(session, item_id)
        if json_output:
            sys.stdout.write("[" + ",".join(v.model_dump_json() for v in views) + "]\n")
            return
        if not views:
            err.print(f"No analysis for content item {item_id} (unknown, or not analyzed yet).")
            raise typer.Exit(code=2)
        a = views[0]
        console.print(f"[bold]{a.title or a.url}[/bold]\n{a.url}")
        console.print(f"{a.method.value} · {a.analyzer_version} · {a.model or '—'} · {a.created_at:%Y-%m-%d %H:%M} · confidence {a.confidence}{' · input condensed' if a.input_truncated else ''}")  # fmt: skip
        rows = [
            ("Summary", a.summary),
            (
                "Topics",
                "; ".join(
                    f"{t.name}{' (' + t.role.value + ')' if t.role.value != 'subtopic' else ''}"
                    for t in a.topics
                    if t.parent is None
                )
                or "—",
            ),
            ("Subtopics", "; ".join(t.name for t in a.topics if t.parent) or "—"),
            ("Format", a.content_format.value),
            ("Audiences", ", ".join(a.target_audiences) or "—"),
            (
                "Intent / funnel",
                f"{a.intent.value if a.intent else '—'} / {a.funnel_stage.value if a.funnel_stage else '—'}",
            ),
            ("Angle", a.primary_angle or "—"),
            ("Themes", "; ".join(a.key_themes) or "—"),
            ("Keywords", ", ".join(a.keywords) or "—"),
            ("Claims", "; ".join(a.positioning_claims) or "—"),
            ("Entities", ", ".join(f"{e.name} ({e.type.value})" for e in a.entities) or "—"),
        ]
        table = Table(show_header=False)
        for label, value in rows:
            table.add_row(label, value)
        console.print(table)
        if len(views) > 1:
            console.print(f"{len(views) - 1} earlier analysis(es): use --json")

    _run_db(work)


# ── topics ───────────────────────────────────────────────────────────────────


@topics_cli.callback(invoke_without_command=True)
def topics_main(ctx: typer.Context) -> None:
    """List top-level topics when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        list_topics(parent=None, search=None, include_merged=False, limit=50)


@topics_cli.command("list")
def list_topics(
    parent: str | None = typer.Option(None, help="List this topic's subtopics."),
    search: str | None = typer.Option(None, help="Search names and slugs."),
    include_merged: bool = typer.Option(False, help="Include merged topics."),
    limit: int = typer.Option(50, min=1, max=1000),
) -> None:
    """Topics with how many analyzed pages and competitors use them."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            parent_topic = await analysis_queries.get_topic(session, parent) if parent else None
            if parent and parent_topic is None:
                err.print(f"[red]Unknown topic {parent!r}[/red]")
                raise typer.Exit(code=2)
            views = await analysis_queries.list_topics(session, parent=parent_topic, include_merged=include_merged, search=search, limit=limit)  # fmt: skip
        table = Table("Slug", "Name", "Items", "Competitors", "Subtopics", "Origin", "Aliases")
        for v in views:
            table.add_row(v.slug, v.name + ("" if v.status.value == "active" else " (merged)"), str(v.items), str(v.competitors), str(v.subtopics), v.origin.value, ", ".join(v.aliases[:4]))  # fmt: skip
        console.print(table if views else "No topics yet: run `analyze` (or `topics import`).")

    _run_db(work)


@topics_cli.command("show")
def show_topic(slug: str, days: int = typer.Option(30, min=7, max=365)) -> None:
    """One topic across competitors: trend, subtopics, recent pages."""
    settings = get_settings()

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        detail = await IntelligenceService(sessions, settings).topic_detail(slug, window_days=days)
        if detail is None:
            err.print(f"[red]Unknown topic {slug!r}[/red]")
            raise typer.Exit(code=2)
        t = detail.topic
        console.print(f"[bold]{t.name}[/bold] ({t.slug}){' ← merged from ' + detail.merged_from if detail.merged_from else ''}")  # fmt: skip
        if detail.trend:
            tr = detail.trend
            per = ", ".join(f"{k} {v}" for k, v in tr.by_competitor.items())
            console.print(f"{tr.items} items · {tr.competitors} competitor(s) ({per}) · last {days}d: {tr.recent}, previous {days}d: {tr.previous} · {tr.trend.value}")  # fmt: skip
        if t.aliases:
            console.print(f"aliases: {', '.join(t.aliases)}")
        if detail.subtopics:
            console.print("subtopics: " + ", ".join(f"{s.topic.name} ({s.items})" for s in detail.subtopics[:15]))  # fmt: skip
        table = Table("Published", "Competitor", "Format", "Title")
        for a in detail.recent_items[:15]:
            table.add_row(_date(a.published_at), a.competitor, a.content_format.value, (a.title or a.url)[:70])  # fmt: skip
        console.print(table)

    _run_db(work)


@topics_cli.command("import")
def import_topics(
    file: Annotated[Path | None, typer.Option(help="YAML file (default: TOPICS_FILE).")] = None,
) -> None:
    """Seed the taxonomy from a YAML file (optional; existing topics are kept)."""
    settings = get_settings()
    try:
        seeds = load_topic_seeds(file or settings.topics_file)
    except ConfigurationError as exc:
        err.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        summary = await TopicAdminService(sessions, LazyLLM(settings), settings).import_seeds(seeds)
        console.print(f"topics created: {summary.topics_created}, subtopics created: {summary.subtopics_created}, aliases added: {summary.aliases_added}")  # fmt: skip
        for conflict in summary.conflicts:
            err.print(f"[yellow]skipped:[/yellow] {conflict}")

    _run_db(work)


@topics_cli.command("merge")
def merge_topics(source: str, target: str) -> None:
    """Fold topic SOURCE into TARGET (pages, aliases and subtopics move over)."""
    settings = get_settings()

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        admin = TopicAdminService(sessions, LazyLLM(settings), settings)
        summary = await admin.merge(source, target, trigger=RunTrigger.CLI)
        console.print(f"merged {summary.source} → {summary.target}: {summary.links_moved + summary.links_combined} page link(s), {summary.aliases_moved} alias(es), {summary.subtopics_moved + summary.subtopics_merged} subtopic(s)")  # fmt: skip

    try:
        _run_db(work)
    except TopicMergeError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


@topics_cli.command("consolidate")
def consolidate_topics(
    apply: bool = typer.Option(False, help="Apply the proposed merges (default: only show them)."),
) -> None:
    """Ask Gemini which topics are duplicates (e.g. "Agentic AI" vs "AI agents")."""
    settings = get_settings()

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        llm = LazyLLM(settings)
        try:
            result = await TopicAdminService(sessions, llm, settings).consolidate(apply=apply, trigger=RunTrigger.CLI)  # fmt: skip
        finally:
            await llm.aclose()
        if not result.proposals:
            console.print("No duplicate topics found.")
            return
        for p in result.proposals:
            console.print(f"{', '.join(s.name for s in p.sources)} → [bold]{p.target.name}[/bold]  [dim]{p.reason}[/dim]")  # fmt: skip
        console.print("merges applied" if apply else "not applied: re-run with --apply to merge")
        if result.rejected:
            console.print(f"[yellow]{result.rejected} invalid proposal(s) ignored[/yellow]")

    try:
        _run_db(work)
    except LLMConfigurationError as exc:
        raise _llm_error(exc) from exc


# ── intelligence ─────────────────────────────────────────────────────────────


def _trend_table(trends: list[TopicTrend], days: int, *, by_competitor: bool = False) -> Table:
    table = Table("Topic", "Items", "Share", f"Last {days}d", f"Prev {days}d", "Trend", *(["Competitors"] if by_competitor else []))  # fmt: skip
    for t in trends:
        extra = [", ".join(f"{k} {v}" for k, v in t.by_competitor.items())] if by_competitor else []
        table.add_row(t.topic.name, str(t.items), _pct(t.share), str(t.recent), str(t.previous), t.trend.value, *extra)  # fmt: skip
    return table


def _print_shifts(shifts: list[MixShift]) -> None:
    for s in shifts:
        console.print(f"shift: {s.dimension} {s.value} {_pct(s.previous_share)} → {_pct(s.recent_share)} ({s.change:+.1f} pp)")  # fmt: skip


@cli.command()
def trends(
    slug: str | None = typer.Argument(None, help="Competitor slug (all active if omitted)."),
    days: int = typer.Option(30, min=7, max=365, help="Window length in days."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Topic, format and audience trends (deterministic; reliable publication dates only)."""
    settings = get_settings()

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        service = IntelligenceService(sessions, settings)
        if slug:
            report = await service.competitor(slug, window_days=days)
            if json_output:
                sys.stdout.write(report.model_dump_json(indent=2) + "\n")
                return
            console.print(_trend_table(report.topics, days))
            c = report.cadence
            console.print(f"published: {c.recent} in the last {days} days ({c.per_week}/week), {c.previous} before; {c.undated} undated")  # fmt: skip
            console.print(f"formats: {_shares(report.formats)}\naudiences: {_shares(report.audiences)}\nintents: {_shares(report.intents)}")  # fmt: skip
            _print_shifts(report.strategy_shifts)
            basis = report.basis
        else:
            landscape = await service.landscape(window_days=days)
            if json_output:
                sys.stdout.write(landscape.model_dump_json(indent=2) + "\n")
                return
            console.print(_trend_table(landscape.topics[:30], days, by_competitor=True))
            console.print("rising: " + (", ".join(f"{t.topic.name} ({t.previous}→{t.recent})" for t in landscape.rising) or "—"))  # fmt: skip
            for n in landscape.neglected[:10]:
                console.print(f"neglected: {n.topic.name} [{n.reason}] {n.detail}")
            console.print(f"formats: {_shares(landscape.formats)}\naudiences: {_shares(landscape.audiences)}")  # fmt: skip
            _print_shifts(landscape.strategy_shifts)
            basis = landscape.basis
        if basis.insufficient_history:
            console.print(f"[yellow]insufficient history for growth comparisons: {', '.join(basis.insufficient_history)}[/yellow]")  # fmt: skip

    try:
        _run_db(work)
    except CompetitorNotFoundError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


def _claim(label: str, claim: Claim | None) -> None:
    if claim is not None:
        sources = ", ".join(e.url for e in claim.evidence[:3])
        console.print(f"[bold]{label}:[/bold] {claim.text} [dim]({sources})[/dim]")


@cli.command("profile")
def show_profile(slug: str, json_output: bool = typer.Option(False, "--json")) -> None:
    """Show the latest AI competitor profile (evidence-backed)."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            competitor = await queries.get_competitor(session, slug)
            if competitor is None:
                err.print(f"[red]Unknown competitor {slug!r}[/red]")
                raise typer.Exit(code=2)
            row = await analysis_queries.latest_profile_row(session, competitor.id)
        if row is None:
            err.print(f"No profile for {slug!r} yet: run `analyze {slug}`.")
            raise typer.Exit(code=2)
        view = analysis_queries.profile_view(row, slug)
        if json_output:
            sys.stdout.write(view.model_dump_json(indent=2) + "\n")
            return
        p = view.profile
        console.print(f"[bold]{p.name}[/bold] profile v{view.version} · {view.created_at:%Y-%m-%d} · {view.model} · confidence {p.confidence}")  # fmt: skip
        for label, claim in (("Tagline", p.tagline), ("Description", p.description), ("Positioning", p.positioning_statement), ("Pricing model", p.pricing_model), ("Content strategy", p.content_strategy)):  # fmt: skip
            _claim(label, claim)
        for label, claims in (("Audiences", p.target_audiences), ("Value propositions", p.value_propositions), ("Key features", p.key_features), ("Differentiators", p.differentiators), ("Notable changes", p.notable_changes)):  # fmt: skip
            if claims:
                console.print(f"[bold]{label}:[/bold] " + "; ".join(c.text for c in claims))
        for tier in p.pricing_tiers:
            console.print(f"  tier {tier.name}: {tier.price or '—'} {tier.billing_period or ''} {'; '.join(tier.highlights)}")  # fmt: skip
        console.print("focus topics: " + ", ".join(f"{t.topic.name} {_pct(t.share)} ({t.trend.value})" for t in p.focus_topics))  # fmt: skip
        console.print(f"formats: {_shares(p.formats)} · audiences: {_shares(p.audiences)}")
        if p.unsupported_claims_dropped:
            console.print(f"[dim]{p.unsupported_claims_dropped} statement(s) without evidence were dropped[/dim]")  # fmt: skip

    _run_db(work)


@cli.command()
def landscape(
    days: int = typer.Option(30, min=7, max=365, help="Window length in days."),
    refresh: bool = typer.Option(False, help="Generate a new AI briefing first (Gemini)."),
    force: bool = typer.Option(False, help="With --refresh: regenerate even if unchanged."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Cross-competitor AI briefing: patterns, rising and neglected subjects, positioning."""
    settings = get_settings()

    async def work(engine: AsyncEngine, sessions: SessionFactory) -> None:
        if refresh:
            llm = LazyLLM(settings)
            try:
                outcome = await LandscapeService(engine, sessions, llm, settings).run(trigger=RunTrigger.CLI, window_days=days, force=force)  # fmt: skip
            finally:
                await llm.aclose()
            if outcome.status == RunStatus.FAILED:
                err.print(f"[red]run {outcome.run_id} failed:[/red] {outcome.error}")
                raise typer.Exit(code=1)
            if outcome.unchanged:
                console.print("[dim]data unchanged since the last briefing; showing it[/dim]")
        async with sessions() as session:
            row = await analysis_queries.latest_landscape_row(session)
        if row is None:
            err.print("No landscape briefing yet: run `landscape --refresh` (after `analyze`).")
            raise typer.Exit(code=2)
        report = analysis_queries.landscape_view(row)
        if json_output:
            sys.stdout.write(report.model_dump_json(indent=2) + "\n")
            return
        n = report.narrative
        console.print(f"[bold]Landscape briefing[/bold] · {report.created_at:%Y-%m-%d %H:%M} · {report.window_days}-day window · {report.model}")  # fmt: skip
        console.print(n.summary)
        sections = (("Patterns", n.patterns), ("Rising", n.rising_subjects), ("Neglected", n.neglected_subjects), ("Formats", n.format_trends), ("Changes", n.notable_changes))  # fmt: skip
        for title, findings in sections:
            if findings:
                console.print(f"\n[bold]{title}[/bold]")
                for f in findings:
                    console.print(f"• {f.text}")
        if n.positioning:
            console.print("\n[bold]Positioning[/bold]")
            for entry in n.positioning:
                console.print(f"• {entry.competitor}: {entry.positioning}")

    try:
        _run_db(work)
    except (LLMConfigurationError, LandscapeAlreadyRunningError) as exc:
        raise _llm_error(exc) from exc


@cli.command()
def usage(days: int = typer.Option(7, min=1, max=90)) -> None:
    """Gemini calls and tokens per day, purpose and model (UTC)."""
    settings = get_settings()

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        now = utcnow()
        async with sessions() as session:
            report = await analysis_queries.llm_usage(session, since=usage_window_start(now, days), today=utc_day_start(now), days=days, daily_budget=settings.llm_daily_token_budget)  # fmt: skip
        table = Table("Day", "Purpose", "Model", "Calls", "Failed", "Input", "Output", "Total")
        for r in report.rows:
            table.add_row(_date(r.day), r.purpose.value, r.model, str(r.calls), str(r.failed), f"{r.input_tokens:,}", f"{r.output_tokens:,}", f"{r.total_tokens:,}")  # fmt: skip
        console.print(table)
        budget = f"{report.daily_budget:,}" if report.daily_budget else "unlimited"
        console.print(f"today: {report.today_tokens:,} tokens (daily budget {budget}); last {days} day(s): {report.total_tokens:,}")  # fmt: skip

    _run_db(work)


# ── company profile (Phase 4) ────────────────────────────────────────────────


@company_cli.command("import")
def import_company(
    file: Annotated[Path | None, typer.Option(help="YAML file (default: COMPANY_FILE).")] = None,
) -> None:
    """Store your company profile as a new version (no-op if unchanged)."""
    settings = get_settings()
    try:
        profile = load_company_profile(file or settings.company_file)
    except ConfigurationError as exc:
        err.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session, session.begin():
            row, created = await save_company_profile(session, profile, source="file", now=utcnow())
        state = "created" if created else "unchanged"
        console.print(f"company profile v{row.version} {state}: {profile.name}")
        if created and row.version > 1:
            console.print("Run `opportunities generate` to re-score opportunities against it.")

    _run_db(work)


@company_cli.command("show")
def show_company(json_output: bool = typer.Option(False, "--json")) -> None:
    """Show the company profile in use."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            row = await latest_company_profile(session)
        if row is None:
            err.print("No company profile yet: copy config/company.example.yaml to config/company.yaml, edit it, then run `company import`.")  # fmt: skip
            raise typer.Exit(code=2)
        view = company_view(row)
        if json_output:
            sys.stdout.write(view.model_dump_json(indent=2) + "\n")
            return
        p = view.profile
        console.print(f"[bold]{p.name}[/bold] · profile v{view.version} · {view.created_at:%Y-%m-%d %H:%M} ({view.source})")  # fmt: skip
        console.print(p.description)
        for label, values in (("products", [x.name for x in p.products]), ("audiences", p.target_audiences), ("core topics", p.core_topics), ("adjacent topics", p.adjacent_topics), ("excluded topics", p.excluded_topics), ("preferred formats", [f.value for f in p.preferred_formats])):  # fmt: skip
            if values:
                console.print(f"{label}: {', '.join(values)}")

    _run_db(work)


@company_cli.command("versions")
def company_versions() -> None:
    """List company profile versions, newest first."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            rows = await list_company_profiles(session)
        table = Table("Version", "Created", "Source", "Name", "Core topics", "Fingerprint")
        for row in rows:
            p = row.to_profile()
            table.add_row(str(row.version), f"{row.created_at:%Y-%m-%d %H:%M}", row.source, p.name, ", ".join(p.core_topics)[:60], row.fingerprint[:12])  # fmt: skip
        console.print(table if rows else "No company profile yet.")

    _run_db(work)


# ── opportunities (Phase 4) ──────────────────────────────────────────────────


def _print_generation(outcome: GenerationOutcome, json_output: bool) -> None:
    s, u = outcome.summary, outcome.usage
    if json_output:
        payload = {"run_id": outcome.run_id, "status": outcome.status.value, "error": outcome.error, "summary": s.as_dict() if s else None, "usage": u.as_dict() if u else None}  # fmt: skip
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return
    color = {"succeeded": "green", "partial": "yellow"}.get(outcome.status.value, "red")
    console.print(f"run {outcome.run_id} [{color}]{outcome.status.value}[/{color}]")
    if s is not None:
        rejected = ", ".join(f"{k} {v}" for k, v in sorted(s.rejected.items())) or "none"
        console.print(f"company profile v{s.company_profile_version} · {s.candidates} candidates → {s.qualified} opportunities (rejected: {rejected})")  # fmt: skip
        console.print(f"created={s.created} rescored={s.rescored} unchanged={s.unchanged} reopened={s.reopened} expired={s.expired}")  # fmt: skip
        console.print(f"interpretation: new={s.interpreted} reused={s.interpretations_reused} failed={s.interpretations_failed} skipped={s.interpretations_skipped} unverified_sentences_removed={s.unverified_sentences_removed}")  # fmt: skip
    if u is not None and u.calls:
        console.print(f"gemini: {u.calls} call(s), {u.total_tokens:,} tokens")
    if outcome.error:
        err.print(f"[yellow]{outcome.error}[/yellow]")


@opportunities_cli.callback(invoke_without_command=True)
def opportunities_main(ctx: typer.Context) -> None:
    """List opportunities when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        list_opportunities(status=None, min_score=None, topic=None, competitor=None, origin=None, limit=25, json_output=False)  # fmt: skip


@opportunities_cli.command("generate")
def generate_opportunities(
    window_days: int | None = typer.Option(
        None, min=7, max=365, help="Override the scoring window."
    ),
    interpret: bool = typer.Option(True, help="Let Gemini interpret the top candidates."),
    force: bool = typer.Option(False, help="Re-assess and re-interpret even if unchanged."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Score content opportunities (deterministic), then interpret the top ones (Gemini)."""
    settings = get_settings()

    async def work(engine: AsyncEngine, sessions: SessionFactory) -> GenerationOutcome:
        llm = LazyLLM(settings)
        try:
            service = OpportunityService(engine, sessions, llm, settings)
            return await service.run(trigger=RunTrigger.CLI, options=GenerationOptions(window_days=window_days, interpret=interpret, force=force))  # fmt: skip
        finally:
            await llm.aclose()

    try:
        outcome = _run_db(work)
    except (NoCompanyProfileError, OpportunityRunAlreadyActiveError, ConfigurationError) as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    _print_generation(outcome, json_output)
    if not json_output and outcome.status != RunStatus.FAILED:
        list_opportunities(status=None, min_score=None, topic=None, competitor=None, origin=None, limit=10, json_output=False)  # fmt: skip
    if outcome.status == RunStatus.FAILED:
        raise typer.Exit(code=1)


def _print_opportunities(rows: Sequence[OpportunitySummary], *, empty: str) -> None:
    table = Table("#", "ID", "Score", "Topic", "Working title", "Gap", "Format", "Audience", "Status")  # fmt: skip
    for r in rows:
        fmt = r.recommended_format.value if r.recommended_format else "—"
        status_text = r.status.value + (" (stale)" if r.stale else "")
        gap = r.primary_gap.value if r.primary_gap else ("editorial" if r.origin is OpportunityOrigin.EDITORIAL else "—")  # fmt: skip
        table.add_row(str(r.rank), str(r.id), f"{r.score:.0f}", escape(r.topic_label[:32]), escape(r.title[:52]), gap, fmt, escape((r.target_audience or "—")[:24]), status_text)  # fmt: skip
    console.print(table if rows else empty)


@opportunities_cli.command("list")
def list_opportunities(
    status: Annotated[
        list[OpportunityStatus] | None,
        typer.Option("--status", help="Repeatable. Default: new, reviewed, approved."),
    ] = None,
    min_score: float | None = typer.Option(None, min=0, max=100),
    topic: str | None = typer.Option(None, help="Topic slug or part of its name."),
    competitor: str | None = typer.Option(None, help="Competitor slug in the evidence."),
    origin: Annotated[
        OpportunityOrigin | None,
        typer.Option(help="competitors or editorial (default: both)."),
    ] = None,
    limit: int = typer.Option(25, min=1, max=200),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Opportunities ranked by score (editorial topics show "editorial" as their gap)."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            rows = await opportunity_queries.list_opportunities(
                session,
                now=utcnow(),
                statuses=tuple(status) if status else opportunity_queries.ACTIONABLE_STATUSES,
                min_score=min_score,
                topic=topic,
                competitor=competitor,
                origin=origin,
                limit=limit,
            )
        if json_output:
            sys.stdout.write("[" + ",".join(r.model_dump_json() for r in rows) + "]\n")
            return
        _print_opportunities(rows, empty="No opportunities: run `opportunities generate` (after `analyze`) or `editorial propose`.")  # fmt: skip

    _run_db(work)


@opportunities_cli.command("show")
def show_opportunity(opportunity_id: int, json_output: bool = typer.Option(False, "--json")) -> None:  # fmt: skip
    """One opportunity: score breakdown, why, gaps, Gemini's interpretation, timeline."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            detail = await opportunity_queries.get_opportunity(
                session, opportunity_id, now=utcnow()
            )
        if detail is None:
            err.print(f"[red]Unknown opportunity {opportunity_id}[/red]")
            raise typer.Exit(code=2)
        if json_output:
            sys.stdout.write(detail.model_dump_json(indent=2) + "\n")
            return
        a = detail.assessment
        console.print(f"[bold]#{detail.id} {detail.title}[/bold]")
        console.print(f"topic: {detail.topic_label} · status: {detail.status.value} · score [bold]{detail.score:.1f}[/bold]/100")  # fmt: skip
        if a is None:
            return
        table = Table("Dimension", "Points", "Detail")
        for c in a.breakdown:
            points = f"{c.points:+.1f}" if c.dimension == "saturation" else f"{c.points:.1f}/{c.max_points:.0f}"  # fmt: skip
            table.add_row(c.dimension.replace("_", " "), points, c.detail[:110])
        console.print(table)
        console.print("[bold]Why[/bold]")
        for reason in a.suggestion.reasons:
            console.print(f"• {reason}")
        gaps = [g for g in sorted(a.gaps, key=lambda g: -g.score) if g.score > 0]
        if gaps:
            console.print("[bold]Gaps[/bold] " + "; ".join(f"{g.type.value} {g.score:.2f}" for g in gaps))  # fmt: skip
        s = a.suggestion
        console.print(f"[bold]Suggested[/bold] format: {s.format.value if s.format else 'open'} · audience: {s.audience or 'open'} · intent: {s.intent.value if s.intent else 'open'}")  # fmt: skip
        i = a.interpretation
        if i is not None:
            console.print(f"[bold]Gemini[/bold] ({a.interpretation_status.value}, {a.interpretation_model}, confidence {i.confidence:.2f})")  # fmt: skip
            for label, text in (("Angle", i.recommended_angle), ("Why now", i.why_now), ("Audience", i.target_audience), ("Format / intent", f"{i.recommended_format.value} / {i.search_intent.value if i.search_intent else '—'}"), ("Differentiation", i.differentiation_strategy), ("Rationale", i.strategic_rationale)):  # fmt: skip
                console.print(f"  {label}: {text}")
        else:
            console.print(f"[dim]interpretation: {a.interpretation_status.value}{' — ' + a.interpretation_error if a.interpretation_error else ''}[/dim]")  # fmt: skip
        if a.change is not None:
            console.print(f"[bold]Changed[/bold] {a.change.previous_score} → {a.change.score}: " + "; ".join(a.change.reasons))  # fmt: skip
        console.print(f"[dim]company profile v{a.company_profile_version} · assessment {a.id} · evidence: `opportunities evidence {detail.id}` · history: `opportunities history {detail.id}`[/dim]")  # fmt: skip

    _run_db(work)


@opportunities_cli.command("evidence")
def opportunity_evidence(opportunity_id: int, json_output: bool = typer.Option(False, "--json")) -> None:  # fmt: skip
    """The evidence behind an opportunity's current score."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            rows = await opportunity_queries.evidence(session, opportunity_id)
        if rows is None:
            err.print(f"[red]Unknown opportunity {opportunity_id}[/red]")
            raise typer.Exit(code=2)
        if json_output:
            sys.stdout.write("[" + ",".join(r.model_dump_json() for r in rows) + "]\n")
            return
        table = Table("ID", "Kind", "Ref", "Competitor", "Evidence")
        for r in rows:
            label = r.label if r.kind.value != "content" else f"{r.label} ({r.data.get('url')})"
            table.add_row(str(r.id), r.kind.value, str(r.ref_id or "—"), r.competitor or "—", label[:120])  # fmt: skip
        console.print(table)

    _run_db(work)


@opportunities_cli.command("history")
def opportunity_history(opportunity_id: int) -> None:
    """How the opportunity's score changed over time, and why."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            detail = await opportunity_queries.get_opportunity(
                session, opportunity_id, now=utcnow()
            )
            rows = await opportunity_queries.history(session, opportunity_id) if detail else []
        if detail is None:
            err.print(f"[red]Unknown opportunity {opportunity_id}[/red]")
            raise typer.Exit(code=2)
        table = Table("Assessed", "Score", "Δ", "Profile", "Why it changed")
        for r in rows:
            delta = f"{r.change.delta:+.1f}" if r.change else "—"
            table.add_row(f"{r.created_at:%Y-%m-%d %H:%M}", f"{r.score:.1f}", delta, f"v{r.company_profile_version}", "; ".join(r.change.reasons)[:120] if r.change else "first assessment")  # fmt: skip
        console.print(table)
        for e in detail.events:
            transition = f"{e.from_status.value if e.from_status else ''} → {e.to_status.value}" if e.to_status else ""  # fmt: skip
            console.print(f"{e.created_at:%Y-%m-%d %H:%M} {e.kind.value} {transition} {e.note or ''} [dim]({e.actor})[/dim]")  # fmt: skip

    _run_db(work)


def _set_opportunity_status(opportunity_id: int, status: OpportunityStatus, note: str | None) -> None:  # fmt: skip
    settings = get_settings()

    async def work(engine: AsyncEngine, sessions: SessionFactory) -> None:
        service = OpportunityService(engine, sessions, LazyLLM(settings), settings)
        await service.set_status(opportunity_id, status, note=note, actor="cli")
        console.print(f"opportunity {opportunity_id}: {status.value}")

    try:
        _run_db(work)
    except (OpportunityNotFoundError, InvalidStatusTransitionError) as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


@opportunities_cli.command("review")
def review_opportunity(opportunity_id: int, note: str | None = typer.Option(None)) -> None:
    """Mark an opportunity as reviewed."""
    _set_opportunity_status(opportunity_id, OpportunityStatus.REVIEWED, note)


@opportunities_cli.command("approve")
def approve_opportunity(opportunity_id: int, note: str | None = typer.Option(None)) -> None:
    """Approve an opportunity for content generation (Phase 5)."""
    _set_opportunity_status(opportunity_id, OpportunityStatus.APPROVED, note)


@opportunities_cli.command("reject")
def reject_opportunity(opportunity_id: int, note: str | None = typer.Option(None)) -> None:
    """Reject an opportunity (it stays rejected when re-scored)."""
    _set_opportunity_status(opportunity_id, OpportunityStatus.REJECTED, note)


@opportunities_cli.command("use")
def use_opportunity(opportunity_id: int, note: str | None = typer.Option(None)) -> None:
    """Mark an approved opportunity as used (content was created from it)."""
    _set_opportunity_status(opportunity_id, OpportunityStatus.USED, note)


@opportunities_cli.command("expire")
def expire_opportunity(opportunity_id: int, note: str | None = typer.Option(None)) -> None:
    """Retire an opportunity as no longer timely."""
    _set_opportunity_status(opportunity_id, OpportunityStatus.EXPIRED, note)


@opportunities_cli.command("reopen")
def reopen_opportunity(opportunity_id: int, note: str | None = typer.Option(None)) -> None:
    """Reopen an expired opportunity."""
    _set_opportunity_status(opportunity_id, OpportunityStatus.NEW, note)


# ── editorial topics: ideas from your company profile ────────────────────────


def _print_proposal(outcome: ProposalOutcome, json_output: bool) -> None:
    s, u = outcome.summary, outcome.usage
    if json_output:
        payload = {"run_id": outcome.run_id, "status": outcome.status.value, "error": outcome.error, "summary": s.as_dict() if s else None, "ideas": [i.model_dump(mode="json") for i in outcome.ideas], "usage": u.as_dict() if u else None}  # fmt: skip
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return
    color = "green" if outcome.status is RunStatus.SUCCEEDED else "red"
    console.print(f"run {outcome.run_id} [{color}]{outcome.status.value}[/{color}]" + (" [dim](dry run: nothing saved)[/dim]" if s and s.dry_run else ""))  # fmt: skip
    if s is not None:
        rejected = ", ".join(f"{k} {v}" for k, v in sorted(s.rejected.items())) or "none"
        site = f"{s.site_posts} post(s) on your site" if s.site_posts is not None else "your site not read"  # fmt: skip
        console.print(f"company profile v{s.company_profile_version} · asked Gemini for {s.asked} · got {s.proposed} · kept {len([i for i in outcome.ideas if not i.rejected])} of {s.requested} wanted (rejected: {rejected})")  # fmt: skip
        console.print(f"checked against {s.covered} covered title(s) ({site}) · created={s.created} expired={s.expired} unverified_sentences_removed={s.unverified_sentences_removed}")  # fmt: skip
        if s.site_error:
            err.print(f"[yellow]your site's posts weren't checked: {escape(s.site_error)}[/yellow]")
    if outcome.ideas:
        table = Table("ID", "Score", "Topic", "Working title", "Format", "Audience", "Result")
        for idea in sorted(outcome.ideas, key=lambda i: (i.rejected is not None, -i.score)):
            result = f"[dim]{escape(idea.rejected)}[/dim]" if idea.rejected else ("[green]created[/green]" if idea.opportunity_id else "[green]kept[/green]")  # fmt: skip
            table.add_row(str(idea.opportunity_id or "—"), f"{idea.score:.0f}", escape(idea.topic[:32]), escape(idea.title[:52]), idea.recommended_format.value, escape(idea.target_audience[:24]), result)  # fmt: skip
        console.print(table)
    if u is not None and u.calls:
        console.print(f"gemini: {u.calls} call(s), {u.total_tokens:,} tokens")
    if outcome.error:
        err.print(f"[red]{escape(outcome.error)}[/red]")
    elif s is not None and s.created:
        console.print(f"[dim]Next: review them (`opportunities show <id>`), then `opportunities approve <id>` and `articles generate <id>`; scores of {get_settings().pipeline_min_opportunity_score:.0f}+ are eligible for the pipeline.[/dim]")  # fmt: skip


@editorial_cli.callback(invoke_without_command=True)
def editorial_main(ctx: typer.Context) -> None:
    """List editorial topics when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        list_editorial(status=None, limit=25, json_output=False)


@editorial_cli.command("propose")
def propose_editorial(
    count: int | None = typer.Option(
        None, min=1, max=25, help="Ideas to keep (default: EDITORIAL_TOPICS_PER_RUN)."
    ),
    dry_run: bool = typer.Option(
        False, help="Show the ideas without saving them (still one Gemini call)."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Propose article ideas from your company profile (Gemini), checked against what is
    already covered; each kept idea becomes an opportunity (status new)."""
    settings = get_settings()

    async def work(engine: AsyncEngine, sessions: SessionFactory) -> ProposalOutcome:
        llm = LazyLLM(settings)
        try:
            async with PoliteFetcher(settings) as fetcher:
                service = EditorialService(engine, sessions, llm, settings, fetcher=fetcher)
                return await service.propose(trigger=RunTrigger.CLI, count=count, dry_run=dry_run)
        finally:
            await llm.aclose()

    try:
        outcome = _run_db(work)
    except LLMConfigurationError as exc:
        raise _llm_error(exc) from exc
    except (NoCompanyProfileError, EditorialRunAlreadyActiveError, ConfigurationError) as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    _print_proposal(outcome, json_output)
    if outcome.status is RunStatus.FAILED:
        raise typer.Exit(code=1)


@editorial_cli.command("list")
def list_editorial(
    status: Annotated[
        list[OpportunityStatus] | None,
        typer.Option("--status", help="Repeatable. Default: new, reviewed, approved."),
    ] = None,
    limit: int = typer.Option(25, min=1, max=200),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Editorial topics (opportunities proposed from your company profile), by score."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            rows = await opportunity_queries.list_opportunities(session, now=utcnow(), statuses=tuple(status) if status else opportunity_queries.ACTIONABLE_STATUSES, origin=OpportunityOrigin.EDITORIAL, limit=limit)  # fmt: skip
        if json_output:
            sys.stdout.write("[" + ",".join(r.model_dump_json() for r in rows) + "]\n")
            return
        _print_opportunities(rows, empty="No editorial topics: run `editorial propose`.")

    _run_db(work)


# ── articles (Phase 5: drafts only, never published) ─────────────────────────


def _article_errors() -> tuple[type[Exception], ...]:
    return (
        LLMConfigurationError,
        OpportunityNotFoundError,
        NoCompanyProfileError,  # an import (or a brief) without a company profile
        ArticleNotFoundError,
        OpportunityNotApprovedError,
        ArticleConflictError,
        ArticleRunActiveError,
        ArticleBudgetExhaustedError,
    )


def _print_article_run(result: ArticleRequestResult, outcome: ArticleOutcome | None, json_output: bool) -> None:  # fmt: skip
    if json_output:
        payload = {
            "article_id": result.article_id,
            "created": result.created,
            "message": result.message,
            "run_id": outcome.run_id if outcome else result.run_id,
            "run_status": outcome.run_status.value if outcome else None,
            "article_status": outcome.status.value if outcome else None,
            "steps": outcome.steps if outcome else {},
            "usage": outcome.usage.as_dict() if outcome and outcome.usage else None,
            "error": outcome.error if outcome else None,
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return
    if result.message:
        console.print(f"article {result.article_id}: {escape(result.message)}")
    if outcome is None:
        return
    color = {"succeeded": "green", "partial": "yellow"}.get(outcome.run_status.value, "red")
    steps = " · ".join(f"{step} {state}" for step, state in outcome.steps.items())
    console.print(f"article {outcome.article_id} run {outcome.run_id} [{color}]{outcome.run_status.value}[/{color}] → article {outcome.status.value}")  # fmt: skip
    if steps:
        console.print(f"steps: {steps}")
    if outcome.usage is not None and outcome.usage.calls:
        console.print(f"gemini: {outcome.usage.calls} call(s), {outcome.usage.total_tokens:,} tokens")  # fmt: skip
    if outcome.error:
        err.print(f"[yellow]{escape(outcome.error)}[/yellow]")


@articles_cli.callback(invoke_without_command=True)
def articles_main(ctx: typer.Context) -> None:
    """List articles when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        list_articles_cmd(status=None, opportunity=None, since=None, limit=25, json_output=False)


@articles_cli.command("brief")
def article_brief_cmd(opportunity_id: int, json_output: bool = typer.Option(False, "--json")) -> None:  # fmt: skip
    """Preview the deterministic brief an article for this opportunity would get (no
    Gemini, nothing stored)."""
    settings = get_settings()

    async def work(engine: AsyncEngine, sessions: SessionFactory) -> ArticleBrief:
        return await ArticleService(engine, sessions, LazyLLM(settings), settings).preview_brief(opportunity_id)  # fmt: skip

    try:
        brief = _run_db(work)
    except _article_errors() as exc:
        err.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    if json_output:
        sys.stdout.write(brief.model_dump_json(indent=2) + "\n")
        return
    console.print(f"[bold]{escape(brief.working_title)}[/bold]")
    console.print(f"opportunity #{brief.opportunity_id} (score {brief.opportunity_score}) · assessment {brief.assessment_id} · company profile v{brief.company_profile_version}")  # fmt: skip
    for label, value in (("Topic", brief.topic), ("Audience", brief.target_audience), ("Intent", brief.search_intent.value), ("Content type", brief.content_type.value), ("Angle", brief.primary_angle), ("Outcome", brief.desired_outcome), ("Why now", brief.why_now or "—"), ("Differentiation", brief.differentiation_strategy)):  # fmt: skip
        console.print(f"[bold]{label}:[/bold] {escape(value)}")
    for label, values in (("Key points", brief.key_points), ("Competitor weaknesses", brief.competitor_weaknesses), ("Avoid", brief.things_to_avoid)):  # fmt: skip
        console.print(f"[bold]{label}[/bold]")
        for value in values:
            console.print(f"  • {escape(value)}")
    console.print(f"[bold]Evidence[/bold] {len(brief.evidence)} competitor page(s) (context, not facts)")  # fmt: skip
    for e in brief.evidence:
        console.print(f"  #{e.evidence_id} {e.competitor or '?'} · {escape(e.title or '')} · {e.url or ''}")  # fmt: skip
    console.print("[dim]provenance: " + "; ".join(f"{k} ← {v}" for k, v in brief.provenance.items()) + "[/dim]")  # fmt: skip


@articles_cli.command("generate")
def generate_article(
    opportunity_id: int,
    regenerate: bool = typer.Option(
        False, help="Start a new attempt after a failed or cancelled article."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Research, outline, draft and edit an article for an approved opportunity (runs
    now). An opportunity has one live article: this returns it if it exists."""
    settings = get_settings()

    async def work(engine: AsyncEngine, sessions: SessionFactory) -> tuple[ArticleRequestResult, ArticleOutcome | None]:  # fmt: skip
        llm = LazyLLM(settings)
        try:
            return await ArticleService(engine, sessions, llm, settings).generate(opportunity_id, trigger=RunTrigger.CLI, regenerate=regenerate)  # fmt: skip
        finally:
            await llm.aclose()

    try:
        result, outcome = _run_db(work)
    except _article_errors() as exc:
        raise _llm_error(exc) from exc
    _print_article_run(result, outcome, json_output)
    if not json_output:
        console.print(f"[dim]details: `articles show {result.article_id}`[/dim]")
    if outcome is not None and outcome.run_status == RunStatus.FAILED:
        raise typer.Exit(code=1)


@articles_cli.command("import")
def import_article_cmd(
    file: Annotated[
        Path,
        typer.Argument(
            help="A Markdown file with YAML frontmatter (see the README, 'Importing an article you wrote')."
        ),
    ],
    opportunity: int | None = typer.Option(
        None,
        "--opportunity",
        help="Attach it to this opportunity instead of creating one of its own.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Import an article you wrote yourself: it becomes a ready article with an authored
    quality report (length, citations, structure and the site's MDX contract checked; no
    Gemini fact-check, originality check or score) that `articles approve` and
    `articles publish` handle like any other. Nothing is published here."""
    settings = get_settings()

    async def work(engine: AsyncEngine, sessions: SessionFactory) -> ImportOutcome:
        return await ArticleImportService(engine, sessions, settings).import_path(file, opportunity_id=opportunity)  # fmt: skip

    try:
        outcome = _run_db(work)
    except ArticleFileError as exc:
        err.print(f"[red]{escape(str(file))} isn't a publishable article:[/red]")
        for problem in exc.problems:
            err.print(f"  • {escape(problem)}")
        raise typer.Exit(code=2) from exc
    except _article_errors() as exc:
        err.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    if json_output:
        sys.stdout.write(json.dumps(outcome.as_dict(), indent=2) + "\n")
        return
    if not outcome.created:
        console.print(f"[yellow]nothing written[/yellow]: {escape(outcome.message or '')}")
        console.print(f"[dim]details: `articles show {outcome.article_id}`[/dim]")
        return
    console.print(f"article [bold]{outcome.article_id}[/bold] [green]ready[/green] · opportunity {outcome.opportunity_id} · version {outcome.version_id} · report #{outcome.quality_report_id}")  # fmt: skip
    console.print(f"[bold]{escape(outcome.title)}[/bold] · slug: {outcome.slug} · {outcome.word_count:,} words · {outcome.sources} source(s)")  # fmt: skip
    console.print("[yellow]written by a person[/yellow]: length, citations, structure and the site's MDX contract were checked; it was not fact-checked, originality-checked or scored by the agent")  # fmt: skip
    console.print(f"next: `articles show {outcome.article_id}` · `articles approve {outcome.article_id}` · `articles publish {outcome.article_id}`")  # fmt: skip


@articles_cli.command("resume")
def resume_article(article_id: int, json_output: bool = typer.Option(False, "--json")) -> None:
    """Continue a failed or interrupted article from its first unfinished step (earlier
    steps are reused); on a completed article, redo only steps whose prompt or settings
    changed."""
    settings = get_settings()

    async def work(engine: AsyncEngine, sessions: SessionFactory) -> tuple[ArticleRequestResult, ArticleOutcome | None]:  # fmt: skip
        llm = LazyLLM(settings)
        try:
            return await ArticleService(engine, sessions, llm, settings).resume_now(article_id, trigger=RunTrigger.CLI)  # fmt: skip
        finally:
            await llm.aclose()

    try:
        result, outcome = _run_db(work)
    except _article_errors() as exc:
        raise _llm_error(exc) from exc
    _print_article_run(result, outcome, json_output)
    if outcome is not None and outcome.run_status == RunStatus.FAILED:
        raise typer.Exit(code=1)


@articles_cli.command("cancel")
def cancel_article(article_id: int, note: str | None = typer.Option(None)) -> None:
    """Stop an article for good (regenerate from the opportunity for a new attempt)."""
    settings = get_settings()

    async def work(engine: AsyncEngine, sessions: SessionFactory) -> None:
        await ArticleService(engine, sessions, LazyLLM(settings), settings).cancel(article_id, note=note)  # fmt: skip

    try:
        _run_db(work)
    except _article_errors() as exc:
        err.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    console.print(f"article {article_id}: cancelled")


@articles_cli.command("list")
def list_articles_cmd(
    status: Annotated[
        list[ArticleStatus] | None, typer.Option("--status", help="Repeatable.")
    ] = None,
    opportunity: int | None = typer.Option(None, help="Opportunity id."),
    since: str | None = typer.Option(None, help="Created since: 24h, 7d, 2w or an ISO date."),
    limit: int = typer.Option(25, min=1, max=200),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Articles, newest first."""
    created_since = _since(since)

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            rows = await article_queries.list_articles(session, statuses=status, opportunity_id=opportunity, created_since=created_since, limit=limit)  # fmt: skip
        if json_output:
            sys.stdout.write("[" + ",".join(r.model_dump_json() for r in rows) + "]\n")
            return
        table = Table("ID", "Opp.", "Try", "Status", "Step", "Words", "Score", "Tokens", "Created", "Title")  # fmt: skip
        for r in rows:
            state = r.status.value + (f" ({r.failed_step.value})" if r.failed_step and r.status.value == "failed" else "")  # fmt: skip
            table.add_row(str(r.id), str(r.opportunity_id), str(r.attempt), state, r.current_step.value if r.current_step else "—", str(r.word_count or "—"), f"{r.quality_score:.1f}" if r.quality_score is not None else "—", f"{r.tokens_used:,}", _date(r.created_at), escape(r.title[:60]))  # fmt: skip
        console.print(table if rows else "No articles yet: approve an opportunity, then `articles generate <opportunity-id>`.")  # fmt: skip

    _run_db(work)


def _writer_fallbacks(runs: list[ArticleRunView]) -> list[str]:
    """Every provider handover the article's runs recorded, oldest first."""
    return [str(note) for run in runs for note in run.summary.get("writer_fallbacks") or []]


@articles_cli.command("show")
def show_article(
    article_id: int,
    content: bool = typer.Option(
        True, "--content/--no-content", help="Print the article (Markdown preview)."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """One article: status, progress, steps, brief, issues and the content."""
    settings = get_settings()

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            detail = await article_queries.get_article(session, article_id, token_budget=settings.article_max_tokens, include_markdown=content)  # fmt: skip
            publication = await publishing_queries.current_publication(session, article_id)
        approval = await ApprovalService(sessions, settings).view(article_id) if detail is not None else None  # fmt: skip
        if detail is None:
            err.print(f"[red]Unknown article {article_id}[/red]")
            raise typer.Exit(code=2)
        if json_output:
            sys.stdout.write(detail.model_dump_json(indent=2) + "\n")
            return
        console.print(f"[bold]#{detail.id} {escape(detail.title)}[/bold] · {detail.status.value}")
        console.print(f"opportunity #{detail.opportunity_id} ({detail.opportunity_status}) · attempt {detail.attempt} · assessment {detail.assessment_id} · company profile v{detail.company_profile_version}")  # fmt: skip
        console.print(f"slug: {detail.slug} · {detail.content_type.value} · audience: {escape(detail.target_audience or '—')} · intent: {detail.search_intent.value if detail.search_intent else '—'} · words: {detail.word_count or '—'} · tokens: {detail.tokens_used:,} of {detail.token_budget:,}")  # fmt: skip
        if detail.origin is not ArticleOrigin.IMPORTED:
            console.print(f"written by: {detail.writer.value if detail.writer else '—'}" + "".join(f"\n  [yellow]fell back[/yellow] · {escape(note)}" for note in _writer_fallbacks(detail.runs)))  # fmt: skip
        if detail.origin is ArticleOrigin.IMPORTED:
            console.print("[yellow]written by a person and imported[/yellow] (`articles import`): its length, citations, structure and the site's MDX contract were checked; it was [bold]not[/bold] fact-checked, originality-checked or scored by the agent")  # fmt: skip
        done = {s.value for s in detail.progress.completed_steps}
        console.print("progress: " + " → ".join(f"{s}{' ✓' if s in done else ''}" for s in ("brief", "research", "outline", "draft", "edit")) + f" ({detail.progress.percent}%)")  # fmt: skip
        table = Table("Step", "Status", "Prompt", "Model", "Calls", "Tokens", "Finished", "Error")
        for s in detail.steps:
            table.add_row(s.step.value, s.status.value + ("" if s.current else " (old)"), s.prompt_version or "—", s.model or "—", str(s.llm_calls), f"{s.tokens:,}", f"{s.finished_at:%Y-%m-%d %H:%M}" if s.finished_at else "—", escape((s.error or "")[:60]))  # fmt: skip
        console.print(table)
        console.print(f"[bold]Angle:[/bold] {escape(detail.brief.primary_angle)}")
        console.print(f"sources: {detail.sources} (`articles sources {detail.id}`) · versions: `articles versions {detail.id}`")  # fmt: skip
        if detail.validated_at is not None:
            console.print(f"quality: {detail.quality_score:.1f}/100 · recommended version {detail.recommended_version_id} · {detail.revision_count} revision(s) · validated {detail.validated_at:%Y-%m-%d %H:%M} (`articles quality {detail.id}`)")  # fmt: skip
        elif detail.status.value == "completed":
            console.print(f"[dim]not validated yet: `articles validate {detail.id}`[/dim]")
        if approval is not None and approval.state.value != "not_ready":
            console.print(f"approval: {approval.state.value}" + (f" (#{approval.decision.id} by {escape(approval.decision.approver)})" if approval.decision else f" (`articles approve {detail.id}`)"))  # fmt: skip
        if publication is not None:
            console.print(f"publication: {publication.status.value} · {publication.cms} post {publication.external_id or '—'} ({publication.external_status or '—'})" + (f" · {publication.url}" if publication.url else "") + f" (`articles publication {detail.id}`)")  # fmt: skip
        if detail.issues:
            console.print(f"[bold]Issues[/bold] ({len(detail.issues)}, for review)")
            for issue in detail.issues[:12]:
                console.print(f"  • {issue.kind}: {escape(issue.detail)}" + (f" — “{escape(issue.excerpt[:100])}”" if issue.excerpt else ""))  # fmt: skip
        if detail.error:
            err.print(f"[yellow]{detail.status.value}{f' at {detail.failed_step.value}' if detail.failed_step else ''}: {escape(detail.error)}[/yellow]")  # fmt: skip
            if detail.status.value == "failed":
                phase6 = detail.failed_step is not None and detail.failed_step.value not in ("brief", "research", "outline", "draft", "edit")  # fmt: skip
                err.print(
                    f"resume with `articles {'validate' if phase6 else 'resume'} {detail.id}`"
                )
        if content and detail.markdown:
            kind = detail.content_version.kind.value if detail.content_version else "draft"
            label = {"final": "edited article", "revision": f"revision v{detail.content_version.number if detail.content_version else '?'}"}.get(kind, "draft")  # fmt: skip
            if detail.recommended_version_id is not None:
                label = f"recommended version: {label}"
            console.rule(f"{label} (preview; not published)")
            console.print(detail.markdown, markup=False, highlight=False)

    _run_db(work)


@articles_cli.command("sources")
def article_sources_cmd(
    article_id: int,
    all_runs: bool = typer.Option(False, "--all", help="Include earlier research runs."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Research sources Gemini actually retrieved, with their facts and citation counts."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            rows = await article_queries.get_sources(session, article_id, include_all=all_runs)
            notes = await article_queries.research_notes(session, article_id)
        if rows is None:
            err.print(f"[red]Unknown article {article_id}[/red]")
            raise typer.Exit(code=2)
        if json_output:
            sys.stdout.write("[" + ",".join(r.model_dump_json() for r in rows) + "]\n")
            return
        table = Table("Label", "Type", "Facts", "Cited", "Relevance", "Source", "URL")
        for r in rows:
            label = r.label + ("" if r.current else f" (step {r.step_id})")
            table.add_row(label, r.source_type.value + (" *" if r.attribution_required else ""), str(len(r.facts)), str(r.citations), f"{r.relevance:.2f}", escape((r.title or r.domain)[:50]), r.url)  # fmt: skip
        console.print(table if rows else "No sources (research hasn't run yet).")
        if any(r.attribution_required for r in rows):
            console.print("[dim]* competitor or company source: its facts may only be stated with attribution[/dim]")  # fmt: skip
        for note in notes:  # how the research went: searches, rejected calls, budgets
            console.print(f"[dim]· {escape(note)}[/dim]")

    _run_db(work)


@articles_cli.command("versions")
def article_versions_cmd(
    article_id: int,
    show: int | None = typer.Option(None, "--show", help="Print this version (id)."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Every outline, draft, edited and revised version (none is ever overwritten)."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            if show is not None:
                detail = await article_queries.get_version(session, article_id, show, include_markdown=True)  # fmt: skip
                if detail is None:
                    err.print("[red]Unknown article or version[/red]")
                    raise typer.Exit(code=2)
                if json_output:
                    sys.stdout.write(detail.model_dump_json(indent=2) + "\n")
                    return
                console.print(f"[bold]{detail.kind.value} v{detail.number}[/bold] · {detail.prompt_version} · {detail.model or ('written by a person' if detail.authored else '—')} · {len(detail.citations)} citation(s)")  # fmt: skip
                if detail.reason:
                    console.print(f"  reason: {escape(detail.reason)} · issues addressed: {', '.join(detail.issues_addressed) or '—'}")  # fmt: skip
                for change in detail.changes:
                    console.print(f"  change: {escape(change)}")
                for issue in detail.issue_details:
                    console.print(f"  issue: {issue.kind}: {escape(issue.detail)}")
                console.print(detail.markdown or json.dumps(detail.content, indent=2), markup=False, highlight=False)  # fmt: skip
                return
            rows = await article_queries.list_versions(session, article_id)
        if rows is None:
            err.print(f"[red]Unknown article {article_id}[/red]")
            raise typer.Exit(code=2)
        if json_output:
            sys.stdout.write("[" + ",".join(r.model_dump_json() for r in rows) + "]\n")
            return
        table = Table("ID", "Kind", "No.", "Parent", "Current", "Words", "Issues", "Prompt", "Model", "Created", "Title")  # fmt: skip
        for r in rows:
            table.add_row(str(r.id), r.kind.value, str(r.number), str(r.parent_version_id or "—"), "yes" if r.current else "", str(r.word_count or "—"), str(r.issues), r.prompt_version or "—", r.model or ("by hand" if r.authored else "—"), f"{r.created_at:%Y-%m-%d %H:%M}", escape(r.title[:50]))  # fmt: skip
        console.print(table)

    _run_db(work)


@articles_cli.command("steps")
def article_steps_cmd(article_id: int, json_output: bool = typer.Option(False, "--json")) -> None:  # fmt: skip
    """The checkpoint log: every step execution, with fingerprint, prompt, model and tokens."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            rows = await article_queries.list_steps(session, article_id)
        if rows is None:
            err.print(f"[red]Unknown article {article_id}[/red]")
            raise typer.Exit(code=2)
        if json_output:
            sys.stdout.write("[" + ",".join(r.model_dump_json() for r in rows) + "]\n")
            return
        table = Table("ID", "Run", "Step", "Status", "Current", "Fingerprint", "Prompt", "Calls", "Tokens", "Error")  # fmt: skip
        for r in rows:
            table.add_row(str(r.id), str(r.run_id or "—"), r.step.value, r.status.value, "yes" if r.current else "", r.fingerprint[:12], r.prompt_version or "—", str(r.llm_calls), f"{r.tokens:,}", escape((r.error or "")[:50]))  # fmt: skip
        console.print(table)

    _run_db(work)


# ── article validation (Phase 6: prepares articles, never publishes) ─────────


def _print_quality_run(result: ArticleRequestResult, outcome: QualityOutcome, json_output: bool) -> None:  # fmt: skip
    if json_output:
        payload = {
            "article_id": outcome.article_id,
            "run_id": outcome.run_id,
            "run_status": outcome.run_status.value,
            "article_status": outcome.status.value,
            "recommended_version_id": outcome.recommended_version_id,
            "quality_score": outcome.quality_score,
            "revisions": outcome.revisions,
            "steps": outcome.steps,
            "usage": outcome.usage.as_dict() if outcome.usage else None,
            "error": outcome.error,
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return
    color = {"succeeded": "green", "partial": "yellow"}.get(outcome.run_status.value, "red")
    state = {"ready": "green", "needs_review": "yellow"}.get(outcome.status.value, "red")
    console.print(f"article {outcome.article_id} run {outcome.run_id} [{color}]{outcome.run_status.value}[/{color}] → article [{state}]{outcome.status.value}[/{state}]")  # fmt: skip
    if outcome.quality_score is not None:
        console.print(f"quality score {outcome.quality_score:.1f}/100 · recommended version {outcome.recommended_version_id} · {outcome.revisions} revision(s) in total")  # fmt: skip
    ran = sum(1 for s in outcome.steps if s.endswith(":ran"))
    reused = sum(1 for s in outcome.steps if s.endswith(":reused"))
    console.print(f"steps: {ran} ran, {reused} reused" + (f" · {', '.join(s for s in outcome.steps if s.endswith(':failed'))}" if any(s.endswith(":failed") for s in outcome.steps) else ""))  # fmt: skip
    if outcome.usage is not None and outcome.usage.calls:
        console.print(f"gemini: {outcome.usage.calls} call(s), {outcome.usage.total_tokens:,} tokens")  # fmt: skip
    if outcome.error:
        err.print(f"[yellow]{escape(outcome.error)}[/yellow]")


def _run_quality(article_id: int, json_output: bool, *, revise: bool, note: str | None = None) -> None:  # fmt: skip
    settings = get_settings()

    async def work(engine: AsyncEngine, sessions: SessionFactory) -> tuple[ArticleRequestResult, QualityOutcome]:  # fmt: skip
        llm = LazyLLM(settings)
        try:
            service = QualityService(engine, sessions, llm, settings)
            if revise:
                return await service.revise_now(article_id, trigger=RunTrigger.CLI, note=note)
            return await service.validate_now(article_id, trigger=RunTrigger.CLI)
        finally:
            await llm.aclose()

    try:
        result, outcome = _run_db(work)
    except _article_errors() as exc:
        raise _llm_error(exc) from exc
    _print_quality_run(result, outcome, json_output)
    if not json_output:
        console.print(f"[dim]details: `articles quality {article_id}` · `articles fact-check {article_id}` · `articles seo {article_id}`[/dim]")  # fmt: skip
    if outcome.run_status == RunStatus.FAILED:
        raise typer.Exit(code=1)


@articles_cli.command("validate")
def validate_article(article_id: int, json_output: bool = typer.Option(False, "--json")) -> None:
    """Validate a completed article (runs now): fact-check, originality, SEO, metrics, the
    Gemini judge, then at most QUALITY_MAX_REVISIONS revisions while a gate fails. Ends
    ready or needs_review. Unchanged steps are reused; a failed validation resumes where it
    stopped. Nothing is published."""
    _run_quality(article_id, json_output, revise=False)


@articles_cli.command("revise")
def revise_article(
    article_id: int,
    note: str | None = typer.Option(None, help="What to change, besides the open issues."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """One more revision of the recommended version, validated like the others (it's
    recommended only if it scores better)."""
    _run_quality(article_id, json_output, revise=True, note=note)


def _version_error(exc: Exception) -> typer.Exit:
    err.print(f"[red]{escape(str(exc))}[/red]")
    return typer.Exit(code=2)


@articles_cli.command("quality")
def article_quality_cmd(
    article_id: int,
    version: int | None = typer.Option(None, help="A version id (default: the recommended one)."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """The quality score and its breakdown, the gates, the issues, the metrics, the judge's
    rubric and every validated version's score."""
    settings = get_settings()

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            try:
                view = await quality_queries.quality_overview(session, article_id, token_budget=settings.quality_max_tokens, version_id=version)  # fmt: skip
            except quality_queries.UnknownVersionError as exc:
                raise _version_error(exc) from exc
        if view is None:
            err.print(f"[red]Unknown article {article_id}[/red]")
            raise typer.Exit(code=2)
        if json_output:
            sys.stdout.write(view.model_dump_json(indent=2) + "\n")
            return
        console.print(f"[bold]article {view.article_id}[/bold] · {view.status}" + (f" ({view.current_step})" if view.current_step else "") + f" · quality tokens {view.quality_tokens_used:,} of {view.token_budget:,}")  # fmt: skip
        report = view.report
        if report is None:
            console.print(f"Not validated yet: `articles validate {article_id}`.")
            return
        verdict = "[green]passes every gate[/green]" if report.passed else "[yellow]needs review[/yellow]"  # fmt: skip
        score = "written by a person: no agent score" if report.authored else f"score [bold]{report.overall_score:.1f}[/bold]/100"  # fmt: skip
        console.print(f"version {report.version_id} ({report.version_kind} v{report.version_number}) · {score} · {verdict}")  # fmt: skip
        if report.authored:
            console.print("[yellow]authored report[/yellow]: the article was written by a person and imported. The deterministic checks ran; the gates that need Gemini are recorded as not run and nothing here claims a fact-check.")  # fmt: skip
        if report.breakdown:
            table = Table("Component", "Weight", "Value", "Points", "Detail")
            for c in report.breakdown:
                table.add_row(c.dimension, f"{c.max_points:g}", f"{c.value:.2f}", f"{c.points:.1f}", escape(c.detail))  # fmt: skip
            console.print(table)
        gates = Table("Gate", "Result", "Detail")
        for g in report.gates:
            result = {GateStatus.PASSED: "[green]yes[/green]", GateStatus.FAILED: "[red]no[/red]", GateStatus.NOT_RUN: "[yellow]not run[/yellow]"}[g.state]  # fmt: skip
            gates.add_row(g.name, result, escape(g.detail))
        console.print(gates)
        if report.issues:
            console.print(f"[bold]Issues[/bold] ({len(report.issues)}, most serious first)")
            for i in report.issues[:15]:
                console.print(f"  {i.id} p{i.priority} {i.kind}: {escape(i.detail[:160])}")
        if view.judge is not None:
            console.print("[bold]Judge[/bold] " + " · ".join(f"{d.dimension} {d.score}/5" for d in view.judge.dimensions))  # fmt: skip
            console.print(f"  {escape(view.judge.summary)}")
        if view.metrics is not None:
            r = view.metrics.readability
            console.print(f"[bold]Metrics[/bold] words {view.metrics.length.get('words')} · Flesch {r.get('flesch_reading_ease')} (grade {r.get('flesch_kincaid_grade')}) · citation coverage {view.metrics.citations.get('citation_coverage', 0):.0%} · max similarity {view.metrics.originality.get('max_similarity', 0):.0%}")  # fmt: skip
        if len(view.versions) > 1:
            console.print("versions: " + " · ".join(f"{v.version_id} ({v.kind} v{v.number}) {v.score:.1f}{' ✓' if v.passed else ''}{' ← recommended' if v.recommended else ''}" for v in view.versions))  # fmt: skip

    _run_db(work)


@articles_cli.command("fact-check")
def article_fact_check_cmd(
    article_id: int,
    version: int | None = typer.Option(None, help="A version id (default: the recommended one)."),
    verdict: Annotated[
        list[ClaimVerdict] | None, typer.Option("--verdict", help="Repeatable.")
    ] = None,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Every claim check: claim, source, verdict, explanation, evidence."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            try:
                view = await quality_queries.fact_check(session, article_id, version_id=version, verdicts=set(verdict) if verdict else None)  # fmt: skip
            except quality_queries.UnknownVersionError as exc:
                raise _version_error(exc) from exc
        if view is None:
            err.print(f"[red]No fact-check for article {article_id} (yet)[/red]")
            raise typer.Exit(code=2)
        if json_output:
            sys.stdout.write(view.model_dump_json(indent=2) + "\n")
            return
        m = view.metrics
        console.print(f"version {view.version_id} · {m.cited_claims} cited claim(s): {m.supported} supported, {m.partial} partial, {m.unsupported} unsupported, {m.contradicted} contradicted · citation coverage {m.citation_coverage:.0%} · uncited needing a source: {m.uncited_factual} · re-read {m.rereads} page(s)")  # fmt: skip
        if not m.integrity_ok:
            err.print("[yellow]citation integrity: " + escape("; ".join(m.integrity_problems)) + "[/yellow]")  # fmt: skip
        table = Table("Kind", "Sec.", "Source", "Verdict", "Conf.", "Claim", "Explanation")
        colors = {"supported": "green", "partial": "yellow", "not_required": "dim"}
        for c in view.checks:
            color = colors.get(c.verdict.value, "red")
            flags = (" ↻" if c.reread else "") + (" (reused)" if c.reused else "")
            table.add_row(c.kind.value, str(c.section + 1), c.source_label or "—", f"[{color}]{c.verdict.value}[/{color}]{flags}", f"{c.confidence:.2f}" if c.confidence is not None else "—", escape(c.claim[:90]), escape(c.explanation[:90]))  # fmt: skip
        console.print(table)
        for note in view.notes:
            console.print(f"[dim]{escape(note)}[/dim]")

    _run_db(work)


@articles_cli.command("originality")
def article_originality_cmd(
    article_id: int,
    version: int | None = typer.Option(None, help="A version id (default: the recommended one)."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """The similarity signal against stored competitor and company pages (not a plagiarism
    verdict)."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            try:
                view = await quality_queries.originality(session, article_id, version_id=version)
            except quality_queries.UnknownVersionError as exc:
                raise _version_error(exc) from exc
        if view is None:
            err.print(f"[red]No originality check for article {article_id} (yet)[/red]")
            raise typer.Exit(code=2)
        if json_output:
            sys.stdout.write(view.model_dump_json(indent=2) + "\n")
            return
        r = view.report
        console.print(f"version {view.version_id} · {r.passages_checked} passage(s) against {r.documents} page(s) ({r.competitor_documents} competitor, {r.company_documents} company) · {r.ngram_size}-word shingles · {r.common_ngrams_ignored} common n-gram(s) ignored")  # fmt: skip
        console.print(f"score {r.score:.2f} · max similarity {r.max_similarity:.0%} · average {r.avg_similarity:.0%} · overall overlap {r.overall_overlap:.0%}" + (" · [red]severe[/red]" if r.severe else ""))  # fmt: skip
        for f in r.flagged:
            console.print(f"  [yellow]{f.similarity:.0%}[/yellow] section {f.section + 1} ↔ {f.source_label} {f.url}")  # fmt: skip
            console.print(
                f"    overlap ({f.overlap_words} words): “{escape(f.overlap_text[:200])}”"
            )
        if not r.flagged:
            console.print("No passage is similar enough to flag.")

    _run_db(work)


@articles_cli.command("seo")
def article_seo_cmd(
    article_id: int,
    version: int | None = typer.Option(None, help="A version id (default: the recommended one)."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """The SEO package and its checks."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            try:
                view = await quality_queries.seo(session, article_id, version_id=version)
            except quality_queries.UnknownVersionError as exc:
                raise _version_error(exc) from exc
        if view is None:
            err.print(f"[red]No SEO package for article {article_id} (yet)[/red]")
            raise typer.Exit(code=2)
        if json_output:
            sys.stdout.write(view.model_dump_json(indent=2) + "\n")
            return
        p = view.report.package
        console.print(f"[bold]primary keyword:[/bold] {escape(p.primary_keyword)} — {escape(p.primary_keyword_reason)}")  # fmt: skip
        console.print(f"  evidence: {escape('; '.join(p.primary_keyword_evidence))}")
        console.print(f"[bold]secondary:[/bold] {escape(', '.join(p.secondary_keywords) or '—')}")
        console.print(f"[bold]meta title[/bold] ({len(p.meta_title)}): {escape(p.meta_title)}")
        console.print(f"[bold]meta description[/bold] ({len(p.meta_description)}): {escape(p.meta_description)}")  # fmt: skip
        console.print(f"[bold]slug:[/bold] {p.slug} · [bold]category:[/bold] {escape(p.category or '—')} · [bold]tags:[/bold] {escape(', '.join(p.tags) or '—')}")  # fmt: skip
        console.print(f"[bold]headings:[/bold] {p.headings.h1_count} H1, {len(p.headings.h2)} H2, {len(p.headings.h3)} H3" + (f" · {escape('; '.join(p.headings.issues))}" if p.headings.issues else ""))  # fmt: skip
        for label, links in (("internal links", p.internal_links), ("external links", p.external_links)):  # fmt: skip
            console.print(f"[bold]{label}:[/bold] {len(links)}")
            for link in links:
                console.print(f"  “{escape(link.anchor_text)}” → {link.url}")
        console.print(f"[bold]FAQ:[/bold] {len(p.faq)}")
        for item in p.faq:
            console.print(f"  Q: {escape(item.question)}")
        if p.image is not None:
            console.print(f"[bold]image idea:[/bold] {escape(p.image.concept)} (alt: {escape(p.image.alt_text)})")  # fmt: skip
        table = Table("Check", "Passed", "Detail")
        for c in view.report.checks:
            table.add_row(c.name, "[green]yes[/green]" if c.passed else "[red]no[/red]", escape(c.detail))  # fmt: skip
        console.print(table)
        for note in view.report.notes:
            console.print(f"[dim]{escape(note)}[/dim]")

    _run_db(work)


@articles_cli.command("revisions")
def article_revisions_cmd(article_id: int, json_output: bool = typer.Option(False, "--json")) -> None:  # fmt: skip
    """The edited version and every revision, with parent, reason, issues addressed and score."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            rows = await quality_queries.revisions(session, article_id)
        if rows is None:
            err.print(f"[red]Unknown article {article_id}[/red]")
            raise typer.Exit(code=2)
        if json_output:
            sys.stdout.write("[" + ",".join(r.model_dump_json() for r in rows) + "]\n")
            return
        table = Table("ID", "Kind", "No.", "Parent", "Score", "Passed", "Recommended", "Tokens", "Issues addressed", "Reason")  # fmt: skip
        for r in rows:
            table.add_row(str(r.version_id), r.kind, str(r.number), str(r.parent_version_id or "—"), f"{r.score:.1f}" if r.score is not None else "—", ("yes" if r.passed else "no") if r.passed is not None else "—", "yes" if r.recommended else "", f"{r.tokens:,}", ", ".join(r.issues_addressed) or "—", escape((r.reason or "")[:60]))  # fmt: skip
        console.print(table)

    _run_db(work)


# ── approval and publishing (Phase 7: approved, ready versions only; drafts by default) ─


def _publishing_errors() -> tuple[type[Exception], ...]:
    return (*_article_errors(), CMSConfigurationError)


def _print_approval(view: ApprovalView) -> None:
    color = {"approved": "green", "pending": "yellow", "rejected": "red", "invalidated": "yellow"}.get(view.state.value, "red")  # fmt: skip
    console.print(f"[bold]article {view.article_id}[/bold] · {view.article_status} · approval [{color}]{view.state.value}[/{color}]")  # fmt: skip
    if view.recommended_version_id is not None:
        score = f"{view.quality_score:.1f}/100" if view.quality_score is not None else "—"
        gates = "every gate passes" if view.gates_passed else "fails: " + ", ".join(g.name for g in view.gates if not g.passed)  # fmt: skip
        console.print(f"recommended version {view.recommended_version_id} ({view.version_kind} v{view.version_number}) · quality report #{view.quality_report_id} · score {score} · {gates}")  # fmt: skip
    if view.decision is not None:
        d = view.decision
        console.print(f"decision #{d.id}: {d.decision.value} by {escape(d.approver)} ({d.method.value}, {d.channel.value}) {d.created_at:%Y-%m-%d %H:%M}" + (f" — {escape(d.note)}" if d.note else ""))  # fmt: skip
    elif view.last_decision is not None:
        d = view.last_decision
        console.print(f"last decision #{d.id}: {d.decision.value} for version {d.version_id} — no longer applies: {escape(d.invalidated_reason or '')}")  # fmt: skip
    for reason in view.blocking:
        err.print(f"[yellow]• {escape(reason)}[/yellow]")
    if view.auto_approve:
        console.print("[dim]PUBLISH_AUTO_APPROVE is on: publishing a ready article records an automatic approval[/dim]")  # fmt: skip


def _approval_view(article_id: int) -> ApprovalView:
    settings = get_settings()

    async def work(_: AsyncEngine, sessions: SessionFactory) -> ApprovalView:
        return await ApprovalService(sessions, settings).view(article_id)

    try:
        return _run_db(work)
    except _publishing_errors() as exc:
        err.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc


@articles_cli.command("approval")
def article_approval_cmd(article_id: int, json_output: bool = typer.Option(False, "--json")) -> None:  # fmt: skip
    """The approval as it stands: recommended version, quality score, gates, decision."""
    view = _approval_view(article_id)
    if json_output:
        sys.stdout.write(view.model_dump_json(indent=2) + "\n")
        return
    _print_approval(view)


@articles_cli.command("approve")
def approve_article_cmd(
    article_id: int,
    note: str | None = typer.Option(None, help="Why (recorded with the approval)."),
    approver: str | None = typer.Option(None, help="Who approves (default: cli)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation."),
) -> None:
    """Approve the recommended version and its quality report for publication (ready
    articles only). A new version or validation needs a new approval."""
    settings = get_settings()
    view = _approval_view(article_id)
    _print_approval(view)
    if view.state.value == "not_ready":
        raise typer.Exit(code=2)
    if not yes:
        typer.confirm(f"Approve version {view.recommended_version_id} (quality report #{view.quality_report_id}) for publication?", abort=True)  # fmt: skip

    async def work(_: AsyncEngine, sessions: SessionFactory) -> tuple[ApprovalRecord, bool]:
        return await ApprovalService(sessions, settings).approve(article_id, channel=ApprovalChannel.CLI, approver=approver, note=note)  # fmt: skip

    try:
        record, created = _run_db(work)
    except _publishing_errors() as exc:
        err.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    console.print(f"[green]approved[/green] (decision #{record.id})" if created else f"already approved (decision #{record.id}): nothing changed")  # fmt: skip


@articles_cli.command("reject")
def reject_article_cmd(
    article_id: int,
    note: str = typer.Option(..., help="Why it's rejected (required)."),
    approver: str | None = typer.Option(None, help="Who rejects (default: cli)."),
) -> None:
    """Reject the recommended version: it can't be published until approved, or until a
    new version is validated and approved."""
    settings = get_settings()

    async def work(_: AsyncEngine, sessions: SessionFactory) -> tuple[ApprovalRecord, bool]:
        return await ApprovalService(sessions, settings).reject(article_id, channel=ApprovalChannel.CLI, approver=approver, note=note)  # fmt: skip

    try:
        record, created = _run_db(work)
    except _publishing_errors() as exc:
        err.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    console.print(f"[red]rejected[/red] (decision #{record.id}): {escape(note)}" if created else f"already rejected (decision #{record.id})")  # fmt: skip


@articles_cli.command("approvals")
def article_approvals_cmd(article_id: int, json_output: bool = typer.Option(False, "--json")) -> None:  # fmt: skip
    """Every decision, oldest first, and why each stopped applying."""
    settings = get_settings()

    async def work(_: AsyncEngine, sessions: SessionFactory) -> list[ApprovalRecord]:
        return await ApprovalService(sessions, settings).history(article_id)

    try:
        rows = _run_db(work)
    except _publishing_errors() as exc:
        err.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    if json_output:
        sys.stdout.write("[" + ",".join(r.model_dump_json() for r in rows) + "]\n")
        return
    table = Table("ID", "Decision", "Version", "Report", "Method", "By", "When", "Live", "Note / why it stopped applying")  # fmt: skip
    for r in rows:
        table.add_row(str(r.id), r.decision.value, str(r.version_id), f"#{r.quality_report_id}", f"{r.method.value} ({r.channel.value})", escape(r.approver), f"{r.created_at:%Y-%m-%d %H:%M}", "yes" if r.live else "", escape((r.invalidated_reason or r.note or "")[:70]))  # fmt: skip
    console.print(table if rows else f"No decisions yet: `articles approve {article_id}`.")


def _print_preflight(report: PreflightReport) -> None:
    verdict = "[green]READY[/green]" if report.ready else "[red]BLOCKED[/red]"
    console.print(f"{verdict} · article {report.article_id} · version {report.version_id} · {report.cms} {report.site or '(not configured)'} · target {report.target_status.value} · action {report.action}")  # fmt: skip
    for c in report.checks:
        mark = "[green]✓[/green]" if c.passed else ("[red]✗[/red]" if c.blocking else "[yellow]![/yellow]")  # fmt: skip
        console.print(f"  {mark} {c.name}: {escape(c.detail)}")


def _publishing(engine: AsyncEngine, sessions: SessionFactory) -> tuple[PublishingService, LazyCMS]:  # fmt: skip
    settings = get_settings()
    # The image model is built on first use, so a publication without covers needs no key.
    covers = CoverService(sessions, settings, LazyLLM(settings))
    cms = LazyCMS(settings, covers=covers)
    return PublishingService(engine, sessions, settings, cms, covers=covers), cms


@articles_cli.command("preflight")
def article_preflight_cmd(
    article_id: int,
    status: Annotated[
        TargetStatus | None,
        typer.Option(
            "--status", help="draft, pending or publish (default: PUBLISH_DEFAULT_STATUS)."
        ),
    ] = None,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Every publishing check, the CMS included (read-only): nothing is changed."""

    async def work(engine: AsyncEngine, sessions: SessionFactory) -> PreflightReport:
        service, cms = _publishing(engine, sessions)
        try:
            return await service.preflight(article_id, target=status)
        finally:
            await cms.aclose()

    try:
        report = _run_db(work)
    except _publishing_errors() as exc:
        err.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    if json_output:
        sys.stdout.write(report.model_dump_json(indent=2) + "\n")
    else:
        _print_preflight(report)
    if not report.ready:
        raise typer.Exit(code=1)


@articles_cli.command("publish")
def publish_article_cmd(
    article_id: int,
    status: Annotated[
        TargetStatus | None,
        typer.Option(
            "--status",
            help="draft, pending or publish (publish needs PUBLISH_ALLOW_DIRECT_PUBLISH).",
        ),
    ] = None,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preflight, render and show the CMS request; change nothing."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Publish the approved recommended version to the CMS (runs now). Leaves a draft by
    default; the same version is never published twice. Nothing is scheduled."""
    if dry_run:
        _publish_dry_run(article_id, status, json_output)
        return

    async def work(engine: AsyncEngine, sessions: SessionFactory) -> tuple[PublishRequestResult, PublishOutcome | None]:  # fmt: skip
        service, cms = _publishing(engine, sessions)
        try:
            return await service.publish_now(article_id, trigger=RunTrigger.CLI, target=status)
        finally:
            await cms.aclose()

    try:
        result, outcome = _run_db(work)
    except _publishing_errors() as exc:
        err.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    status_value = (outcome.status if outcome else result.status).value if (outcome or result.status) else None  # type: ignore[union-attr]  # fmt: skip
    if json_output:
        payload = {"publication_id": result.publication_id, "created": result.created, "message": result.message, "run_id": result.run_id, "run_status": outcome.run_status.value if outcome else None, "status": status_value, "action": outcome.action if outcome else None, "external_id": outcome.external_id if outcome else None, "url": outcome.url if outcome else None, "error": outcome.error if outcome else None, "warnings": outcome.warnings if outcome else []}  # fmt: skip
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        if result.message:
            console.print(escape(result.message))
        if outcome is not None:
            color = {"published": "green", "draft_created": "green"}.get(
                outcome.status.value, "red"
            )
            console.print(f"publication {outcome.publication_id} run {outcome.run_id} → [{color}]{outcome.status.value}[/{color}]" + (f" · {outcome.action}" if outcome.action else "") + (f" · post {outcome.external_id}" if outcome.external_id else "") + (f" · {outcome.url}" if outcome.url else ""))  # fmt: skip
            for warning in outcome.warnings:
                err.print(f"[yellow]! {escape(warning)}[/yellow]")
            if outcome.error:
                err.print(f"[red]{escape(outcome.error)}[/red]")
            console.print(f"[dim]details: `articles publication {article_id}`[/dim]")
    if outcome is not None and outcome.run_status == RunStatus.FAILED:
        raise typer.Exit(code=1)


def _publish_dry_run(article_id: int, status: TargetStatus | None, json_output: bool) -> None:
    async def work(engine: AsyncEngine, sessions: SessionFactory) -> DryRunReport:
        service, cms = _publishing(engine, sessions)
        try:
            return await service.dry_run(article_id, target=status)
        finally:
            await cms.aclose()

    try:
        report = _run_db(work)
    except _publishing_errors() as exc:
        err.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    if json_output:
        sys.stdout.write(report.model_dump_json(indent=2) + "\n")
        return
    _print_preflight(report.preflight)
    console.rule("CMS request (dry run: nothing sent)")
    console.print_json(json.dumps(report.payload))
    doc = report.document
    if doc is not None:
        console.print(f"meta title: {escape(doc.meta_title)} · meta description: {escape(doc.excerpt)}")  # fmt: skip
        if doc.image is not None:
            console.print(f"image suggestion (not generated): {escape(doc.image.concept)} · alt: {escape(doc.image.alt_text)}")  # fmt: skip
        for note in doc.notes:
            console.print(f"[dim]{escape(note)}[/dim]")


def _print_publication(p: PublicationView) -> None:
    color = {"published": "green", "draft_created": "green", "blocked": "red", "failed": "red"}.get(p.status.value, "yellow")  # fmt: skip
    console.print(f"[bold]publication {p.id}[/bold] · article {p.article_id} version {p.version_id} · approval #{p.approval_id} · [{color}]{p.status.value}[/{color}] (target {p.target_status.value})")  # fmt: skip
    console.print(f"{p.cms} {p.site} · post {p.external_id or '—'} ({p.external_status or '—'})" + (f" · {p.url}" if p.url else "") + (f" · edit: {p.edit_url}" if p.edit_url else ""))  # fmt: skip
    d = p.details
    if d:
        console.print(f"title: {escape(str(d.get('title', '')))} · slug: {d.get('slug')} · meta title: {escape(str(d.get('meta_title', '')))}")  # fmt: skip
        console.print(f"meta description: {escape(str(d.get('meta_description', '')))}")
        category = d.get("category") or {}
        console.print(f"category: {escape(str(category.get('name', '—')))} · tags: {escape(', '.join(t['name'] for t in d.get('tags', [])) or '—')}" + (f" · left out: {escape(', '.join(d['missing_tags']))}" if d.get("missing_tags") else ""))  # fmt: skip
        if d.get("image_suggestion"):
            console.print(f"image suggestion (no image published): {escape(str(d['image_suggestion'].get('concept')))}")  # fmt: skip
    console.print(f"attempts {p.attempt_count} · created {p.created_at:%Y-%m-%d %H:%M} · updated {p.updated_at:%Y-%m-%d %H:%M}" + (f" · published {p.published_at:%Y-%m-%d %H:%M}" if p.published_at else "") + (f" · superseded by #{p.superseded_by_id}" if p.superseded_by_id else ""))  # fmt: skip
    for a in p.attempts:
        console.print(f"  {a.started_at:%H:%M:%S} {a.action.value} → {a.outcome.value}" + (f" ({a.http_status})" if a.http_status else "") + (f" post {a.external_id}" if a.external_id else "") + (f" — {escape(a.error[:120])}" if a.error else ""))  # fmt: skip
    if p.last_error:
        err.print(f"[yellow]{escape(p.last_error)}[/yellow]")


@articles_cli.command("publication")
def article_publication_cmd(article_id: int, json_output: bool = typer.Option(False, "--json")) -> None:  # fmt: skip
    """The latest publication: status, CMS post, URL, what was mapped, every attempt."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> PublicationView | None:
        async with sessions() as session:
            return await publishing_queries.current_publication(session, article_id)

    view = _run_db(work)
    if view is None:
        err.print(f"[red]Article {article_id} has no publication[/red]")
        raise typer.Exit(code=2)
    if json_output:
        sys.stdout.write(view.model_dump_json(indent=2) + "\n")
        return
    _print_publication(view)


@articles_cli.command("publications")
def article_publications_cmd(article_id: int, json_output: bool = typer.Option(False, "--json")) -> None:  # fmt: skip
    """Every publication (one per version and site), newest first."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> list[PublicationView] | None:
        async with sessions() as session:
            return await publishing_queries.list_publications(session, article_id)

    rows = _run_db(work)
    if rows is None:
        err.print(f"[red]Unknown article {article_id}[/red]")
        raise typer.Exit(code=2)
    if json_output:
        sys.stdout.write("[" + ",".join(r.model_dump_json() for r in rows) + "]\n")
        return
    table = Table("ID", "Version", "Approval", "Status", "Target", "Post", "URL", "Attempts", "Updated", "Superseded")  # fmt: skip
    for r in rows:
        table.add_row(str(r.id), str(r.version_id), f"#{r.approval_id}", r.status.value, r.target_status.value, f"{r.external_id or '—'} ({r.external_status or '—'})", r.url or "—", str(r.attempt_count), f"{r.updated_at:%Y-%m-%d %H:%M}", f"#{r.superseded_by_id}" if r.superseded_by_id else "")  # fmt: skip
    console.print(table if rows else f"No publications yet: `articles publish {article_id}` (a draft by default).")  # fmt: skip


# ── Phase 8: jobs, the schedule, the pipeline, the worker ────────────────────

jobs_cli = typer.Typer(no_args_is_help=True, help="Jobs: scheduled and manual pipeline runs, their stages and reports.")  # fmt: skip
schedule_cli = typer.Typer(no_args_is_help=True, help="The schedule: status (the dashboard), list, run a job now, pause, resume.")  # fmt: skip
pipeline_cli = typer.Typer(no_args_is_help=True, help="The autonomous pipeline: scan → analyze → opportunities → generate → validate → approval → publish.")  # fmt: skip
cli.add_typer(jobs_cli, name="jobs")
cli.add_typer(schedule_cli, name="schedule")
cli.add_typer(pipeline_cli, name="pipeline")

_JOB_COLORS = {"completed": "green", "completed_with_warnings": "yellow", "failed": "red", "skipped": "dim", "cancelled": "dim", "running": "cyan", "queued": "cyan"}  # fmt: skip


def _run_jobs[T](work: Callable[[Scheduling], Awaitable[T]]) -> T:
    """Like ``_run_db``, with every service a job may call (created on first use)."""
    settings = get_settings()

    async def runner() -> T:
        async with standalone(settings, pooled=False) as scheduling:
            return await work(scheduling)

    try:
        return asyncio.run(runner())
    except ProgrammingError as exc:
        err.print(f"[red]Database schema problem:[/red] {exc.orig}")
        err.print("Run [bold]uv run python -m app db upgrade[/bold] to apply migrations.")
        raise typer.Exit(code=2) from exc
    except (OperationalError, DBAPIError, OSError) as exc:
        err.print(f"[red]Cannot reach the database[/red] at {settings.database_url_display}")
        err.print("Start it with [bold]docker compose up -d db[/bold] or set DATABASE_URL.")
        raise typer.Exit(code=2) from exc


def _status_text(value: str) -> str:
    color = _JOB_COLORS.get(value, "white")
    return f"[{color}]{value}[/{color}]"


def _short(summary: dict[str, object], limit: int = 90) -> str:
    parts = []
    for key, value in summary.items():
        if isinstance(value, bool | int | float) or value is None:
            parts.append(f"{key}={value}")
        elif isinstance(value, list):
            parts.append(f"{key}={len(value)}" if len(value) > 5 else f"{key}={value}")
    text = ", ".join(parts)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _print_job(job: JobView) -> None:
    when = f" · for {job.scheduled_for:%Y-%m-%d %H:%M} UTC" if job.scheduled_for else ""
    console.print(f"job {job.id} · {job.job_type.value} · {_status_text(job.status.value)} · {job.trigger.value}{when} · attempt {job.attempt_count}/{job.max_attempts}" + (" · dry run" if job.dry_run else "") + (f" · retry of {job.parent_id}" if job.parent_id else ""))  # fmt: skip
    if job.checkpoint:
        console.print(f"checkpoint: {job.checkpoint}")
    if job.status is JobStatus.QUEUED and job.attempt_count:
        console.print(f"[yellow]retried after {job.run_after:%Y-%m-%d %H:%M} UTC (by the worker: `python -m app worker`)[/yellow]")  # fmt: skip
    if job.last_error:
        (err.print if job.status is JobStatus.FAILED else console.print)(f"[{'red' if job.status is JobStatus.FAILED else 'yellow'}]{escape(job.last_error[:600])}[/]" + (f" ({job.error_kind.value})" if job.error_kind else ""))  # fmt: skip
    if job.stages:
        table = Table("Stage", "Status", "Started", "Took", "Summary", "Warnings")
        for s in job.stages:
            took = f"{(s.finished_at - s.started_at).total_seconds():.0f}s" if s.started_at and s.finished_at else "—"  # fmt: skip
            table.add_row(s.stage.value, _status_text(s.status.value), f"{s.started_at:%H:%M:%S}" if s.started_at else "—", took, escape(_short(s.summary)), str(len(s.warnings)) if s.warnings else "")  # fmt: skip
        console.print(table)
        for s in job.stages:
            for warning in s.warnings[:5]:
                console.print(f"[yellow]! {s.stage.value}: {escape(warning[:300])}[/yellow]")
    today = job.report.get("today")
    if isinstance(today, dict):
        console.print(f"today ({today.get('date')}, {today.get('timezone')}): generated {today.get('generated')}/{today.get('generation_limit')} · ready {today.get('ready')} · published {today.get('published')}/{today.get('publication_limit')} (remaining {today.get('remaining')}) · drafts {today.get('drafts')}")  # fmt: skip


def _print_plan(plan: PipelinePlan) -> None:
    console.print(f"[bold]Plan[/bold] for {plan.job_type.value} at {plan.generated_at:%Y-%m-%d %H:%M} UTC: stages {', '.join(s.value for s in plan.stages)}")  # fmt: skip
    for note in plan.notes:
        console.print(f"[dim]· {escape(note)}[/dim]")
    if plan.analysis:
        table = Table("Competitor", "To analyze", "Carried forward", "Batches", "≈ input tokens")
        for a in plan.analysis:
            table.add_row(str(a.get("competitor")), str(a.get("to_analyze", a.get("error", "—"))), str(a.get("carry_forward", "—")), str(a.get("batches", "—")), f"{a.get('estimated_input_tokens', 0):,}")  # fmt: skip
        console.print(table)
    if plan.opportunities:
        table = Table("Opp.", "Origin", "Score", "Fit", "Evidence", "Status", "Selected", "Title")
        for o in plan.opportunities[:15]:
            table.add_row(str(o.opportunity_id), o.origin.value, f"{o.score:.1f}", f"{o.strategic_fit:.2f}" if o.strategic_fit is not None else "—", str(o.evidence), o.status, "[green]yes[/green]" if o.selected else "no", escape(o.title[:60]))  # fmt: skip
        console.print(table)
    t = plan.today
    console.print(f"generation: {plan.generation_remaining} of {plan.generation_limit} left today (competitors) · {t.editorial_remaining} of {t.editorial_limit} (editorial) · publishing: {plan.publication_remaining} of {plan.publication_limit} left today")  # fmt: skip
    if plan.editorial_topics_needed:
        console.print(f"editorial: {plan.editorial_topics_needed} new idea(s) would be proposed (one Gemini call)")  # fmt: skip
    for label, rows in (("validation", plan.validation), ("publishing", plan.publishing)):
        if rows:
            table = Table("Article", "Status", "Score", "Approval", "Action", "Title", title=label)
            for r in rows[:15]:
                table.add_row(str(r.article_id), r.status, f"{r.score:.1f}" if r.score is not None else "—", r.approval or "—", escape(r.action), escape(r.title[:50]))  # fmt: skip
            console.print(table)


def _job_exit(job: JobView) -> None:
    if job.status is JobStatus.FAILED:
        raise typer.Exit(code=1)


def _run_job_now(job_type: JobType, *, dry_run: bool, json_output: bool) -> None:
    """Queue a job and run it here, now: the same locks, limits, gates and switches as a
    scheduled run (a job of the same type already running means this one is skipped)."""

    async def work(scheduling: Scheduling) -> JobView:
        view, _ = await scheduling.jobs.enqueue(job_type, trigger=JobTrigger.CLI, dry_run=dry_run)
        return await scheduling.jobs.run(view.id)

    job = _run_jobs(work)
    plan = PipelinePlan.model_validate(job.report["plan"]) if dry_run and job.report.get("plan") else None  # fmt: skip
    if json_output:
        payload = {"job": job.model_dump(mode="json"), "plan": plan.model_dump(mode="json") if plan else None}  # fmt: skip
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    elif plan is not None:
        _print_plan(plan)
    else:
        _print_job(job)
    _job_exit(job)


@jobs_cli.command("list")
def jobs_list_cmd(
    status: Annotated[JobStatus | None, typer.Option("--status")] = None,
    job_type: Annotated[JobType | None, typer.Option("--type")] = None,
    limit: int = typer.Option(25, min=1, max=500),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Recent jobs, newest first (scheduled, manual, skipped, retried)."""

    async def work(scheduling: Scheduling) -> list[JobView]:
        return await scheduling.jobs.find(status=status, job_type=job_type, limit=limit)

    rows = _run_jobs(work)
    if json_output:
        sys.stdout.write("[" + ",".join(r.model_dump_json() for r in rows) + "]\n")
        return
    table = Table("ID", "Type", "Status", "Trigger", "For (UTC)", "Created", "Try", "Checkpoint", "Error")  # fmt: skip
    for j in rows:
        table.add_row(str(j.id), j.job_type.value + (" (dry run)" if j.dry_run else ""), _status_text(j.status.value), j.trigger.value, f"{j.scheduled_for:%m-%d %H:%M}" if j.scheduled_for else "—", f"{j.created_at:%Y-%m-%d %H:%M}", f"{j.attempt_count}/{j.max_attempts}", j.checkpoint or "—", escape((j.last_error or "")[:60]))  # fmt: skip
    console.print(table if rows else "No jobs yet: `pipeline run --dry-run`, `schedule run <job>`, or start the worker.")  # fmt: skip


@jobs_cli.command("show")
def jobs_show_cmd(job_id: int = typer.Argument(..., min=1), json_output: bool = typer.Option(False, "--json")) -> None:  # fmt: skip
    """One job: its stages, checkpoint, attempts, error and report."""

    async def work(scheduling: Scheduling) -> JobView:
        return await scheduling.jobs.get(job_id)

    try:
        job = _run_jobs(work)
    except JobNotFoundError as exc:
        err.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    if json_output:
        sys.stdout.write(job.model_dump_json(indent=2) + "\n")
    else:
        _print_job(job)


@jobs_cli.command("retry")
def jobs_retry_cmd(job_id: int = typer.Argument(..., min=1), json_output: bool = typer.Option(False, "--json")) -> None:  # fmt: skip
    """Retry a failed job now: a new job continuing from its checkpoints (finished stages
    aren't repeated; no duplicate article, approval or post)."""

    async def work(scheduling: Scheduling) -> JobView:
        view = await scheduling.jobs.retry(job_id, actor="cli")
        return await scheduling.jobs.run(view.id)

    try:
        job = _run_jobs(work)
    except (JobNotFoundError, JobConflictError) as exc:
        err.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    if json_output:
        sys.stdout.write(job.model_dump_json(indent=2) + "\n")
    else:
        _print_job(job)
    _job_exit(job)


@jobs_cli.command("cancel")
def jobs_cancel_cmd(job_id: int = typer.Argument(..., min=1)) -> None:
    """Cancel a queued job, or stop a running one at its next checkpoint."""

    async def work(scheduling: Scheduling) -> JobView:
        return await scheduling.jobs.cancel(job_id, actor="cli")

    try:
        job = _run_jobs(work)
    except (JobNotFoundError, JobConflictError) as exc:
        err.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    note = "stops at its next checkpoint" if job.cancel_requested and job.status is JobStatus.RUNNING else job.status.value  # fmt: skip
    console.print(f"job {job.id}: {note}")


@schedule_cli.command("status")
def schedule_status_cmd(json_output: bool = typer.Option(False, "--json")) -> None:
    """The dashboard: switches, today's articles and allowances, today's jobs, next runs."""

    async def work(scheduling: Scheduling) -> SchedulerStatus:
        return await scheduling.state.status()

    s = _run_jobs(work)
    if json_output:
        sys.stdout.write(s.model_dump_json(indent=2) + "\n")
        return
    state = "[green]enabled[/green]" if s.enabled else ("[yellow]paused[/yellow]" if s.paused else "[dim]disabled (SCHEDULER_ENABLED=false)[/dim]")  # fmt: skip
    console.print(f"scheduler: {state} · timezone {s.timezone} · pipelines at once: {s.max_concurrent_pipelines}")  # fmt: skip
    console.print(f"publishing: automated {'[green]on[/green]' if s.automated_publishing else '[dim]off[/dim]'} · auto-approve {'on' if s.auto_approve else 'off'} · target {s.publish_target} · direct publish {'allowed' if s.direct_publish else 'not allowed'}")  # fmt: skip
    t = s.today
    console.print(f"today {t.date}: generated {t.generated}/{t.generation_limit} (left {t.generation_remaining}) · editorial {t.editorial_generated}/{t.editorial_limit} (left {t.editorial_remaining}) · ready {t.ready} · published {t.published}/{t.publication_limit} (left {t.remaining}) · drafts {t.drafts}")  # fmt: skip
    if s.llm_tokens_left_today is not None:
        console.print(f"LLM tokens left today (UTC): {s.llm_tokens_left_today:,}")
    console.print("jobs today: " + (", ".join(f"{k} {v}" for k, v in sorted(s.jobs_today.items())) or "none") + (f" · running: {s.running}" if s.running else ""))  # fmt: skip
    for v in s.next_runs:
        nxt = v.next_runs[0].astimezone(ZoneInfo(s.timezone)) if v.next_runs else None
        console.print(f"next {v.job_type.value}: {nxt:%Y-%m-%d %H:%M %Z}" if nxt else f"next {v.job_type.value}: —")  # fmt: skip
    for warning in s.warnings:
        console.print(f"[yellow]! {escape(warning)}[/yellow]")


@schedule_cli.command("list")
def schedule_list_cmd(json_output: bool = typer.Option(False, "--json")) -> None:
    """The configured schedules, their next runs and their last job."""

    async def work(scheduling: Scheduling) -> list[ScheduleView]:
        return await scheduling.state.schedules()

    rows = _run_jobs(work)
    if json_output:
        sys.stdout.write("[" + ",".join(r.model_dump_json() for r in rows) + "]\n")
        return
    table = Table("Job", "Setting", "Cron", "Timezone", "Next runs (local)", "Last job")
    for v in rows:
        tz = ZoneInfo(v.timezone)
        table.add_row(v.job_type.value, v.setting, v.expression, v.timezone, ", ".join(f"{r.astimezone(tz):%m-%d %H:%M}" for r in v.next_runs), f"{v.last_job_id} ({v.last_status.value})" if v.last_job_id and v.last_status else "—")  # fmt: skip
    console.print(table if rows else "No schedule is set: e.g. FULL_PIPELINE_SCHEDULE=\"0 6 * * *\" (and SCHEDULER_ENABLED=true for the worker).")  # fmt: skip


@schedule_cli.command("run")
def schedule_run_cmd(job_type: JobType, json_output: bool = typer.Option(False, "--json")) -> None:  # fmt: skip
    """Run a job now (full_pipeline or one stage), with every lock, limit, gate and switch
    of a scheduled run. Works while the scheduler is paused or disabled."""
    _run_job_now(job_type, dry_run=False, json_output=json_output)


@schedule_cli.command("pause")
def schedule_pause_cmd(reason: str | None = typer.Option(None, help="Why (shown in the status).")) -> None:  # fmt: skip
    """Pause scheduled runs (recorded as skipped) without a restart. Manual runs still work."""

    async def work(scheduling: Scheduling) -> SchedulerStatus:
        return await scheduling.state.pause(reason=reason, actor="cli")

    _run_jobs(work)
    console.print("scheduler paused: scheduled occurrences are recorded as skipped until `schedule resume`")  # fmt: skip


@schedule_cli.command("resume")
def schedule_resume_cmd() -> None:
    async def work(scheduling: Scheduling) -> SchedulerStatus:
        return await scheduling.state.resume(actor="cli")

    status = _run_jobs(work)
    console.print("scheduler resumed" + ("" if status.configured else " (but SCHEDULER_ENABLED=false: nothing fires until it is true)"))  # fmt: skip


@pipeline_cli.command("run")
def pipeline_run_cmd(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Planning mode: no site fetched, no Gemini call, no CMS change, no allowance used.",
    ),
    job_type: Annotated[
        JobType, typer.Option("--job", help="full_pipeline (default) or one stage.")
    ] = JobType.FULL_PIPELINE,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run the pipeline now (or plan it with --dry-run)."""
    _run_job_now(job_type, dry_run=dry_run, json_output=json_output)


@cli.command()
def worker() -> None:
    """The scheduler process: fires the configured schedules (SCHEDULER_ENABLED=true) and runs
    queued jobs (manual runs, retries). Stop it with Ctrl-C: running jobs are requeued and
    continue from their checkpoints on the next start."""
    asyncio.run(run_worker(get_settings()))
