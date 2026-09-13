"""Command-line interface: ``uv run python -m app --help``."""

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import get_settings, load_competitors, load_topic_seeds
from app.core.errors import ConfigurationError
from app.core.logging import configure_logging
from app.core.timeutils import parse_since, utcnow
from app.crawling.fetcher import PoliteFetcher
from app.db import analysis_queries, migrate, queries
from app.db.session import SessionFactory, create_engine, create_session_factory
from app.domain.analysis import MixShift, Share, TopicTrend
from app.domain.competitor_profile import Claim
from app.domain.content import ContentType
from app.domain.history import ChangeType, RunStatus, RunTrigger
from app.domain.scan import ScanResult
from app.llm import LazyLLM, LLMConfigurationError
from app.services.analysis import (
    AnalysisAlreadyRunningError,
    AnalysisOptions,
    AnalysisOutcome,
    AnalysisService,
)
from app.services.intelligence import IntelligenceService
from app.services.landscape import LandscapeAlreadyRunningError, LandscapeService
from app.services.llm_usage import usage_window_start, utc_day_start
from app.services.monitoring import MonitoringService
from app.services.scans import CompetitorNotFoundError, ScanError, ScanOutcome, ScanService
from app.services.topic_admin import TopicAdminService
from app.services.topics import TopicMergeError

cli = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Competitor intelligence agent: compliant website monitoring, persisted history, "
    "and AI analysis (Gemini).",
)
db_cli = typer.Typer(no_args_is_help=True, help="Database migrations.")
competitors_cli = typer.Typer(help="Manage monitored competitors (stored in the database).")
topics_cli = typer.Typer(help="The topic taxonomy: list, inspect, seed, merge, consolidate.")
cli.add_typer(db_cli, name="db")
cli.add_typer(competitors_cli, name="competitors")
cli.add_typer(topics_cli, name="topics")
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
