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

from app.config import get_settings, load_competitors
from app.core.errors import ConfigurationError
from app.core.logging import configure_logging
from app.core.timeutils import parse_since, utcnow
from app.crawling.fetcher import PoliteFetcher
from app.db import migrate, queries
from app.db.session import SessionFactory, create_engine, create_session_factory
from app.domain.content import ContentType
from app.domain.history import ChangeType, RunTrigger
from app.domain.scan import ScanResult
from app.services.monitoring import MonitoringService
from app.services.scans import ScanError, ScanOutcome, ScanService

cli = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Competitor intelligence agent: compliant website monitoring with persisted history.",
)
db_cli = typer.Typer(no_args_is_help=True, help="Database migrations.")
competitors_cli = typer.Typer(help="Manage monitored competitors (stored in the database).")
cli.add_typer(db_cli, name="db")
cli.add_typer(competitors_cli, name="competitors")
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
        else "not set (not needed for Phases 1 and 2; required from Phase 3)",
    )
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
    """List recent scan runs."""

    async def work(_: AsyncEngine, sessions: SessionFactory) -> None:
        async with sessions() as session:
            views = await queries.list_runs(session, competitor=slug, limit=limit)
        table = Table("Run", "Competitor", "Trigger", "Status", "Started", "Fetched", "New", "Updated", "Error")  # fmt: skip
        for r in views:
            changes_ = r.summary.get("changes", {})
            table.add_row(
                str(r.id), r.competitor or "—", r.trigger.value, r.status.value,
                f"{r.started_at:%Y-%m-%d %H:%M}" if r.started_at else "—",
                str(r.stats.get("fetched", "—")), str(changes_.get("new_urls", "—")),
                str(changes_.get("updated", "—")), (r.error or "")[:50],
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
