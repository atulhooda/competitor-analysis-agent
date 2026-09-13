"""Command-line interface: ``uv run python -m app --help``."""

import asyncio
import sys
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table

from app.config import Settings, find_competitor, get_settings, load_competitors
from app.core.errors import ConfigurationError
from app.core.logging import configure_logging
from app.core.timeutils import parse_since
from app.crawling.fetcher import PoliteFetcher
from app.domain.competitors import CompetitorConfig
from app.domain.scan import ScanResult
from app.services.monitoring import MonitoringService

cli = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Competitor intelligence agent. Phase 1: compliant competitor website monitoring.",
)
console = Console()
err = Console(stderr=True)


@cli.callback()
def _main(
    log_level: str | None = typer.Option(None, help="Override LOG_LEVEL (DEBUG, INFO, WARNING)."),
) -> None:
    settings = get_settings()
    configure_logging(log_level or settings.log_level, json_output=settings.log_json)


def _load_or_exit(settings: Settings) -> list[CompetitorConfig]:
    try:
        return load_competitors(settings.competitors_file)
    except ConfigurationError as exc:
        err.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


@cli.command()
def check() -> None:
    """Validate configuration. Secret values are never printed."""
    settings = get_settings()
    table = Table(title="Configuration", show_header=False)
    table.add_column("Setting")
    table.add_column("Value")
    ok = True
    try:
        competitors = load_competitors(settings.competitors_file)
        table.add_row("Competitors", f"{len(competitors)} in {settings.competitors_file}")
    except ConfigurationError as exc:
        ok = False
        table.add_row("Competitors", f"[red]{exc}[/red]")
    table.add_row("Environment", settings.app_env)
    table.add_row("Crawler user agent", settings.crawler_user_agent)
    table.add_row("Crawler minimum delay", f"{settings.crawler_min_delay_seconds}s per host")
    table.add_row("LLM provider", "Google Gemini")
    table.add_row("GEMINI_MODEL", settings.gemini_model)
    table.add_row(
        "GEMINI_API_KEY",
        "set"
        if settings.llm_configured
        else "not set (not needed for Phase 1; required from Phase 3)",
    )
    table.add_row("API_KEY", "set" if settings.api_key else "not set (allowed in development only)")
    console.print(table)
    raise typer.Exit(code=0 if ok else 1)


@cli.command("competitors")
def list_competitors() -> None:
    """List configured competitors."""
    competitors = _load_or_exit(get_settings())
    table = Table("Slug", "Name", "Website", "Tracked pages")
    for c in competitors:
        table.add_row(c.slug, c.name, str(c.website), str(len(c.tracked_pages)))
    console.print(table)


@cli.command()
def scan(
    slug: str = typer.Argument(..., help="Competitor slug from the competitors file."),
    since: str | None = typer.Option(None, help="Window start: 7d, 24h, 2w or an ISO date."),
    limit: int | None = typer.Option(None, min=1, max=500, help="Max discovered pages to fetch."),
    json_output: bool = typer.Option(False, "--json", help="Print the full result as JSON."),
    include_text: bool = typer.Option(False, help="Include extracted main text (JSON output)."),
) -> None:
    """Scan one competitor's website (robots-compliant, rate-limited, no LLM)."""
    settings = get_settings()
    competitor = find_competitor(_load_or_exit(settings), slug)
    if competitor is None:
        err.print(f"[red]Unknown competitor {slug!r}.[/red] Run `python -m app competitors`.")
        raise typer.Exit(code=2)
    try:
        since_at = parse_since(since) if since else None
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--since") from exc
    result = asyncio.run(_run_scan(settings, competitor, since_at, limit, include_text))
    if json_output:
        sys.stdout.write(result.model_dump_json(indent=2) + "\n")
    else:
        _render(result)
    raise typer.Exit(code=1 if result.status == "failed" else 0)


async def _run_scan(
    settings: Settings,
    competitor: CompetitorConfig,
    since: datetime | None,
    limit: int | None,
    include_text: bool,
) -> ScanResult:
    async with PoliteFetcher(settings) as fetcher:
        service = MonitoringService(fetcher, settings)
        return await service.scan(competitor, since=since, limit=limit, include_text=include_text)


def _render(result: ScanResult) -> None:
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
    console.print(f"feeds: {', '.join(result.feeds) or 'none found'}")
    console.print(f"sitemaps read: {len(result.sitemaps)}")
    table = Table("Published", "Type", "Title", "URL", "Words", show_lines=False)
    for item in result.items:
        date = item.published_at or item.modified_at or item.sitemap_lastmod
        table.add_row(
            f"{date:%Y-%m-%d}" if date else "—",
            item.content_type.value,
            (item.title or "")[:70],
            item.final_url,
            str(item.word_count),
        )
    console.print(table)
    s = result.stats
    console.print(
        f"candidates={s.candidates} (feeds {s.from_feeds}, sitemaps {s.from_sitemaps}) "
        f"fetched={s.fetched} items={s.items} outside_window={s.outside_window} "
        f"excluded={s.excluded} over_limit={s.over_limit} robots_disallowed={s.robots_disallowed} "
        f"errors={s.errors}"
    )
    for issue in result.skipped[:10]:
        console.print(f"[dim]skipped[/dim] {issue.reason}: {issue.url}")
    for issue in result.errors[:10]:
        console.print(f"[yellow]error[/yellow] {issue.reason}: {issue.url} {issue.detail or ''}")
