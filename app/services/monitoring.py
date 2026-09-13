"""Competitor website monitoring: discover → filter → fetch → extract → classify.

Deterministic, no database, no LLM. Pages are fetched politely and per-page failures
are isolated, so one bad URL never aborts a scan. Given what earlier scans captured
(``known``), a scan becomes incremental: captured pages are re-fetched only when a feed
or sitemap reports a newer date (plus a small revisit budget), with conditional GET.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

import structlog

from app.config import Settings
from app.core.timeutils import utcnow
from app.crawling.classify import classify_page, classify_url
from app.crawling.errors import (
    BlockedError,
    ContentParseError,
    FetchError,
    HTTPStatusError,
    OutOfScopeError,
    RobotsDisallowedError,
)
from app.crawling.extract import ExtractedPage, extract_page
from app.crawling.feeds import COMMON_FEED_PATHS, FeedEntry, parse_feed
from app.crawling.fetcher import FetchKind, FetchResult, PoliteFetcher
from app.crawling.html import decode_html, scan_html
from app.crawling.sitemaps import crawl_sitemaps
from app.crawling.urls import SiteScope, normalize_url, origin_of
from app.domain.competitors import CompetitorConfig
from app.domain.content import DATED_CONTENT_TYPES, ContentType, DateSource, DiscoverySource
from app.domain.history import ItemStatus, KnownPage
from app.domain.scan import (
    DiscoveredPage,
    Heading,
    RobotsSummary,
    ScanIssue,
    ScanItem,
    ScanResult,
    ScanStats,
    ScanStatus,
)

log = structlog.get_logger(__name__)

_BLOG_INDEX_PATH = re.compile(
    r"^/(?:[a-z]{2}(?:-[a-z]{2})?/)?(?:blog|news|articles|insights|resources|posts)/?$"
)
_TYPE_PRIORITY = {
    ContentType.BLOG_POST: 0,
    ContentType.CASE_STUDY: 1,
    ContentType.PRICING: 2,
    ContentType.PRODUCT: 3,
    ContentType.LANDING_PAGE: 4,
    ContentType.PRESS: 5,
    ContentType.CHANGELOG: 6,
    ContentType.RESOURCE: 7,
    ContentType.DOCS: 8,
    ContentType.OTHER: 9,
}
_FEED_TYPE_UPGRADES = frozenset({ContentType.OTHER, ContentType.LANDING_PAGE, ContentType.PRODUCT})
_MAX_ISSUES = 200
GONE_STATUSES = frozenset({404, 410})


@dataclass
class _Candidate:
    url: str
    sources: set[DiscoverySource] = field(default_factory=set)
    title: str | None = None
    feed_published: datetime | None = None
    feed_updated: datetime | None = None
    feed_author: str | None = None
    feed_tags: tuple[str, ...] = ()
    lastmod: datetime | None = None
    news_published: datetime | None = None

    @property
    def always_fetch(self) -> bool:
        return bool(self.sources & {DiscoverySource.HOMEPAGE, DiscoverySource.TRACKED})

    @property
    def window_date(self) -> datetime | None:
        dates = [self.feed_published, self.feed_updated, self.news_published, self.lastmod]
        known = [d for d in dates if d is not None]
        return max(known) if known else None


@dataclass
class _ScanState:
    stats: ScanStats = field(default_factory=ScanStats)
    items: list[ScanItem] = field(default_factory=list)
    skipped: list[ScanIssue] = field(default_factory=list)
    errors: list[ScanIssue] = field(default_factory=list)
    seen_urls: set[str] = field(default_factory=set)
    feeds: list[str] = field(default_factory=list)
    prefetched_feeds: dict[str, list[FeedEntry]] = field(default_factory=dict)
    discovered: list[DiscoveredPage] = field(default_factory=list)
    captured_outside_window: list[ScanItem] = field(default_factory=list)
    aliases: list[tuple[str, str]] = field(default_factory=list)
    not_modified: list[str] = field(default_factory=list)

    def skip(
        self, url: str, reason: str, detail: str | None = None, http_status: int | None = None
    ) -> None:
        if len(self.skipped) < _MAX_ISSUES:
            self.skipped.append(
                ScanIssue(url=url, reason=reason, detail=detail, http_status=http_status)
            )

    def error(
        self, url: str, reason: str, detail: str | None = None, http_status: int | None = None
    ) -> None:
        self.stats.errors += 1
        if len(self.errors) < _MAX_ISSUES:
            self.errors.append(
                ScanIssue(url=url, reason=reason, detail=detail, http_status=http_status)
            )

    def record_fetch_failure(self, url: str, exc: FetchError) -> None:
        status = exc.status if isinstance(exc, HTTPStatusError) else None
        if isinstance(exc, RobotsDisallowedError):
            self.stats.robots_disallowed += 1
            self.skip(url, exc.code, exc.detail)
        elif isinstance(exc, BlockedError | OutOfScopeError):
            self.skip(url, exc.code, exc.detail)
        elif status in GONE_STATUSES:
            # The page no longer exists: an observation about the competitor, not a failure
            # of the scan. History records it as a removal.
            self.skip(url, "gone", exc.detail, http_status=status)
        else:
            self.error(url, exc.code, exc.detail, http_status=status)


class MonitoringService:
    def __init__(
        self,
        fetcher: PoliteFetcher,
        settings: Settings,
        *,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        self._fetcher = fetcher
        self._settings = settings
        self._now = now

    async def scan(
        self,
        competitor: CompetitorConfig,
        *,
        since: datetime | None = None,
        limit: int | None = None,
        include_text: bool = False,
        known: Mapping[str, KnownPage] | None = None,
    ) -> ScanResult:
        """Scan one competitor. ``known`` (URL → what earlier scans captured) makes it incremental."""
        started_at = self._now()
        known = known or {}
        limit = limit or self._settings.crawler_default_scan_limit
        requests_before = self._fetcher.request_count
        state = _ScanState()
        slog = log.bind(competitor=competitor.slug)
        home = str(competitor.website)
        scope = SiteScope.from_urls(
            [home, *map(str, competitor.feeds), *map(str, competitor.sitemaps)]
            + [str(u) for u in competitor.tracked_pages],
            competitor.allowed_domains,
        )
        slog.info("scan.start", since=since.isoformat() if since else None, limit=limit)

        def finish(
            status: ScanStatus | None = None,
            robots: RobotsSummary | None = None,
            sitemaps: list[str] | None = None,
        ) -> ScanResult:
            state.stats.items = len(state.items)
            state.stats.discovered = len(state.discovered)
            state.stats.http_requests = self._fetcher.request_count - requests_before
            state.items.sort(key=_item_sort_key)
            result = ScanResult(
                competitor=competitor.slug,
                status=status or ("partial" if state.errors else "ok"),
                started_at=started_at,
                finished_at=self._now(),
                since=since,
                limit=limit,
                robots=robots,
                feeds=state.feeds,
                sitemaps=sitemaps or [],
                items=state.items,
                skipped=state.skipped,
                errors=state.errors,
                not_modified=state.not_modified,
                stats=state.stats,
                discovered=state.discovered,
                captured_outside_window=state.captured_outside_window,
                aliases=state.aliases,
            )
            slog.info("scan.done", status=result.status, **state.stats.model_dump())
            return result

        # 1. robots.txt governs everything else.
        try:
            policy = await self._fetcher.robots_policy(home)
        except FetchError as exc:
            state.error(home, exc.code, exc.detail)
            return finish("failed")
        robots = RobotsSummary(url=policy.url, status=policy.status, crawl_delay=policy.crawl_delay)
        if policy.status == "unreachable":
            state.error(policy.url, "robots_unreachable", "RFC 9309: assuming complete disallow")
            return finish("failed", robots)

        # 2. Homepage (always fetched; also the starting point for feed discovery).
        candidates: dict[str, _Candidate] = {}
        home_candidate = self._add(candidates, home, DiscoverySource.HOMEPAGE)
        for page in competitor.tracked_pages:
            self._add(candidates, str(page), DiscoverySource.TRACKED)
        home_fetch = await self._fetch_extract(home, scope, state)
        if home_fetch is not None:
            self._record_item(state, home_candidate, *home_fetch, competitor, since, include_text)
        site_url = home_fetch[0].final_url if home_fetch else home  # after redirects
        home_page = home_fetch[1] if home_fetch else None

        # 3. Feeds: configured, advertised, or probed.
        feed_urls = [str(f) for f in competitor.feeds] or await self._discover_feeds(
            site_url, home_page, scope, state
        )
        for feed_url in feed_urls[: self._settings.crawler_max_feeds]:
            for entry in await self._read_feed(feed_url, scope, state):
                candidate = self._add(candidates, entry.url, DiscoverySource.FEED)
                candidate.title = candidate.title or entry.title
                candidate.feed_published = entry.published
                candidate.feed_updated = entry.updated
                candidate.feed_author = entry.author
                candidate.feed_tags = entry.tags
                state.stats.from_feeds += 1

        # 4. Sitemaps: configured, declared in robots.txt, or the conventional location.
        declared = [str(s) for s in competitor.sitemaps] or [
            s for s in policy.sitemaps if scope.contains(s)
        ]
        roots = declared or [origin_of(site_url) + "/sitemap.xml"]
        crawl = await crawl_sitemaps(
            self._fetcher,
            roots,
            scope=scope,
            since=since,
            max_files=self._settings.crawler_max_sitemap_files,
            max_urls=self._settings.crawler_max_sitemap_urls,
        )
        for url, detail in crawl.errors:
            if declared:
                state.error(url, "sitemap_error", detail)
            else:
                state.skip(url, "no_sitemap", detail)  # a guessed location simply may not exist
        for sitemap_entry in crawl.entries:
            candidate = self._add(candidates, sitemap_entry.url, DiscoverySource.SITEMAP)
            candidate.lastmod = sitemap_entry.lastmod
            candidate.news_published = sitemap_entry.news_published
            state.stats.from_sitemaps += 1
        slog.info(
            "scan.discovered",
            candidates=len(candidates),
            feeds=len(state.feeds),
            sitemaps=len(crawl.fetched),
            sitemap_truncated=crawl.truncated,
        )

        # 5. Filter and prioritize, drop robots-disallowed URLs, then fetch within the limits.
        fixed, ordered, stale = self._select(candidates, competitor, scope, since, state, known)
        selected, over_limit = await self._apply_robots_and_limit(ordered, limit, state)
        revisits, _ = await self._apply_robots_and_limit(
            stale, self._settings.crawler_revisit_limit, state
        )
        state.stats.over_limit = over_limit
        state.stats.revisited = len(revisits)
        state.discovered = await self._robots_allowed(state.discovered)
        for candidate in [*fixed, *selected, *revisits]:
            if candidate.url in state.seen_urls:
                continue
            fetched = await self._fetch_extract(
                candidate.url, scope, state, previous=known.get(candidate.url)
            )
            if fetched is not None:
                self._record_item(state, candidate, *fetched, competitor, since, include_text)

        return finish(robots=robots, sitemaps=crawl.fetched)

    # ── discovery ────────────────────────────────────────────────────────────

    @staticmethod
    def _add(candidates: dict[str, _Candidate], url: str, source: DiscoverySource) -> _Candidate:
        normalized = normalize_url(url) or url
        candidate = candidates.setdefault(normalized, _Candidate(url=normalized))
        candidate.sources.add(source)
        return candidate

    async def _discover_feeds(
        self, site_url: str, home_page: ExtractedPage | None, scope: SiteScope, state: _ScanState
    ) -> list[str]:
        if home_page is not None:
            advertised = [u for u in home_page.signals.feed_links if scope.contains(u)]
            if advertised:
                return advertised
            # The blog index often advertises the feed even when the homepage doesn't.
            index = next(
                (
                    link
                    for link in home_page.signals.links
                    if scope.contains(link) and _BLOG_INDEX_PATH.match(urlsplit(link).path.lower())
                ),
                None,
            )
            if index is not None:
                try:
                    result = await self._fetcher.fetch(index, FetchKind.PAGE, scope=scope)
                    signals = scan_html(
                        decode_html(result.content, result.content_type), result.final_url
                    )
                    advertised = [u for u in signals.feed_links if scope.contains(u)]
                    if advertised:
                        return advertised
                except FetchError as exc:
                    log.info("feed.index_unavailable", url=index, error=str(exc))
        # Last resort: probe a few conventional locations (each probe is a paced request).
        origin = origin_of(site_url)
        for path in COMMON_FEED_PATHS:
            probe = origin + path
            try:
                result = await self._fetcher.fetch(probe, FetchKind.FEED, scope=scope)
                entries = parse_feed(result.content, result.final_url)
            except (FetchError, ContentParseError):
                continue
            state.prefetched_feeds[result.final_url] = entries
            return [result.final_url]
        return []

    async def _read_feed(
        self, feed_url: str, scope: SiteScope, state: _ScanState
    ) -> list[FeedEntry]:
        if feed_url in state.prefetched_feeds:
            state.feeds.append(feed_url)
            return state.prefetched_feeds[feed_url]
        try:
            result = await self._fetcher.fetch(feed_url, FetchKind.FEED, scope=scope)
            entries = parse_feed(result.content, result.final_url)
        except FetchError as exc:
            state.record_fetch_failure(feed_url, exc)
            return []
        except ContentParseError as exc:
            state.error(feed_url, exc.code, exc.detail)
            return []
        state.feeds.append(feed_url)
        return entries

    # ── selection ────────────────────────────────────────────────────────────

    def _select(
        self,
        candidates: dict[str, _Candidate],
        competitor: CompetitorConfig,
        scope: SiteScope,
        since: datetime | None,
        state: _ScanState,
        known: Mapping[str, KnownPage],
    ) -> tuple[list[_Candidate], list[_Candidate], list[_Candidate]]:
        """Return (always-fetched pages, candidates in priority order, stale captured pages).

        Priority: never-captured pages with a publication date, then with only a sitemap
        lastmod, then captured pages whose feed/sitemap date is newer than our last fetch,
        then the undated backlog. Captured pages with no sign of change are skipped; the
        stalest active ones are returned separately for the revisit budget.
        """
        includes = [re.compile(p) for p in competitor.include_patterns]
        excludes = [re.compile(p) for p in competitor.exclude_patterns]
        stats = state.stats
        stats.candidates = len(candidates)
        now = self._now()
        revisit_cutoff = now - timedelta(days=self._settings.crawler_revisit_after_days)
        fixed: list[_Candidate] = []
        dated: list[tuple[_Candidate, ContentType]] = []
        undated: list[tuple[_Candidate, ContentType]] = []
        refresh: list[_Candidate] = []
        stale: list[tuple[_Candidate, datetime]] = []
        for candidate in candidates.values():
            if not scope.contains(candidate.url):
                stats.out_of_scope += 1
                continue
            if candidate.always_fetch:
                state.discovered.append(
                    _discovered(candidate, classify_url(candidate.url).content_type, now)
                )
                if DiscoverySource.HOMEPAGE not in candidate.sources:  # homepage: fetched already
                    stats.tracked += 1
                    fixed.append(candidate)
                continue
            if includes and not any(p.search(candidate.url) for p in includes):
                stats.excluded += 1
                continue
            if any(p.search(candidate.url) for p in excludes):
                stats.excluded += 1
                continue
            content_type = classify_url(candidate.url).content_type
            if DiscoverySource.FEED in candidate.sources and content_type in _FEED_TYPE_UPGRADES:
                content_type = ContentType.BLOG_POST
            if content_type in competitor.exclude_types:
                stats.excluded += 1
                continue
            state.discovered.append(_discovered(candidate, content_type, now))
            window_date = candidate.window_date
            if since is not None and (
                (window_date is not None and window_date < since)
                or (window_date is None and content_type not in DATED_CONTENT_TYPES)
            ):
                stats.outside_window += 1  # older, or undated and not provably inside the window
                continue
            previous = known.get(candidate.url)
            if previous is None or previous.last_fetched_at is None:
                (dated if window_date is not None else undated).append((candidate, content_type))
            elif window_date is not None and window_date > previous.last_fetched_at:
                refresh.append(candidate)  # the site reports a change since our last fetch
            elif previous.status is ItemStatus.ACTIVE and previous.last_fetched_at < revisit_cutoff:
                stale.append((candidate, previous.last_fetched_at))
            else:
                stats.known_unchanged += 1

        dated.sort(key=lambda pair: _dated_priority(pair[0]))
        refresh.sort(key=lambda c: (-_timestamp(c.window_date), c.url))
        undated.sort(key=lambda pair: (_TYPE_PRIORITY.get(pair[1], 99), pair[0].url))
        stale.sort(key=lambda pair: (pair[1], pair[0].url))  # least recently fetched first
        ordered = [c for c, _ in dated] + refresh + [c for c, _ in undated]
        return fixed, ordered, [c for c, _ in stale]

    async def _robots_allowed(self, pages: list[DiscoveredPage]) -> list[DiscoveredPage]:
        """Never record URLs that robots.txt disallows, even ones we won't fetch now.

        Policies are cached per origin, so this makes no requests per URL.
        """
        allowed = []
        for page in pages:
            try:
                if (await self._fetcher.robots_policy(page.url)).can_fetch(page.url):
                    allowed.append(page)
            except FetchError:
                continue
        return allowed

    async def _apply_robots_and_limit(
        self, ordered: list[_Candidate], limit: int, state: _ScanState
    ) -> tuple[list[_Candidate], int]:
        """Drop robots-disallowed URLs *before* applying the limit, so they don't use up budget.

        Policies are cached per origin, so this costs no extra requests per URL.
        Returns (selected candidates, allowed candidates left over the limit).
        """
        allowed: list[_Candidate] = []
        for candidate in ordered:
            try:
                policy = await self._fetcher.robots_policy(candidate.url)
            except FetchError as exc:
                state.record_fetch_failure(candidate.url, exc)
                continue
            if policy.can_fetch(candidate.url):
                allowed.append(candidate)
            else:
                state.stats.robots_disallowed += 1
                state.skip(candidate.url, "robots_disallowed", f"robots.txt status={policy.status}")
        return allowed[:limit], max(0, len(allowed) - limit)

    # ── fetch, extract, record ───────────────────────────────────────────────

    async def _fetch_extract(
        self, url: str, scope: SiteScope, state: _ScanState, previous: KnownPage | None = None
    ) -> tuple[FetchResult, ExtractedPage] | None:
        # Conditional GET for pages we already hold a capture of.
        captured = previous is not None and previous.status is ItemStatus.ACTIVE
        try:
            result = await self._fetcher.fetch(
                url,
                FetchKind.PAGE,
                scope=scope,
                etag=previous.etag if captured and previous else None,
                last_modified=previous.last_modified if captured and previous else None,
            )
        except FetchError as exc:
            state.record_fetch_failure(url, exc)
            return None
        if result.not_modified:
            state.not_modified.append(url)
            state.stats.not_modified += 1
            return None
        state.stats.fetched += 1
        try:
            page = extract_page(decode_html(result.content, result.content_type), result.final_url)
        except Exception as exc:  # per-page isolation: a pathological page must not abort the scan
            log.warning("extract.failed", url=url, error=repr(exc))
            state.error(url, "extract_error", repr(exc))
            return None
        return result, page

    def _record_item(
        self,
        state: _ScanState,
        candidate: _Candidate,
        result: FetchResult,
        page: ExtractedPage,
        competitor: CompetitorConfig,
        since: datetime | None,
        include_text: bool,
    ) -> None:
        canonical = page.signals.canonical
        keys = {candidate.url, result.final_url}
        # Some sites wrongly point every page's canonical at the homepage; only trust a
        # root canonical on the root page itself, or every later page would look like a duplicate.
        if canonical and (not _is_root(canonical) or _is_root(result.final_url)):
            keys.add(canonical)
        if keys & state.seen_urls:
            # Duplicate of a page already recorded (redirect or canonical): remember the alias
            # so later scans don't keep re-fetching it as if it were uncaptured.
            state.stats.excluded += 1
            target = result.final_url if result.final_url in state.seen_urls else canonical
            if target and target != candidate.url:
                state.aliases.append((candidate.url, target))
            return
        state.seen_urls.update(keys)

        classification = classify_page(
            result.final_url,
            discovered_via=candidate.sources,
            jsonld_types=page.signals.jsonld_types,
            og_type=page.signals.meta.get("og:type"),
        )
        if not candidate.always_fetch and classification.content_type in competitor.exclude_types:
            state.stats.excluded += 1
            return
        published_at, date_source = _resolve_published(page, candidate)
        if (
            date_source is DateSource.PAGE
            and classification.content_type not in DATED_CONTENT_TYPES
        ):
            # trafilatura's date heuristics invent dates for non-articles (a homepage came
            # back as "2008-01-01" from a copyright year), so only trust them for articles.
            published_at, date_source = None, None
        now = self._now()
        if not _plausible(published_at, now):
            published_at, date_source = None, None
        modified_at = page.modified_at or candidate.feed_updated
        if not _plausible(modified_at, now):
            modified_at = None
        item = ScanItem(
            url=candidate.url,
            final_url=result.final_url,
            canonical_url=canonical,
            content_type=classification.content_type,
            classification_reason=classification.reason,
            discovered_via=sorted(candidate.sources),
            title=page.title or candidate.title,
            description=page.description,
            author=page.author or candidate.feed_author,
            published_at=published_at,
            modified_at=modified_at,
            date_source=date_source,
            sitemap_lastmod=candidate.lastmod,
            categories=list(page.categories),
            tags=list(page.tags or candidate.feed_tags),
            language=page.language,
            word_count=page.word_count,
            content_hash=page.content_hash,
            is_thin=page.is_thin,
            text=page.text if include_text else None,
            fetched_at=now,
            http_status=result.status,
            headings=[Heading(level=level, text=text) for level, text in page.headings],
            structured_types=sorted(page.signals.jsonld_types),
            full_text=page.text,
            raw_html=result.content,
            raw_content_type=result.content_type,
            etag=result.etag,
            last_modified=result.last_modified,
        )
        if since is not None and not candidate.always_fetch:
            dates = [d for d in (published_at, modified_at, candidate.lastmod) if d is not None]
            if not dates or max(dates) < since:
                state.stats.outside_window += 1
                state.captured_outside_window.append(item)  # persisted, not reported
                return
        state.items.append(item)


def _resolve_published(
    page: ExtractedPage, candidate: _Candidate
) -> tuple[datetime | None, DateSource | None]:
    """Structured data and meta tags beat feed dates, which beat sitemap-news and heuristics."""
    if page.date_source in (DateSource.STRUCTURED_DATA, DateSource.META):
        return page.published_at, page.date_source
    if candidate.feed_published is not None:
        return candidate.feed_published, DateSource.FEED
    if candidate.news_published is not None:
        return candidate.news_published, DateSource.SITEMAP_NEWS
    return page.published_at, page.date_source


def _discovered(candidate: _Candidate, content_type: ContentType, now: datetime) -> DiscoveredPage:
    """Discovery-time facts. Only feed and news-sitemap dates count as publication dates;
    sitemap lastmod is kept separately as a modification hint."""
    published_at: datetime | None = None
    source: DateSource | None = None
    if candidate.feed_published is not None and _plausible(candidate.feed_published, now):
        published_at, source = candidate.feed_published, DateSource.FEED
    elif candidate.news_published is not None and _plausible(candidate.news_published, now):
        published_at, source = candidate.news_published, DateSource.SITEMAP_NEWS
    return DiscoveredPage(
        url=candidate.url,
        discovered_via=sorted(candidate.sources),
        content_type=content_type,
        title=candidate.title,
        published_at=published_at,
        published_at_source=source,
        feed_updated=candidate.feed_updated if _plausible(candidate.feed_updated, now) else None,
        sitemap_lastmod=candidate.lastmod if _plausible(candidate.lastmod, now) else None,
    )


_EARLIEST_PLAUSIBLE = datetime(1995, 1, 1, tzinfo=UTC)
_FUTURE_TOLERANCE = timedelta(days=1)


def _plausible(value: datetime | None, now: datetime) -> bool:
    """Web content dates before 1995 or in the future are extraction errors."""
    return value is None or _EARLIEST_PLAUSIBLE <= value <= now + _FUTURE_TOLERANCE


def _dated_priority(candidate: _Candidate) -> tuple[int, float, str]:
    """Publication dates (feed, news sitemap) outrank sitemap ``lastmod``.

    ``lastmod`` records modification, not publication: a deploy or bulk edit can stamp
    a quarter of a site with today's date and crowd out genuinely new posts.
    """
    published = candidate.feed_published or candidate.news_published
    return (
        0 if published is not None else 1,
        -_timestamp(published or candidate.window_date),
        candidate.url,
    )


def _timestamp(value: datetime | None) -> float:
    return value.timestamp() if value is not None else 0.0


def _is_root(url: str) -> bool:
    return urlsplit(url).path in ("", "/")


def _item_sort_key(item: ScanItem) -> tuple[int, float, str]:
    fixed_first = 0 if item.content_type is ContentType.HOMEPAGE else 1
    date = item.published_at or item.modified_at or item.sitemap_lastmod
    return (fixed_first, -_timestamp(date), item.url)
