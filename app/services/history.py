"""Records a scan into history: discovered URLs, captured versions, and detected changes.

Runs in one transaction per scan while the competitor's scan lock is held, so no other
writer touches this competitor's rows concurrently. Deterministic; no LLM.
"""

import dataclasses
import gzip
import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import case, func, literal_column, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.crawling.extract import EXTRACTOR_VERSION
from app.db.models import (
    ChangeEvent,
    Competitor,
    ContentItem,
    ContentVersion,
    RawDocument,
    Run,
    RunEvent,
)
from app.domain.content import ContentType, DateSource
from app.domain.history import DATE_SOURCE_TRUST, ChangeType, ItemStatus
from app.domain.scan import DiscoveredPage, ScanIssue, ScanItem, ScanResult
from app.services.change_detection import detect_price_change, diff_texts
from app.services.monitoring import GONE_STATUSES

_BATCH = 1_000
_EMPTY_HASH = hashlib.sha256(b"").hexdigest()


@dataclass
class RecordSummary:
    baseline: bool = False  # first scan of this competitor: nothing is reported as "new"
    new_urls: int = 0  # URLs seen for the first time (after the baseline)
    first_captures: int = 0  # pages captured for the first time (version 1)
    updated: int = 0  # significant main-text changes
    minor_updates: int = 0
    pricing_changed: int = 0
    unchanged: int = 0
    not_modified: int = 0
    removed: int = 0
    restored: int = 0
    duplicates: int = 0
    raw_documents: int = 0

    def as_dict(self) -> dict[str, int | bool]:
        return dataclasses.asdict(self)


def _batched(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class HistoryRecorder:
    def __init__(self, *, store_raw_html: bool = True) -> None:
        self._store_raw_html = store_raw_html

    async def record(
        self, session: AsyncSession, competitor: Competitor, run: Run, result: ScanResult
    ) -> RecordSummary:
        now = result.finished_at
        summary = RecordSummary(baseline=not await self._has_content(session, competitor.id))
        new_ids = await self._upsert_discovered(session, competitor, run, result.discovered, now, summary)  # fmt: skip
        for item in [*result.items, *result.captured_outside_window]:
            await self._record_capture(session, competitor, run, item, now, summary, new_ids)
        for url, target_url in result.aliases:
            await self._mark_duplicate(session, competitor, url, target_url, now, summary, new_ids)
        await self._record_not_modified(session, competitor, result.not_modified, now, summary)
        await self._record_removals(session, competitor, run, result.skipped, now, summary)
        if not summary.baseline:
            for item_id in sorted(new_ids):
                session.add(self._event(competitor, run, item_id, ChangeType.NEW, now))
        summary.new_urls = 0 if summary.baseline else len(new_ids)
        self._record_run_events(session, run, result, now)
        await session.flush()
        return summary

    # ── discovery ────────────────────────────────────────────────────────────

    @staticmethod
    async def _has_content(session: AsyncSession, competitor_id: int) -> bool:
        query = select(ContentItem.id).where(ContentItem.competitor_id == competitor_id).limit(1)
        return await session.scalar(query) is not None

    async def _upsert_discovered(
        self,
        session: AsyncSession,
        competitor: Competitor,
        run: Run,
        pages: Sequence[DiscoveredPage],
        now: datetime,
        summary: RecordSummary,
    ) -> set[int]:
        """Insert new URLs, refresh known ones; returns ids of rows inserted now."""
        if not pages:
            return set()
        urls = list(dict.fromkeys(page.url for page in pages))
        existing: set[str] = set()
        for chunk in _batched(urls, _BATCH):
            found = await session.scalars(
                select(ContentItem.url).where(
                    ContentItem.competitor_id == competitor.id, ContentItem.url.in_(chunk)
                )
            )
            existing.update(found)

        insert = pg_insert(ContentItem)
        excluded = insert.excluded
        upsert = insert.on_conflict_do_update(
            constraint="uq_content_items_competitor_id_url",
            set_={
                "last_seen_at": excluded.last_seen_at,
                "updated_at": excluded.updated_at,
                "sitemap_lastmod": func.coalesce(
                    excluded.sitemap_lastmod, ContentItem.sitemap_lastmod
                ),
                "modified_at": func.coalesce(excluded.modified_at, ContentItem.modified_at),
                "title": func.coalesce(ContentItem.title, excluded.title),
                "discovered_via": literal_column(
                    "ARRAY(SELECT DISTINCT unnest(content_items.discovered_via "
                    "|| EXCLUDED.discovered_via) ORDER BY 1)"
                ),
                # A page-level classification (after capture) beats the URL-based guess.
                "content_type": case(
                    (ContentItem.current_version_id.is_(None), excluded.content_type),
                    else_=ContentItem.content_type,
                ),
                # Never overwrite a known publication date from discovery-level signals.
                "published_at_source": case(
                    (ContentItem.published_at.is_(None), excluded.published_at_source),
                    else_=ContentItem.published_at_source,
                ),
                "published_at": func.coalesce(ContentItem.published_at, excluded.published_at),
            },
        )
        rows = [
            {
                "competitor_id": competitor.id,
                "url": page.url,
                "status": ItemStatus.DISCOVERED.value,
                "content_type": page.content_type.value,
                "title": page.title,
                "discovered_via": [s.value for s in page.discovered_via],
                "in_baseline": summary.baseline,
                "published_at": page.published_at,
                "published_at_source": page.published_at_source.value
                if page.published_at_source
                else None,
                "modified_at": page.feed_updated,
                "sitemap_lastmod": page.sitemap_lastmod,
                "first_seen_at": now,
                "last_seen_at": now,
                "first_seen_run_id": run.id,
                "created_at": now,
                "updated_at": now,
            }
            for page in {p.url: p for p in pages}.values()
        ]
        for chunk in _batched(rows, _BATCH):
            await session.execute(upsert, list(chunk))

        new_urls = [url for url in urls if url not in existing]
        new_ids: set[int] = set()
        for chunk in _batched(new_urls, _BATCH):
            ids = await session.scalars(
                select(ContentItem.id).where(
                    ContentItem.competitor_id == competitor.id, ContentItem.url.in_(chunk)
                )
            )
            new_ids.update(ids)
        return new_ids

    # ── captures ─────────────────────────────────────────────────────────────

    async def _record_capture(
        self,
        session: AsyncSession,
        competitor: Competitor,
        run: Run,
        item: ScanItem,
        now: datetime,
        summary: RecordSummary,
        new_ids: set[int],
    ) -> None:
        # Content is keyed by where it actually lives: the URL after redirects.
        content = await self._get_item(session, competitor.id, item.final_url)
        if content is None:
            content = ContentItem(
                competitor_id=competitor.id,
                url=item.final_url,
                status=ItemStatus.ACTIVE.value,
                content_type=item.content_type.value,
                discovered_via=[],
                in_baseline=summary.baseline,
                first_seen_at=now,
                last_seen_at=now,
                first_seen_run_id=run.id,
            )
            session.add(content)
            await session.flush()
            new_ids.add(content.id)
        if item.url != item.final_url:
            await self._mark_duplicate(
                session, competitor, item.url, item.final_url, now, summary, new_ids
            )

        previous_status = content.status
        content.status = ItemStatus.ACTIVE.value
        content.duplicate_of_id = None
        content.content_type = item.content_type.value
        content.title = item.title or content.title
        content.discovered_via = sorted(
            set(content.discovered_via) | {s.value for s in item.discovered_via}
        )
        content.last_seen_at = now
        content.last_fetched_at = item.fetched_at or now
        content.etag = item.etag
        content.last_modified_header = item.last_modified
        content.sitemap_lastmod = item.sitemap_lastmod or content.sitemap_lastmod
        content.modified_at = item.modified_at or content.modified_at
        _apply_publication_date(content, item.published_at, item.date_source)
        if previous_status == ItemStatus.REMOVED.value:
            summary.restored += 1
            session.add(self._event(competitor, run, content.id, ChangeType.RESTORED, now))

        current = (
            await session.get(ContentVersion, content.current_version_id)
            if content.current_version_id
            else None
        )
        content_hash = item.content_hash or _EMPTY_HASH
        if current is not None and current.content_hash == content_hash:
            summary.unchanged += 1
            return

        version = ContentVersion(
            content_item_id=content.id,
            version_no=content.version_count + 1,
            run_id=run.id,
            raw_document_id=await self._store_raw(session, competitor, item, now, summary),
            observed_at=item.fetched_at or now,
            final_url=item.final_url,
            canonical_url=item.canonical_url,
            http_status=item.http_status or 200,
            content_type=item.content_type.value,
            classification_reason=item.classification_reason,
            title=item.title,
            description=item.description,
            author=item.author,
            language=item.language,
            published_at=item.published_at,
            published_at_source=item.date_source.value if item.date_source else None,
            modified_at=item.modified_at,
            categories=item.categories,
            tags=item.tags,
            headings=[heading.model_dump() for heading in item.headings],
            structured_types=item.structured_types,
            text=item.full_text,
            word_count=item.word_count,
            content_hash=content_hash,
            is_thin=item.is_thin,
            extractor_version=EXTRACTOR_VERSION,
        )
        session.add(version)
        await session.flush()
        content.current_version_id = version.id
        content.version_count += 1
        if current is None:
            summary.first_captures += 1
            return

        content.last_changed_at = version.observed_at
        diff = diff_texts(
            current.text, version.text, old_title=current.title, new_title=version.title
        )
        details: dict[str, Any] = {
            **diff.as_details(),
            "title_before": current.title,
            "title_after": version.title,
            "word_count_before": current.word_count,
            "word_count_after": version.word_count,
        }
        session.add(
            self._event(
                competitor, run, content.id, ChangeType.UPDATED, now,
                from_version=current, to_version=version, is_minor=diff.is_minor, details=details,
            )
        )  # fmt: skip
        if diff.is_minor:
            summary.minor_updates += 1
        else:
            summary.updated += 1
        pricing = ContentType.PRICING.value
        if pricing in (current.content_type, version.content_type):
            price_change = detect_price_change(current.text, version.text)
            if price_change is not None:
                summary.pricing_changed += 1
                session.add(
                    self._event(
                        competitor, run, content.id, ChangeType.PRICING_CHANGED, now,
                        from_version=current, to_version=version, details=price_change.as_details(),
                    )
                )  # fmt: skip

    async def _store_raw(
        self,
        session: AsyncSession,
        competitor: Competitor,
        item: ScanItem,
        now: datetime,
        summary: RecordSummary,
    ) -> int | None:
        if not self._store_raw_html or item.raw_html is None:
            return None
        raw = RawDocument(
            competitor_id=competitor.id,
            url=item.url,
            final_url=item.final_url,
            fetched_at=item.fetched_at or now,
            http_status=item.http_status or 200,
            content_type=item.raw_content_type,
            etag=item.etag,
            last_modified=item.last_modified,
            sha256=hashlib.sha256(item.raw_html).hexdigest(),
            size_bytes=len(item.raw_html),
            compression="gzip",
            body=gzip.compress(item.raw_html, compresslevel=6),
        )
        session.add(raw)
        await session.flush()
        summary.raw_documents += 1
        return raw.id

    async def _mark_duplicate(
        self,
        session: AsyncSession,
        competitor: Competitor,
        url: str,
        target_url: str,
        now: datetime,
        summary: RecordSummary,
        new_ids: set[int],
    ) -> None:
        """``url`` redirects to (or declares as canonical) ``target_url``: not separate content."""
        target = await self._get_item(session, competitor.id, target_url)
        if target is None or url == target_url:
            return
        alias = await self._get_item(session, competitor.id, url)
        if alias is None:
            alias = ContentItem(
                competitor_id=competitor.id,
                url=url,
                content_type=target.content_type,
                discovered_via=[],
                in_baseline=summary.baseline,
                first_seen_at=now,
                first_seen_run_id=None,
                last_seen_at=now,
            )
            session.add(alias)
        alias.status = ItemStatus.DUPLICATE.value
        alias.duplicate_of_id = target.id
        alias.last_seen_at = now
        alias.last_fetched_at = now  # fetched this scan: don't re-fetch without a change signal
        await session.flush()
        new_ids.discard(alias.id)  # an alias is not new content
        summary.duplicates += 1

    async def _record_not_modified(
        self,
        session: AsyncSession,
        competitor: Competitor,
        urls: Sequence[str],
        now: datetime,
        summary: RecordSummary,
    ) -> None:
        for url in urls:
            content = await self._get_item(session, competitor.id, url)
            if content is not None:
                content.last_fetched_at = now
                content.last_seen_at = now
                summary.not_modified += 1

    async def _record_removals(
        self,
        session: AsyncSession,
        competitor: Competitor,
        run: Run,
        issues: Sequence[ScanIssue],
        now: datetime,
        summary: RecordSummary,
    ) -> None:
        """404/410 means gone. Absence from a sitemap does NOT: scans are bounded."""
        for issue in issues:
            if issue.http_status not in GONE_STATUSES:
                continue
            content = await self._get_item(session, competitor.id, issue.url)
            if content is None or content.status not in (ItemStatus.ACTIVE, ItemStatus.DISCOVERED):
                continue
            was_captured = content.status == ItemStatus.ACTIVE.value
            content.status = ItemStatus.REMOVED.value
            content.last_fetched_at = now
            if was_captured:
                summary.removed += 1
                session.add(
                    self._event(
                        competitor, run, content.id, ChangeType.REMOVED, now,
                        details={"http_status": issue.http_status},
                    )
                )  # fmt: skip

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    async def _get_item(session: AsyncSession, competitor_id: int, url: str) -> ContentItem | None:
        item: ContentItem | None = await session.scalar(
            select(ContentItem).where(
                ContentItem.competitor_id == competitor_id, ContentItem.url == url
            )
        )
        return item

    @staticmethod
    def _event(
        competitor: Competitor,
        run: Run,
        item_id: int,
        change_type: ChangeType,
        now: datetime,
        *,
        from_version: ContentVersion | None = None,
        to_version: ContentVersion | None = None,
        is_minor: bool = False,
        details: dict[str, Any] | None = None,
    ) -> ChangeEvent:
        return ChangeEvent(
            competitor_id=competitor.id,
            content_item_id=item_id,
            run_id=run.id,
            change_type=change_type.value,
            detected_at=now,
            is_minor=is_minor,
            from_version_id=from_version.id if from_version else None,
            to_version_id=to_version.id if to_version else None,
            details=details or {},
        )

    @staticmethod
    def _record_run_events(
        session: AsyncSession, run: Run, result: ScanResult, now: datetime
    ) -> None:
        for level, issues in (("info", result.skipped), ("warning", result.errors)):
            for issue in issues:
                session.add(
                    RunEvent(
                        run_id=run.id,
                        created_at=now,
                        level=level,
                        event=issue.reason,
                        url=issue.url,
                        detail=issue.detail,
                        data={"http_status": issue.http_status} if issue.http_status else {},
                    )
                )


def _apply_publication_date(
    content: ContentItem, published_at: datetime | None, source: DateSource | None
) -> None:
    """Keep the stored date unless the new one comes from a more trustworthy source."""
    if published_at is None or source is None:
        return
    current_trust = (
        DATE_SOURCE_TRUST[DateSource(content.published_at_source)]
        if content.published_at is not None and content.published_at_source
        else 0
    )
    if DATE_SOURCE_TRUST[source] > current_trust:
        content.published_at = published_at
        content.published_at_source = source.value
