"""Assembles competitor and landscape intelligence from the analysis layer. No LLM.

Metrics are computed on request from the latest analyses, so they always reflect the
current data. Stored narratives (profiles, landscape reports) are attached as they are.
"""

from collections.abc import Callable
from datetime import datetime, timedelta

from sqlalchemy import select

from app.config import Settings
from app.core.timeutils import utcnow
from app.db import analysis_queries, queries
from app.db.models import Competitor
from app.db.session import SessionFactory
from app.domain.analysis import TopicRef
from app.domain.intelligence import (
    CompetitorIntelligence,
    CompetitorSnapshot,
    Landscape,
    RecentItem,
    TopicDetail,
)
from app.services.analysis import ANALYZER_VERSION
from app.services.scans import CompetitorNotFoundError
from app.services.topics import TopicRegistry
from app.services.trends import TrendEngine

TOP_TOPICS = 20
TOP_SUBTOPICS = 15
LANDSCAPE_TOPICS = 60


class IntelligenceService:
    def __init__(
        self, sessions: SessionFactory, settings: Settings, *, now: Callable[[], datetime] = utcnow
    ) -> None:
        self._sessions = sessions
        self._settings = settings
        self._now = now

    async def competitor(self, slug: str, *, window_days: int = 30) -> CompetitorIntelligence:
        now = self._now()
        async with self._sessions() as session:
            competitor = await queries.get_competitor(session, slug)
            if competitor is None:
                raise CompetitorNotFoundError(f"Unknown competitor {slug!r}")
            facts, topics = await analysis_queries.load_facts(session, competitor_ids=[competitor.id])  # fmt: skip
            coverage = await analysis_queries.coverage(session, competitor.id, self._settings, ANALYZER_VERSION)  # fmt: skip
            changes = await analysis_queries.recent_changes(
                session, competitor_ids=[competitor.id], since=now - timedelta(days=window_days)
            )
            recent = await analysis_queries.list_analyses(
                session,
                competitor=slug,
                published_since=now - timedelta(days=window_days),
                limit=20,
            )
            profile_row = await analysis_queries.latest_profile_row(session, competitor.id)
        engine = TrendEngine(facts, topics, now=now, window_days=window_days)
        return CompetitorIntelligence(
            competitor=competitor.slug,
            name=competitor.name,
            generated_at=now,
            basis=engine.basis(),
            coverage=coverage,
            cadence=engine.cadence(competitor=slug),
            topics=engine.topic_trends(competitor=slug)[:TOP_TOPICS],
            subtopics=engine.topic_trends(competitor=slug, subtopics=True)[:TOP_SUBTOPICS],
            formats=engine.mix("format", competitor=slug),
            audiences=engine.mix("audience", competitor=slug)[:12],
            intents=engine.mix("intent", competitor=slug),
            funnel_stages=engine.mix("funnel_stage", competitor=slug),
            strategy_shifts=engine.mix_shifts(competitor=slug),
            recent_items=[
                RecentItem(
                    content_item_id=a.content_item_id,
                    url=a.url,
                    title=a.title,
                    published_at=a.published_at,
                    content_format=a.content_format,
                    summary=a.summary,
                    topics=[
                        TopicRef(slug=t.slug, name=t.name) for t in a.topics if t.parent is None
                    ],
                )
                for a in recent
                if a.published_at is not None
            ],
            recent_changes=changes,
            profile=analysis_queries.profile_view(profile_row, slug) if profile_row else None,
        )

    async def topic_detail(self, slug: str, *, window_days: int = 30) -> TopicDetail | None:
        """One topic across all active competitors (follows merges)."""
        now = self._now()
        async with self._sessions() as session:
            requested = await analysis_queries.get_topic(session, slug)
            if requested is None:
                return None
            topic = await TopicRegistry(session).active(requested)
            views = await analysis_queries.list_topics(session, topic_ids=[topic.id])
            ids = list(await session.scalars(select(Competitor.id).where(Competitor.active.is_(True))))  # fmt: skip
            facts, topics = await analysis_queries.load_facts(session, competitor_ids=ids)
            recent = await analysis_queries.list_analyses(session, topic=topic.slug, limit=20)
        engine = TrendEngine(facts, topics, now=now, window_days=window_days)
        subtopic = topic.parent_id is not None
        trend = engine.topic_trends(subtopics=subtopic, topic_ids=[topic.id])
        return TopicDetail(
            topic=views[0],
            merged_from=slug if requested.id != topic.id else None,
            basis=engine.basis(),
            trend=trend[0] if trend else None,
            subtopics=[] if subtopic else engine.topic_trends(subtopics=True, parent_id=topic.id),
            recent_items=recent,
        )

    async def landscape(self, *, window_days: int = 30) -> Landscape:
        """Cross-competitor metrics for all active competitors."""
        now = self._now()
        async with self._sessions() as session:
            competitors = list(
                await session.scalars(
                    select(Competitor).where(Competitor.active.is_(True)).order_by(Competitor.slug)
                )
            )
            ids = [c.id for c in competitors]
            facts, topics = await analysis_queries.load_facts(session, competitor_ids=ids)
            changes = await analysis_queries.recent_changes(
                session,
                competitor_ids=ids,
                since=now - timedelta(days=window_days),
                summarized_only=True,
                limit=20,
            )
            positioning: dict[int, str | None] = {}
            for competitor in competitors:
                row = await analysis_queries.latest_profile_row(session, competitor.id)
                statement = row.profile.get("positioning_statement") if row else None
                positioning[competitor.id] = statement.get("text") if isinstance(statement, dict) else None  # fmt: skip
        engine = TrendEngine(facts, topics, now=now, window_days=window_days)
        trends = engine.topic_trends()
        return Landscape(
            generated_at=now,
            basis=engine.basis(),
            competitors=[
                CompetitorSnapshot(
                    competitor=c.slug,
                    name=c.name,
                    analyzed_items=engine.analyzed_items(c.slug),
                    cadence=engine.cadence(competitor=c.slug),
                    top_topics=engine.topic_trends(competitor=c.slug)[:8],
                    formats=engine.mix("format", competitor=c.slug)[:6],
                    audiences=engine.mix("audience", competitor=c.slug)[:6],
                    positioning_statement=positioning.get(c.id),
                )
                for c in competitors
            ],
            topics=trends[:LANDSCAPE_TOPICS],
            rising=engine.rising(trends)[:15],
            neglected=engine.neglected(trends),
            formats=engine.mix("format"),
            audiences=engine.mix("audience")[:15],
            intents=engine.mix("intent"),
            strategy_shifts=engine.mix_shifts(),
            recent_changes=changes,
        )
