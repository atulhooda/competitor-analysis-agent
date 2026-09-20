"""The canonical topic taxonomy: label resolution, seeds, merges.

Topics have two levels (topic → subtopics). Every label that has ever resolved to a topic
is stored as an alias keyed by ``label_key`` within its scope (top level, or one parent's
subtopics), so resolution is deterministic and stable: once "Agentic AI" is merged into
"AI agents", every later "agentic AI" label lands on "AI agents" without an LLM.

All writes happen inside a transaction holding the taxonomy advisory lock, so concurrent
analysis runs can't create the same topic twice.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import PermanentError
from app.db.models import ContentAnalysisTopic, Topic, TopicAlias
from app.db.models.analysis import TOP_LEVEL_SCOPE
from app.domain.analysis import TopicOrigin, TopicRole, TopicStatus
from app.domain.topics import TopicSeed
from app.services.labels import clean_label, label_key, slugify

_TAXONOMY_LOCK_KEY = (72_010 << 32) | 1
_ROLE_RANK = {TopicRole.PRIMARY: 0, TopicRole.SECONDARY: 1, TopicRole.SUBTOPIC: 2}
_MAX_MERGE_DEPTH = 20
# When an analysis links to both the merged topic (:source) and the kept one (:target).
_MERGE_SHARED_LINKS = text(
    """
    UPDATE content_analysis_topics AS t
    SET relevance = GREATEST(t.relevance, s.relevance),
        role = CASE
            WHEN 'primary' IN (t.role, s.role) THEN 'primary'
            WHEN 'secondary' IN (t.role, s.role) THEN 'secondary'
            ELSE 'subtopic'
        END
    FROM content_analysis_topics AS s
    WHERE s.analysis_id = t.analysis_id AND s.topic_id = :source AND t.topic_id = :target
    """
)
_DELETE_SHARED_LINKS = text(
    """
    DELETE FROM content_analysis_topics AS s
    USING content_analysis_topics AS t
    WHERE s.analysis_id = t.analysis_id AND s.topic_id = :source AND t.topic_id = :target
    """
)


class TopicMergeError(PermanentError):
    pass


class TopicLabelLike(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def relevance(self) -> float: ...
    @property
    def subtopics(self) -> Sequence[str]: ...


@dataclass(frozen=True)
class ResolvedTopic:
    topic_id: int
    role: TopicRole
    relevance: float
    label: str


@dataclass(frozen=True)
class TaxonomyEntry:
    name: str
    subtopics: tuple[str, ...]


@dataclass
class SeedImportSummary:
    topics_created: int = 0
    subtopics_created: int = 0
    aliases_added: int = 0
    conflicts: list[str] = field(default_factory=list)


@dataclass
class MergeSummary:
    source: str
    target: str
    links_moved: int = 0  # analysis links re-pointed to the target
    links_combined: int = 0  # analyses tagged with both: merged into the target's link
    aliases_moved: int = 0
    subtopics_moved: int = 0
    subtopics_merged: int = 0


async def lock_taxonomy(session: AsyncSession) -> None:
    """Serialize taxonomy writes until the current transaction ends."""
    await session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _TAXONOMY_LOCK_KEY})


class TopicRegistry:
    """Resolves labels to topics within one session/transaction (call ``lock_taxonomy`` first)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._cache: dict[tuple[int, str], Topic] = {}
        self.created: list[Topic] = []

    # ── resolution ───────────────────────────────────────────────────────────

    async def resolve(
        self, label: str, parent: Topic | None = None, *, origin: TopicOrigin = TopicOrigin.LLM
    ) -> Topic | None:
        """The active topic for ``label`` (under ``parent``), created if unknown.

        Returns None for labels with no usable characters, and for a subtopic whose label
        is just its parent's name.
        """
        name = clean_label(label)
        key = label_key(name)
        if not key or (parent is not None and key == label_key(parent.name)):
            return None
        scope = parent.id if parent is not None else TOP_LEVEL_SCOPE
        cached = self._cache.get((scope, key))
        if cached is not None:
            return cached
        alias = await self._session.scalar(
            select(TopicAlias).where(TopicAlias.scope_id == scope, TopicAlias.key == key)
        )
        if alias is not None:
            topic = await self.active(await self._session.get_one(Topic, alias.topic_id))
        else:
            topic = Topic(
                slug=await self._unique_slug(name, parent),
                name=name,
                parent_id=parent.id if parent is not None else None,
                status=TopicStatus.ACTIVE.value,
                origin=origin.value,
            )
            self._session.add(topic)
            await self._session.flush()
            self._session.add(
                TopicAlias(
                    scope_id=scope, key=key, label=name, topic_id=topic.id, origin=origin.value
                )
            )
            await self._session.flush()
            self.created.append(topic)
        self._cache[(scope, key)] = topic
        return topic

    async def resolve_labels(self, labels: Sequence[TopicLabelLike]) -> list[ResolvedTopic]:
        """Topics and subtopics for one analysis. The most relevant topic becomes primary;
        labels that normalize to the same topic are combined (best role, max relevance)."""
        ranked = sorted(enumerate(labels), key=lambda pair: (-pair[1].relevance, pair[0]))
        links: dict[int, ResolvedTopic] = {}
        for _, label in ranked:
            topic = await self.resolve(label.name)
            if topic is None:
                continue
            has_primary = any(link.role is TopicRole.PRIMARY for link in links.values())
            role = TopicRole.SECONDARY if has_primary else TopicRole.PRIMARY
            _combine(links, ResolvedTopic(topic.id, role, label.relevance, label.name))
            for sub_label in label.subtopics:
                subtopic = await self.resolve(sub_label, topic)
                if subtopic is not None:
                    _combine(
                        links,
                        ResolvedTopic(subtopic.id, TopicRole.SUBTOPIC, label.relevance, sub_label),
                    )
        return sorted(links.values(), key=lambda link: (_ROLE_RANK[link.role], -link.relevance))

    async def active(self, topic: Topic) -> Topic:
        """Follow merges to the topic that absorbed this one."""
        for _ in range(_MAX_MERGE_DEPTH):
            if topic.merged_into_id is None:
                return topic
            topic = await self._session.get_one(Topic, topic.merged_into_id)
        raise TopicMergeError(f"Merge chain too deep at topic {topic.slug!r}")

    async def add_alias(self, topic: Topic, label: str, origin: TopicOrigin) -> str | None:
        """Make ``label`` resolve to ``topic``. Returns a conflict message if it already
        resolves to a different topic (existing mappings are never silently overridden)."""
        name = clean_label(label)
        key = label_key(name)
        if not key:
            return None
        scope = topic.parent_id if topic.parent_id is not None else TOP_LEVEL_SCOPE
        existing = await self._session.scalar(
            select(TopicAlias).where(TopicAlias.scope_id == scope, TopicAlias.key == key)
        )
        if existing is not None:
            if existing.topic_id == topic.id:
                return None
            owner = await self.active(await self._session.get_one(Topic, existing.topic_id))
            return None if owner.id == topic.id else f"{name!r} already means {owner.name!r}"
        self._session.add(
            TopicAlias(scope_id=scope, key=key, label=name, topic_id=topic.id, origin=origin.value)
        )
        await self._session.flush()
        self._cache[(scope, key)] = topic
        return None

    # ── seeds ────────────────────────────────────────────────────────────────

    async def import_seeds(self, seeds: Sequence[TopicSeed]) -> SeedImportSummary:
        summary = SeedImportSummary()
        for seed in seeds:
            before = len(self.created)
            topic = await self.resolve(seed.name, origin=TopicOrigin.SEED)
            if topic is None:
                summary.conflicts.append(f"{seed.name!r} has no usable characters")
                continue
            summary.topics_created += len(self.created) - before
            if seed.description and not topic.description:
                topic.description = seed.description
            for alias in seed.aliases:
                conflict = await self.add_alias(topic, alias, TopicOrigin.SEED)
                if conflict:
                    summary.conflicts.append(conflict)
                else:
                    summary.aliases_added += 1
            for sub_label in seed.subtopics:
                before = len(self.created)
                await self.resolve(sub_label, topic, origin=TopicOrigin.SEED)
                summary.subtopics_created += len(self.created) - before
        return summary

    # ── merges ───────────────────────────────────────────────────────────────

    async def merge(self, source: Topic, target: Topic) -> MergeSummary:
        """Fold ``source`` into ``target``: its content links, aliases and subtopics move
        over; ``source`` is kept as ``merged`` so old references still resolve."""
        source = await self.active(source)
        target = await self.active(target)
        if source.id == target.id:
            raise TopicMergeError(f"{source.name!r} and {target.name!r} are the same topic")
        if (source.parent_id is None) != (target.parent_id is None):
            raise TopicMergeError("Merge a topic into a topic, or a subtopic into a subtopic")
        summary = MergeSummary(source=source.slug, target=target.slug)
        summary.links_moved, summary.links_combined = await self._move_links(source.id, target.id)
        moved = await self._session.execute(
            update(TopicAlias).where(TopicAlias.topic_id == source.id).values(topic_id=target.id)
        )
        summary.aliases_moved = moved.rowcount  # type: ignore[attr-defined]
        for child in list(
            await self._session.scalars(
                select(Topic).where(
                    Topic.parent_id == source.id, Topic.status == TopicStatus.ACTIVE.value
                )
            )
        ):
            twin = await self._session.scalar(
                select(Topic)
                .join(TopicAlias, TopicAlias.topic_id == Topic.id)
                .where(TopicAlias.scope_id == target.id, TopicAlias.key == label_key(child.name))
            )
            if twin is not None and twin.id != child.id:
                await self.merge(child, twin)
                summary.subtopics_merged += 1
            else:
                child.parent_id = target.id
                await self._rescope_aliases(child, source.id, target.id)
                summary.subtopics_moved += 1
        source.status = TopicStatus.MERGED.value
        source.merged_into_id = target.id
        self._cache.clear()
        await self._session.flush()
        return summary

    async def _move_links(self, source_id: int, target_id: int) -> tuple[int, int]:
        """Re-point analysis links; where an analysis has both topics, keep one link with
        the stronger role and the higher relevance. Returns (moved, combined)."""
        ids = {"source": source_id, "target": target_id}
        await self._session.execute(_MERGE_SHARED_LINKS, ids)
        combined = await self._session.execute(_DELETE_SHARED_LINKS, ids)
        moved = await self._session.execute(
            update(ContentAnalysisTopic)
            .where(ContentAnalysisTopic.topic_id == source_id)
            .values(topic_id=target_id)
        )
        return int(moved.rowcount), int(combined.rowcount)  # type: ignore[attr-defined]

    async def _rescope_aliases(self, child: Topic, old_scope: int, new_scope: int) -> None:
        aliases = await self._session.scalars(
            select(TopicAlias).where(
                TopicAlias.topic_id == child.id, TopicAlias.scope_id == old_scope
            )
        )
        for alias in list(aliases):
            clash = await self._session.scalar(
                select(TopicAlias.id).where(
                    TopicAlias.scope_id == new_scope, TopicAlias.key == alias.key
                )
            )
            if clash is None:
                alias.scope_id = new_scope
        await self._session.flush()

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _unique_slug(self, name: str, parent: Topic | None) -> str:
        base = slugify(name)
        if parent is not None:
            base = f"{parent.slug}--{base}"[:190]
        taken = set(
            await self._session.scalars(
                select(Topic.slug).where((Topic.slug == base) | Topic.slug.like(f"{base}-%"))
            )
        )
        if base not in taken:
            return base
        suffix = 2
        while f"{base}-{suffix}" in taken:
            suffix += 1
        return f"{base}-{suffix}"


def _combine(links: dict[int, ResolvedTopic], link: ResolvedTopic) -> None:
    current = links.get(link.topic_id)
    if current is None:
        links[link.topic_id] = link
        return
    role = min(current.role, link.role, key=_ROLE_RANK.__getitem__)
    links[link.topic_id] = ResolvedTopic(
        link.topic_id, role, max(current.relevance, link.relevance), current.label
    )


async def prompt_taxonomy(
    session: AsyncSession, *, limit: int, subtopics_per_topic: int = 6
) -> list[TaxonomyEntry]:
    """The most-used active topics (and their most-used subtopics) to show the analyzer,
    so it reuses existing names. Deterministic order: usage, then name."""
    if limit <= 0:
        return []
    uses = func.count(ContentAnalysisTopic.analysis_id)
    top_rows = (
        await session.execute(
            select(Topic.id, Topic.name)
            .outerjoin(ContentAnalysisTopic, ContentAnalysisTopic.topic_id == Topic.id)
            .where(Topic.status == TopicStatus.ACTIVE.value, Topic.parent_id.is_(None))
            .group_by(Topic.id, Topic.name)
            .order_by(uses.desc(), Topic.name)
            .limit(limit)
        )
    ).all()
    ids = [row.id for row in top_rows]
    sub_rows = (
        await session.execute(
            select(Topic.parent_id, Topic.name)
            .outerjoin(ContentAnalysisTopic, ContentAnalysisTopic.topic_id == Topic.id)
            .where(Topic.status == TopicStatus.ACTIVE.value, Topic.parent_id.in_(ids))
            .group_by(Topic.id, Topic.parent_id, Topic.name)
            .order_by(uses.desc(), Topic.name)
        )
    ).all()
    subtopics: dict[int, list[str]] = {}
    for parent_id, name in sub_rows:
        names = subtopics.setdefault(parent_id, [])
        if len(names) < subtopics_per_topic:
            names.append(name)
    return [TaxonomyEntry(row.name, tuple(subtopics.get(row.id, ()))) for row in top_rows]
