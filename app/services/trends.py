"""Deterministic trend and coverage metrics over stored analyses. No LLM.

Time basis: **reliable publication dates only** (the Phase 2 date rule). Items without one
count in distributions ("what they publish") but never in time comparisons, and
first-seen dates are never used as publication dates.

Window comparisons are only meaningful where the captured history reaches back to the
start of the previous window. Scans capture newest content first, so a competitor whose
oldest dated analyzed item is more recent than that would look artificially "rising".
Such competitors are listed in ``TrendBasis.insufficient_history`` and left out of growth
comparisons (they still count everywhere else).
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from app.domain.analysis import (
    Cadence,
    ContentFormat,
    FunnelStage,
    MixShift,
    NeglectedTopic,
    SearchIntent,
    Share,
    TopicRef,
    TopicRole,
    TopicTrend,
    TrendBasis,
    TrendDirection,
)
from app.domain.content import ContentType
from app.services.labels import label_key

MIN_TREND_ITEMS = 2  # fewer dated items than this is noise, not a trend
RISING_RATIO = 1.5  # (recent + 1) / (previous + 1) at or above this is rising
MIN_SHIFT_ITEMS = 4  # items needed in each window before reporting a mix shift
MIN_SHIFT_CHANGE = 0.10  # 10 percentage points
DORMANT_MIN_ITEMS = 3

Dimension = Literal["format", "audience", "intent", "funnel_stage"]


@dataclass(frozen=True)
class FactTopic:
    topic_id: int
    role: TopicRole
    relevance: float


@dataclass(frozen=True)
class AnalysisFact:
    """The latest substantive analysis of one active content item."""

    analysis_id: int
    competitor: str
    content_item_id: int
    content_type: ContentType
    published_at: datetime | None  # reliable publication date, else None
    content_format: ContentFormat
    intent: SearchIntent | None
    funnel_stage: FunnelStage | None
    audiences: tuple[str, ...]
    key_themes: tuple[str, ...]
    word_count: int
    topics: tuple[FactTopic, ...]
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class TopicInfo:
    id: int
    slug: str
    name: str
    parent_id: int | None
    parent_slug: str | None

    def ref(self) -> TopicRef:
        return TopicRef(slug=self.slug, name=self.name, parent=self.parent_slug)


def classify_trend(
    recent: int, previous: int, *, before_window: int, undated: int, reliable: bool
) -> TrendDirection:
    """``before_window``: dated items older than the previous window."""
    if not reliable:
        return TrendDirection.INSUFFICIENT_HISTORY
    if recent == 0 and previous == 0:
        return TrendDirection.DORMANT
    if previous == 0 and before_window == 0 and undated == 0 and recent >= MIN_TREND_ITEMS:
        return TrendDirection.NEW
    if recent >= MIN_TREND_ITEMS and (recent + 1) / (previous + 1) >= RISING_RATIO:
        return TrendDirection.RISING
    if previous >= MIN_TREND_ITEMS and recent <= previous / 2:
        return TrendDirection.DECLINING
    return TrendDirection.STEADY


def _shares(counter: Counter[str], total: int, labels: Mapping[str, str] | None = None) -> list[Share]:  # fmt: skip
    rows = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        Share(
            value=(labels or {}).get(key, key),
            count=count,
            share=round(count / total, 4) if total else 0.0,
        )
        for key, count in rows
    ]


class TrendEngine:
    def __init__(
        self,
        facts: Sequence[AnalysisFact],
        topics: Mapping[int, TopicInfo],
        *,
        now: datetime,
        window_days: int,
    ) -> None:
        self._facts = list(facts)
        self._topics = topics
        self._slug_ids = {info.slug: info.id for info in topics.values()}
        self.now = now
        self.window_days = window_days
        self.window_start = now - timedelta(days=window_days)
        self.previous_start = now - timedelta(days=2 * window_days)
        oldest: dict[str, datetime] = {}
        for fact in self._facts:
            if fact.published_at is not None:
                current = oldest.get(fact.competitor)
                oldest[fact.competitor] = min(current, fact.published_at) if current else fact.published_at  # fmt: skip
        competitors = {fact.competitor for fact in self._facts}
        self.compared = sorted(c for c in competitors if c in oldest and oldest[c] <= self.previous_start)  # fmt: skip
        self.insufficient = sorted(competitors - set(self.compared))

    # ── scopes ───────────────────────────────────────────────────────────────

    def basis(self) -> TrendBasis:
        return TrendBasis(
            window_days=self.window_days,
            window_start=self.window_start,
            previous_window_start=self.previous_start,
            now=self.now,
            compared_competitors=self.compared,
            insufficient_history=self.insufficient,
        )

    def _scope(self, competitor: str | None) -> list[AnalysisFact]:
        return [f for f in self._facts if competitor is None or f.competitor == competitor]

    def bucket(self, fact: AnalysisFact) -> Literal["recent", "previous", "before"] | None:
        if fact.published_at is None or fact.published_at > self.now:
            return None
        if fact.published_at > self.window_start:
            return "recent"
        if fact.published_at > self.previous_start:
            return "previous"
        return "before"

    def _comparison(self, scope: Sequence[AnalysisFact]) -> tuple[list[AnalysisFact], bool]:
        """Facts usable for window comparisons, and whether any are reliable."""
        compared = [f for f in scope if f.competitor in self.compared]
        return (compared, True) if compared else (list(scope), False)

    # ── topics ───────────────────────────────────────────────────────────────

    def topic_trends(
        self,
        *,
        competitor: str | None = None,
        subtopics: bool = False,
        parent_id: int | None = None,
        topic_ids: Iterable[int] | None = None,
    ) -> list[TopicTrend]:
        scope = self._scope(competitor)
        comparison, reliable = self._comparison(scope)
        comparison_ids = {f.analysis_id for f in comparison}
        wanted = set(topic_ids) if topic_ids is not None else None
        by_topic: dict[int, list[tuple[AnalysisFact, FactTopic]]] = defaultdict(list)
        for fact in scope:
            for link in fact.topics:
                info = self._topics.get(link.topic_id)
                if info is None or (info.parent_id is not None) != subtopics:
                    continue
                if parent_id is not None and info.parent_id != parent_id:
                    continue
                if wanted is not None and link.topic_id not in wanted:
                    continue
                by_topic[link.topic_id].append((fact, link))
        total = len(scope)
        trends = []
        for topic_id, rows in by_topic.items():
            per_competitor = Counter(fact.competitor for fact, _ in rows)
            buckets = Counter(self.bucket(fact) for fact, _ in rows if fact.analysis_id in comparison_ids)  # fmt: skip
            dated = [fact.published_at for fact, _ in rows if fact.published_at is not None]
            trends.append(
                TopicTrend(
                    topic=self._topics[topic_id].ref(),
                    items=len(rows),
                    share=round(len(rows) / total, 4) if total else 0.0,
                    primary_items=sum(1 for _, link in rows if link.role is TopicRole.PRIMARY),
                    recent=buckets["recent"],
                    previous=buckets["previous"],
                    trend=classify_trend(
                        buckets["recent"],
                        buckets["previous"],
                        before_window=buckets["before"],
                        undated=buckets[None],
                        reliable=reliable,
                    ),
                    last_published_at=max(dated) if dated else None,
                    competitors=len(per_competitor),
                    by_competitor=dict(sorted(per_competitor.items())),
                )
            )
        return sorted(trends, key=lambda t: (-t.items, -t.recent, t.topic.name.casefold()))

    def rising(self, trends: Sequence[TopicTrend]) -> list[TopicTrend]:
        """New or rising topics; those growing at more competitors first."""
        candidates = [t for t in trends if t.trend in (TrendDirection.NEW, TrendDirection.RISING)]
        momentum = {t.topic.slug: self._growing_competitors(t) for t in candidates}
        return sorted(candidates, key=lambda t: (-momentum[t.topic.slug], -t.recent, -t.items))

    def _growing_competitors(self, trend: TopicTrend) -> int:
        topic_id = self._slug_ids[trend.topic.slug]
        subtopic = trend.topic.parent is not None
        count = 0
        for competitor in self.compared:
            rows = self.topic_trends(
                competitor=competitor, subtopics=subtopic, topic_ids=[topic_id]
            )
            if rows and rows[0].recent > rows[0].previous:
                count += 1
        return count

    def neglected(self, trends: Sequence[TopicTrend], *, limit: int = 25) -> list[NeglectedTopic]:
        """Topics the landscape covers little, narrowly, or no longer. Descriptive only:
        whether any of them is an opportunity is Phase 4's question."""
        competitors_total = len({f.competitor for f in self._facts})
        found: list[NeglectedTopic] = []

        def flag(t: TopicTrend, reason: str, detail: str) -> None:
            found.append(
                NeglectedTopic.model_validate(
                    {
                        "topic": t.topic,
                        "reason": reason,
                        "detail": detail,
                        "items": t.items,
                        "competitors": sorted(t.by_competitor),
                        "last_published_at": t.last_published_at,
                    }
                )
            )

        for t in trends:
            if t.trend is TrendDirection.DORMANT and t.items >= DORMANT_MIN_ITEMS:
                last = f"{t.last_published_at:%Y-%m-%d}" if t.last_published_at else "unknown"
                flag(t, "dormant", f"{t.items} items, none published in the last {2 * self.window_days} days (last: {last})")  # fmt: skip
            elif t.trend is TrendDirection.DECLINING:
                flag(t, "declining", f"{t.previous} items in the previous {self.window_days} days, {t.recent} in the last {self.window_days}")  # fmt: skip
            elif competitors_total >= 2 and t.competitors == 1 and t.items >= MIN_TREND_ITEMS:
                only = next(iter(t.by_competitor))
                flag(t, "single_competitor", f"only {only} covers this ({t.items} items); the other {competitors_total - 1} competitor(s) don't")  # fmt: skip
        if competitors_total >= 2:
            for parent in [t for t in trends if t.competitors >= 2][:10]:
                parent_id = self._slug_ids[parent.topic.slug]
                for sub in self.topic_trends(subtopics=True, parent_id=parent_id):
                    if sub.competitors == 1 and sub.items >= MIN_TREND_ITEMS:
                        only = next(iter(sub.by_competitor))
                        flag(sub, "thin_subtopic", f"{parent.competitors} competitors cover {parent.topic.name!r}; only {only} covers this angle")  # fmt: skip
        order = {"dormant": 0, "declining": 1, "single_competitor": 2, "thin_subtopic": 3}
        return sorted(found, key=lambda n: (order[n.reason], -n.items, n.topic.slug))[:limit]

    # ── mixes ────────────────────────────────────────────────────────────────

    def _values(self, dimension: Dimension) -> Callable[[AnalysisFact], list[str]]:
        def values(fact: AnalysisFact) -> list[str]:
            if dimension == "format":
                return [fact.content_format.value]
            if dimension == "intent":
                return [fact.intent.value] if fact.intent else []
            if dimension == "funnel_stage":
                return [fact.funnel_stage.value] if fact.funnel_stage else []
            return list(dict.fromkeys(label_key(a) for a in fact.audiences if label_key(a)))

        return values

    def _audience_labels(self) -> dict[str, str]:
        spellings: dict[str, Counter[str]] = defaultdict(Counter)
        for fact in self._facts:
            for audience in fact.audiences:
                spellings[label_key(audience)][audience] += 1
        return {key: counter.most_common(1)[0][0] for key, counter in spellings.items()}

    def mix(self, dimension: Dimension, *, competitor: str | None = None, facts: Sequence[AnalysisFact] | None = None) -> list[Share]:  # fmt: skip
        scope = list(facts) if facts is not None else self._scope(competitor)
        values = self._values(dimension)
        counter: Counter[str] = Counter(v for fact in scope for v in values(fact))
        labels = self._audience_labels() if dimension == "audience" else None
        return _shares(counter, len(scope), labels)

    def mix_shifts(self, *, competitor: str | None = None) -> list[MixShift]:
        comparison, reliable = self._comparison(self._scope(competitor))
        if not reliable:
            return []
        recent = [f for f in comparison if self.bucket(f) == "recent"]
        previous = [f for f in comparison if self.bucket(f) == "previous"]
        if len(recent) < MIN_SHIFT_ITEMS or len(previous) < MIN_SHIFT_ITEMS:
            return []
        shifts = []
        dimensions: tuple[Dimension, ...] = ("format", "audience", "intent", "funnel_stage")
        for dimension in dimensions:
            now_mix = {s.value: s.share for s in self.mix(dimension, facts=recent)}
            before_mix = {s.value: s.share for s in self.mix(dimension, facts=previous)}
            for value in sorted(set(now_mix) | set(before_mix)):
                change = now_mix.get(value, 0.0) - before_mix.get(value, 0.0)
                if abs(change) >= MIN_SHIFT_CHANGE:
                    shifts.append(
                        MixShift(
                            dimension=dimension,
                            value=value,
                            previous_share=before_mix.get(value, 0.0),
                            recent_share=now_mix.get(value, 0.0),
                            change=round(change * 100, 1),
                        )
                    )
        return sorted(shifts, key=lambda s: (-abs(s.change), s.dimension, s.value))[:10]

    def themes(self, *, competitor: str, types: Iterable[ContentType] | None = None, limit: int = 10) -> list[Share]:  # fmt: skip
        allowed = set(types) if types is not None else None
        scope = [f for f in self._scope(competitor) if allowed is None or f.content_type in allowed]
        counter: Counter[str] = Counter()
        spellings: dict[str, Counter[str]] = defaultdict(Counter)
        for fact in scope:
            for theme in dict.fromkeys(fact.key_themes):
                key = label_key(theme)
                if key:
                    counter[key] += 1
                    spellings[key][theme] += 1
        labels = {key: c.most_common(1)[0][0] for key, c in spellings.items()}
        return [s for s in _shares(counter, len(scope), labels) if s.count >= 2][:limit]

    def cadence(self, *, competitor: str | None = None) -> Cadence:
        scope = self._scope(competitor)
        buckets = Counter(self.bucket(f) for f in scope)
        return Cadence(
            window_days=self.window_days,
            recent=buckets["recent"],
            previous=buckets["previous"],
            per_week=round(buckets["recent"] / (self.window_days / 7), 2),
            undated=buckets[None],
        )

    def analyzed_items(self, competitor: str | None = None) -> int:
        return len(self._scope(competitor))
