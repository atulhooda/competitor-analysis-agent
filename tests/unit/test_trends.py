from datetime import UTC, datetime, timedelta

import pytest

from app.domain.analysis import ContentFormat, SearchIntent, TopicRole, TrendDirection
from app.domain.content import ContentType
from app.services.trends import (
    AnalysisFact,
    FactTopic,
    TopicInfo,
    TrendEngine,
    classify_trend,
)

NOW = datetime(2026, 9, 13, 12, tzinfo=UTC)
TOPICS = {
    1: TopicInfo(1, "ai-agents", "AI agents", None, None),
    2: TopicInfo(2, "pricing", "Pricing", None, None),
    3: TopicInfo(3, "data-privacy", "Data privacy", None, None),
    4: TopicInfo(4, "ai-agents--evaluation", "Evaluation", 1, "ai-agents"),
    5: TopicInfo(5, "ai-agents--pricing-models", "Pricing models", 1, "ai-agents"),
    6: TopicInfo(6, "webinars", "Webinars", None, None),
}
_ids = iter(range(1, 10_000))


def fact(
    competitor: str,
    days_ago: float | None,
    *topics: int,
    fmt: ContentFormat = ContentFormat.ARTICLE,
    audiences: tuple[str, ...] = ("Developers",),
    intent: SearchIntent | None = SearchIntent.INFORMATIONAL,
) -> AnalysisFact:
    links = tuple(
        FactTopic(t, TopicRole.PRIMARY if i == 0 else TopicRole.SECONDARY, 0.8)
        for i, t in enumerate(topics)
    )
    subtopic_links = tuple(FactTopic(t, TopicRole.SUBTOPIC, 0.5) for t in topics if TOPICS[t].parent_id)  # fmt: skip
    return AnalysisFact(
        analysis_id=next(_ids),
        competitor=competitor,
        content_item_id=next(_ids),
        content_type=ContentType.BLOG_POST,
        published_at=NOW - timedelta(days=days_ago) if days_ago is not None else None,
        content_format=fmt,
        intent=intent,
        funnel_stage=None,
        audiences=audiences,
        key_themes=("Automation",),
        word_count=900,
        topics=tuple(link for link in links if not TOPICS[link.topic_id].parent_id)
        + subtopic_links,
    )


def history(competitor: str) -> AnalysisFact:
    """An old dated item: proves the competitor's captured history covers both windows."""
    return fact(competitor, 200, 6)


@pytest.mark.parametrize(
    ("recent", "previous", "before", "undated", "reliable", "expected"),
    [
        (0, 0, 3, 0, True, TrendDirection.DORMANT),
        (3, 0, 0, 0, True, TrendDirection.NEW),
        (3, 0, 2, 0, True, TrendDirection.RISING),  # existed before: not new
        (3, 0, 0, 1, True, TrendDirection.RISING),  # undated items existed: not provably new
        (1, 0, 0, 0, True, TrendDirection.STEADY),  # one item is not a trend
        (4, 3, 0, 0, True, TrendDirection.STEADY),
        (5, 3, 0, 0, True, TrendDirection.RISING),  # (5+1)/(3+1) = 1.5
        (6, 3, 0, 0, True, TrendDirection.RISING),
        (1, 4, 0, 0, True, TrendDirection.DECLINING),
        (9, 0, 0, 0, False, TrendDirection.INSUFFICIENT_HISTORY),
    ],
)
def test_classify_trend(
    recent: int, previous: int, before: int, undated: int, reliable: bool, expected: TrendDirection
) -> None:
    assert classify_trend(recent, previous, before_window=before, undated=undated, reliable=reliable) == expected  # fmt: skip


def test_topic_counts_shares_and_trends() -> None:
    facts = [
        history("acme"),
        fact("acme", 2, 1, 4),
        fact("acme", 5, 1, 5),
        fact("acme", 10, 1, 2),
        fact("acme", 45, 2),
        fact("acme", 50, 2),
        fact("acme", None, 2),  # undated: counts in distributions, never in windows
    ]
    engine = TrendEngine(facts, TOPICS, now=NOW, window_days=30)
    trends = {t.topic.slug: t for t in engine.topic_trends(competitor="acme")}
    agents, pricing = trends["ai-agents"], trends["pricing"]
    assert (agents.items, agents.recent, agents.previous, agents.primary_items) == (3, 3, 0, 3)
    assert agents.trend is TrendDirection.NEW
    assert (pricing.items, pricing.recent, pricing.previous) == (4, 1, 2)
    assert pricing.trend is TrendDirection.DECLINING
    assert agents.share == round(3 / 7, 4)
    assert engine.basis().compared_competitors == ["acme"]
    sub = {t.topic.slug for t in engine.topic_trends(competitor="acme", subtopics=True)}
    assert sub == {"ai-agents--evaluation", "ai-agents--pricing-models"}
    assert engine.cadence(competitor="acme").model_dump() == {
        "window_days": 30, "recent": 3, "previous": 2, "per_week": 0.7, "undated": 1,
    }  # fmt: skip


def test_short_history_is_not_reported_as_growth() -> None:
    # A first scan captured only recent posts: everything looks "new" but isn't provably so.
    facts = [fact("fresh", 1, 1), fact("fresh", 3, 1), fact("fresh", 8, 1)]
    engine = TrendEngine(facts, TOPICS, now=NOW, window_days=30)
    [agents] = engine.topic_trends()
    assert agents.trend is TrendDirection.INSUFFICIENT_HISTORY
    assert engine.basis().insufficient_history == ["fresh"]
    assert engine.mix_shifts() == []


def test_growth_comparisons_exclude_competitors_without_history() -> None:
    facts = [history("acme"), fact("acme", 40, 1), fact("acme", 45, 1)]
    facts += [fact("fresh", 1, 1), fact("fresh", 2, 1), fact("fresh", 3, 1)]
    engine = TrendEngine(facts, TOPICS, now=NOW, window_days=30)
    [agents] = [t for t in engine.topic_trends() if t.topic.slug == "ai-agents"]
    assert agents.by_competitor == {"acme": 2, "fresh": 3}  # everyone counts in coverage
    assert (agents.recent, agents.previous) == (0, 2)  # only acme is compared
    assert agents.trend is TrendDirection.DECLINING


def test_rising_orders_by_competitors_growing() -> None:
    facts = [history("a"), history("b")]
    facts += [fact("a", d, 1) for d in (1, 2, 3)] + [fact("b", d, 1) for d in (4, 5)]
    facts += [fact("a", d, 2) for d in (1, 2, 3, 4, 5)]
    engine = TrendEngine(facts, TOPICS, now=NOW, window_days=30)
    rising = engine.rising(engine.topic_trends())
    assert [t.topic.slug for t in rising][:2] == ["ai-agents", "pricing"]  # 2 competitors beat 1


def test_neglected_topics() -> None:
    facts = [history("a"), history("b")]
    facts += [fact("a", d, 3) for d in (100, 120, 150)]  # dormant: nothing in 60 days
    facts += [fact("a", d, 2) for d in (40, 45, 50)] + [fact("a", 1, 2)]  # declining
    facts += [
        fact("b", 3, 1, 4),
        fact("a", 4, 1),
        fact("b", 9, 1, 4),
    ]  # thin subtopic under shared topic
    engine = TrendEngine(facts, TOPICS, now=NOW, window_days=30)
    neglected = {(n.topic.slug, n.reason) for n in engine.neglected(engine.topic_trends())}
    assert ("data-privacy", "dormant") in neglected
    assert ("pricing", "declining") in neglected
    assert ("ai-agents--evaluation", "thin_subtopic") in neglected
    assert ("webinars", "single_competitor") not in neglected  # both competitors cover it


def test_single_competitor_topics_need_two_competitors_in_the_data() -> None:
    solo = TrendEngine([history("a"), fact("a", 1, 2), fact("a", 2, 2)], TOPICS, now=NOW, window_days=30)  # fmt: skip
    assert not any(n.reason == "single_competitor" for n in solo.neglected(solo.topic_trends()))
    pair = TrendEngine([history("a"), history("b"), fact("a", 1, 2), fact("a", 2, 2)], TOPICS, now=NOW, window_days=30)  # fmt: skip
    assert ("pricing", "single_competitor") in {(n.topic.slug, n.reason) for n in pair.neglected(pair.topic_trends())}  # fmt: skip


def test_mixes_normalize_audience_spellings() -> None:
    facts = [
        history("a"),
        fact("a", 1, 1, audiences=("Developers", "Founders")),
        fact("a", 2, 1, audiences=("developer",)),
        fact("a", 3, 1, fmt=ContentFormat.TUTORIAL, audiences=()),
    ]
    engine = TrendEngine(facts, TOPICS, now=NOW, window_days=30)
    audiences = {s.value: s.count for s in engine.mix("audience")}
    assert audiences["Developers"] == 3  # history fact + "Developers" + "developer"
    assert audiences["Founders"] == 1
    formats = engine.mix("format")
    assert formats[0].value == "article"
    assert formats[0].count == 3


def test_mix_shift_between_windows() -> None:
    facts = [history("a")]
    facts += [fact("a", d, 1, fmt=ContentFormat.ARTICLE) for d in (35, 40, 45, 50)]
    facts += [fact("a", d, 1, fmt=ContentFormat.TUTORIAL) for d in (1, 2, 3)] + [fact("a", 4, 1)]
    engine = TrendEngine(facts, TOPICS, now=NOW, window_days=30)
    shifts = {(s.dimension, s.value): s.change for s in engine.mix_shifts()}
    assert shifts[("format", "tutorial")] == 75.0
    assert shifts[("format", "article")] == -75.0
