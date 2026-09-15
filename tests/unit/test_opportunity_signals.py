"""The deterministic opportunity engine on synthetic facts (no database, no LLM)."""

from datetime import UTC, datetime, timedelta
from itertools import count

import pytest

from app.domain.analysis import ContentFormat, SearchIntent, TopicRole
from app.domain.company import CompanyProfile
from app.domain.content import ContentType
from app.domain.opportunities import GapType, ScoringConfig, ScoringWeights
from app.services.opportunity_signals import Candidate, OpportunitySignalEngine, explain_change
from app.services.trends import AnalysisFact, FactTopic, TopicInfo

NOW = datetime(2026, 9, 14, 12, tzinfo=UTC)
TOPICS = {
    1: TopicInfo(1, "ai-agents", "AI agents", None, None),
    2: TopicInfo(2, "pricing", "Pricing", None, None),
    3: TopicInfo(3, "gardening", "Gardening", None, None),
    4: TopicInfo(4, "workflow-automation", "Workflow automation", None, None),
    5: TopicInfo(5, "automating-workflows", "Automating workflows", None, None),
    10: TopicInfo(10, "ai-agents--evaluation", "Evaluation", 1, "ai-agents"),
    11: TopicInfo(11, "ai-agents--security", "Security", 1, "ai-agents"),
    12: TopicInfo(12, "ai-agents--costs", "Costs", 1, "ai-agents"),
}
_ids = count(1)


def fact(
    competitor: str,
    days_ago: float | None,
    *topics: int,
    fmt: ContentFormat = ContentFormat.ARTICLE,
    intent: SearchIntent | None = SearchIntent.INFORMATIONAL,
    audiences: tuple[str, ...] = ("developers",),
    words: int = 1500,
    keywords: tuple[str, ...] = (),
) -> AnalysisFact:
    links = tuple(
        FactTopic(
            t,
            TopicRole.SUBTOPIC
            if TOPICS[t].parent_id
            else (TopicRole.PRIMARY if i == 0 else TopicRole.SECONDARY),
            0.8,
        )
        for i, t in enumerate(topics)
    )
    return AnalysisFact(
        analysis_id=next(_ids), competitor=competitor, content_item_id=next(_ids),
        content_type=ContentType.BLOG_POST,
        published_at=NOW - timedelta(days=days_ago) if days_ago is not None else None,
        content_format=fmt, intent=intent, funnel_stage=None, audiences=audiences,
        key_themes=(), word_count=words, topics=links, keywords=keywords,
    )  # fmt: skip


def profile(**overrides: object) -> CompanyProfile:
    values: dict[str, object] = {
        "name": "Example",
        "description": "A test company.",
        "target_audiences": ["founders"],
        "core_topics": ["AI agents", "Workflow automation"],
        "excluded_topics": ["Pricing"],
    }
    values.update(overrides)
    return CompanyProfile.model_validate(values)


def history(competitor: str) -> AnalysisFact:
    """An old dated page on an unrelated topic: proves the history covers both windows."""
    return fact(competitor, 300, 3)


def engine(facts: list[AnalysisFact], config: ScoringConfig | None = None, **company: object) -> OpportunitySignalEngine:  # fmt: skip
    return OpportunitySignalEngine(facts, TOPICS, profile(**company), config or ScoringConfig(), now=NOW)  # fmt: skip


def by_key(candidates: list[Candidate]) -> dict[str, Candidate]:
    return {c.key: c for c in candidates}


def points(candidate: Candidate, dimension: str) -> float:
    return next(c.points for c in candidate.breakdown if c.dimension == dimension)


def gap(candidate: Candidate, kind: GapType) -> float:
    return next((g.score for g in candidate.gaps if g.type is kind), 0.0)


def rising_landscape() -> list[AnalysisFact]:
    facts = [history("a"), history("b"), history("c")]
    facts += [fact("a", d, 1) for d in (70, 80)]  # previous window: 2 pages
    facts += [
        fact(c, d, 1) for c in ("a", "b") for d in (5, 10, 20)
    ]  # recent: 6 pages at 2 competitors
    return facts


def test_momentum_combines_growth_and_breadth() -> None:
    agents = by_key(engine(rising_landscape()).candidates())["topic:1"]
    s = agents.signals
    assert (s["recent"], s["previous"], s["growing_competitors"]) == (6, 2, ["a", "b"])
    assert s["growth_pct"] == 200.0
    momentum = next(c for c in agents.breakdown if c.dimension == "momentum")
    # growth: 0.5 + log2(7/3)/4 = 0.8055; breadth: 2 of 3 compared competitors
    assert momentum.value == pytest.approx(0.6 * 0.8055 + 0.4 * 2 / 3, abs=1e-3)
    assert "2 competitors increased publishing on this topic (a, b)" in agents.suggestion.reasons
    assert "topic growth: +200% (2 → 6 items, 60-day windows)" in agents.suggestion.reasons


def test_momentum_is_neutral_without_trustworthy_history() -> None:
    fresh = [fact("a", d, 1) for d in (1, 2, 3)]  # everything recent: a first scan
    agents = by_key(engine(fresh).candidates())["topic:1"]
    assert not agents.signals["growth_reliable"]
    assert next(c.value for c in agents.breakdown if c.dimension == "momentum") == 0.3


def test_coverage_gap_and_recency() -> None:
    agents = by_key(engine(rising_landscape()).candidates())["topic:1"]
    assert agents.signals["competitors_covering"] == 2
    assert agents.signals["competitors_total"] == 3
    assert gap(agents, GapType.TOPIC) == pytest.approx(1 / 3, abs=1e-3)
    assert agents.signals["days_since_last"] == 5
    recency = next(c.value for c in agents.breakdown if c.dimension == "recency")
    assert recency == pytest.approx(0.5 ** (5 / 30), abs=1e-3)


def test_saturation_counts_volume_breadth_frequency_and_variety() -> None:
    crowded = [history(c) for c in "abc"]
    formats = [ContentFormat.ARTICLE, ContentFormat.GUIDE, ContentFormat.TUTORIAL, ContentFormat.COMPARISON, ContentFormat.CASE_STUDY]  # fmt: skip
    crowded += [fact(c, d, 1, fmt=formats[d % 5], audiences=("founders",), intent=SearchIntent.COMMERCIAL) for c in "abc" for d in range(1, 31, 2)]  # fmt: skip
    light = [history(c) for c in "abc"] + [fact("a", 10, 1, audiences=("founders",), intent=SearchIntent.COMMERCIAL), fact("a", 20, 1, audiences=("founders",), intent=SearchIntent.COMMERCIAL)]  # fmt: skip
    heavy = by_key(engine(crowded).candidates())["topic:1"]
    sparse = by_key(engine(light).candidates())["topic:1"]
    assert heavy.signals["saturation"]["raw"] > 0.8
    assert sparse.signals["saturation"]["raw"] < 0.4
    assert points(heavy, "saturation") < points(sparse, "saturation") <= 0


def test_weak_coverage_relieves_saturation() -> None:
    """A crowded topic whose pages are old still leaves room: saturation is relieved."""
    base = [history(c) for c in "abc"]
    served = {
        "audiences": ("founders",),
        "intent": SearchIntent.COMMERCIAL,
    }  # only freshness differs
    fresh = base + [fact(c, d, 1, **served) for c in "abc" for d in (5, 15, 25, 35)]  # type: ignore[arg-type]
    stale = base + [fact(c, d, 1, **served) for c in "abc" for d in (300, 330, 360, 400)]  # type: ignore[arg-type]
    new = by_key(engine(fresh).candidates())["topic:1"].signals["saturation"]
    old = by_key(engine(stale).candidates())["topic:1"].signals["saturation"]
    assert old["relief"] > new["relief"]
    assert old["effective"] < old["raw"]


def test_strategic_relevance_gates_candidates() -> None:
    facts = rising_landscape() + [fact("a", d, 3) for d in (1, 2, 3)] + [fact("b", d, 2) for d in (1, 2, 3)]  # fmt: skip
    qualified, rejected = engine(facts).opportunities()
    keys = {c.key for c in qualified}
    reasons = {c.key: c.rejected for c in rejected}
    assert "topic:1" in keys  # a core topic
    assert "strategic fit 0.0 is below the minimum" in (reasons["topic:3"] or "")  # gardening
    assert reasons["topic:2"] == "excluded by your company profile ('Pricing')"


def test_audience_fit_and_gap() -> None:
    base = [history(c) for c in "ab"]
    served = base + [fact(c, d, 1, audiences=("SaaS founders",)) for c in "ab" for d in (5, 10, 15)]
    unserved = base + [fact(c, d, 1, audiences=("developers",)) for c in "ab" for d in (5, 10, 15)]
    ok = by_key(engine(served).candidates())["topic:1"]
    missing = by_key(engine(unserved).candidates())["topic:1"]
    assert next(c.value for c in ok.breakdown if c.dimension == "audience_fit") == 1.0
    assert gap(ok, GapType.AUDIENCE) == 0.0
    assert next(c.value for c in missing.breakdown if c.dimension == "audience_fit") == 0.0
    assert gap(missing, GapType.AUDIENCE) == 1.0
    assert missing.suggestion.audience == "founders"


def test_intent_and_format_gaps() -> None:
    facts = [history(c) for c in "ab"] + [fact(c, d, 1, fmt=ContentFormat.ARTICLE, intent=SearchIntent.INFORMATIONAL) for c in "ab" for d in (5, 10, 15)]  # fmt: skip
    agents = by_key(engine(facts).candidates())["topic:1"]
    assert gap(agents, GapType.INTENT) == 1.0  # no commercial or comparison pages at all
    assert gap(agents, GapType.FORMAT) == 1.0
    assert agents.suggestion.intent in (SearchIntent.COMMERCIAL, SearchIntent.COMPARISON)
    assert agents.suggestion.format is not None
    with_comparisons = [history(c) for c in "ab"] + [fact(c, d, 1, fmt=ContentFormat.COMPARISON, intent=SearchIntent.COMPARISON) for c in "ab" for d in (5, 10, 15)]  # fmt: skip
    served = by_key(engine(with_comparisons).candidates())["topic:1"]
    assert gap(served, GapType.INTENT) == 0.0  # the valuable intents are covered, together
    assert gap(served, GapType.FORMAT) == 0.0
    mixed = [history(c) for c in "ab"] + [fact(c, d, 1, intent=SearchIntent.INFORMATIONAL if d != 5 else SearchIntent.COMMERCIAL) for c in "ab" for d in (5, 10, 15, 20, 25)]  # fmt: skip
    partial = by_key(engine(mixed).candidates())["topic:1"]
    assert gap(partial, GapType.INTENT) == 0.0  # 20% commercial meets the expected share
    assert partial.gaps[[g.type for g in partial.gaps].index(GapType.INTENT)].data["underserved"][0] == "comparison"  # fmt: skip


def test_company_preferred_formats_define_the_format_gap() -> None:
    facts = [history(c) for c in "ab"] + [fact(c, d, 1, fmt=ContentFormat.TUTORIAL) for c in "ab" for d in (5, 10, 15)]  # fmt: skip
    default = by_key(engine(facts).candidates())["topic:1"]
    wants_research = by_key(engine(facts, preferred_formats=["research"]).candidates())["topic:1"]
    assert wants_research.suggestion.format == ContentFormat.RESEARCH
    assert gap(wants_research, GapType.FORMAT) == 1.0
    assert gap(default, GapType.FORMAT) >= 0.0


def test_freshness_gap() -> None:
    old = [history(c) for c in "ab"] + [fact(c, d, 1) for c in "ab" for d in (400, 420, 450)]
    agents = by_key(engine(old).candidates())["topic:1"]
    assert gap(agents, GapType.FRESHNESS) == 1.0
    assert "median competitor page is" in next(g.detail for g in agents.gaps if g.type is GapType.FRESHNESS)  # fmt: skip


def test_depth_gap_from_fragmented_or_shallow_coverage() -> None:
    fragmented = [history(c) for c in "ab"] + [fact("a", 5, 1, 10), fact("b", 6, 1, 11), fact("a", 7, 1, 12), fact("b", 8, 1)]  # fmt: skip
    shallow = [history(c) for c in "ab"] + [fact(c, d, 1, words=250) for c in "ab" for d in (5, 10)]
    assert gap(by_key(engine(fragmented).candidates())["topic:1"], GapType.DEPTH) == 1.0
    assert gap(by_key(engine(shallow).candidates())["topic:1"], GapType.DEPTH) > 0.9


def test_differentiation_gap_when_everyone_writes_the_same_thing() -> None:
    same = [history(c) for c in "ab"] + [fact(c, d, 1, fmt=ContentFormat.LISTICLE) for c in "ab" for d in (5, 10)]  # fmt: skip
    assert gap(by_key(engine(same).candidates())["topic:1"], GapType.DIFFERENTIATION) == 1.0


def test_gaps_from_few_pages_are_scaled_down() -> None:
    two = [history(c) for c in "ab"] + [fact("a", 5, 1), fact("b", 5, 1)]
    agents = by_key(engine(two).candidates())["topic:1"]
    assert gap(agents, GapType.INTENT) == 0.5  # 2 of the 4 pages needed for full strength


def test_score_is_normalized_and_decomposable() -> None:
    for candidate in engine(rising_landscape()).candidates():
        assert 0 <= candidate.score <= 100
        assert candidate.score == pytest.approx(max(0.0, sum(c.points for c in candidate.breakdown)), abs=0.1)  # fmt: skip
        positive = [c for c in candidate.breakdown if c.dimension != "saturation"]
        assert sum(c.max_points for c in positive) == pytest.approx(100)


def test_weights_are_configurable_and_rescaled() -> None:
    config = ScoringConfig(weights=ScoringWeights(momentum=1, strategic_fit=1, audience_fit=0, content_gap=0, recency=0, saturation=0))  # fmt: skip
    agents = by_key(engine(rising_landscape(), config).candidates())["topic:1"]
    maxima = {c.dimension: c.max_points for c in agents.breakdown}
    assert maxima["momentum"] == maxima["strategic_fit"] == 50
    assert maxima["recency"] == 0
    with pytest.raises(ValueError, match="positive scoring weight"):
        engine(rising_landscape(), ScoringConfig(weights=ScoringWeights(momentum=0, strategic_fit=0, audience_fit=0, content_gap=0, recency=0)))  # fmt: skip


def test_core_topics_nobody_covers_are_topic_gaps_once_the_corpus_is_big_enough() -> None:
    small = [fact("a", d, 1) for d in range(1, 6)]
    assert "core:workflow automation" not in by_key(engine(small).candidates())
    corpus = [fact(c, d, 1) for c in "ab" for d in range(1, 16)]
    gaps = by_key(engine(corpus).candidates())
    workflow = gaps["core:workflow automation"]
    assert workflow.topic_id is None
    assert gap(workflow, GapType.TOPIC) == 1.0
    assert workflow.signals["corpus_items"] == 30
    assert "none of the 2 competitors covers it" in workflow.gaps[0].detail


def test_near_duplicate_topics_become_one_opportunity() -> None:
    facts = [history(c) for c in "ab"]
    facts += [fact(c, d, 4, audiences=("founders",)) for c in "ab" for d in (5, 10, 15)]
    facts += [fact("a", d, 5, audiences=("founders",)) for d in (6, 12)]
    qualified, rejected = engine(facts).opportunities()
    keys = [c.key for c in qualified]
    assert keys.count("topic:4") + keys.count("topic:5") == 1
    kept = next(c for c in qualified if c.key in ("topic:4", "topic:5"))
    duplicate = next(c for c in rejected if c.key in ("topic:4", "topic:5"))
    assert duplicate.rejected == f"near-duplicate of '{kept.label}'"
    assert kept.signals["related_topics"] == [duplicate.label]


def test_same_inputs_give_identical_scores() -> None:
    first = [(c.key, c.score, [p.points for p in c.breakdown]) for c in engine(rising_landscape()).candidates()]  # fmt: skip
    second = [(c.key, c.score, [p.points for p in c.breakdown]) for c in engine(rising_landscape()).candidates()]  # fmt: skip
    assert first == second


def test_score_changes_are_explained() -> None:
    before = by_key(engine([history(c) for c in "abc"] + [fact("a", 70, 1), fact("a", 80, 1), fact("a", 10, 1)]).candidates())["topic:1"]  # fmt: skip
    after = by_key(engine(rising_landscape()).candidates())["topic:1"]
    previous = {"score": before.score, "breakdown": [c.model_dump() for c in before.breakdown], "signals": before.signals}  # fmt: skip
    basis = {"scoring": "s1", "company_scoring": "c1", "window_days": 60, "evidence_ids": [1, 2, 3]}  # fmt: skip
    change = explain_change(previous, after, old_basis=basis, new_basis={**basis, "evidence_ids": [2, 3, 4, 5]})  # fmt: skip
    assert change.delta == pytest.approx(after.score - before.score, abs=0.1)
    assert change.dimensions["momentum"] > 0
    assert any(r.startswith("momentum +") and "recent items 1 → 6" in r for r in change.reasons)
    assert "analysed pages: 2 added, 1 dropped" in change.reasons
    moved = explain_change(previous, after, old_basis=basis, new_basis={**basis, "scoring": "s2", "company_scoring": "c2", "window_days": 90})  # fmt: skip
    assert {"scoring configuration changed", "window 60 → 90 days"} <= set(moved.reasons)
    assert any("company profile changed" in r for r in moved.reasons)
    tone_only = explain_change(previous, after, old_basis={**basis, "company": "p1"}, new_basis={**basis, "company": "p2"})  # fmt: skip
    assert "company profile changed (fields that don't affect scores)" in tone_only.reasons
    unknown = explain_change(previous, after, old_basis=basis, new_basis={**basis, "company": "p2"})  # fmt: skip
    assert not any("company profile" in r for r in unknown.reasons)  # an older basis without it


def test_unchanged_rescore_says_why_it_exists() -> None:
    candidate = by_key(engine(rising_landscape()).candidates())["topic:1"]
    previous = {"score": candidate.score, "breakdown": [c.model_dump() for c in candidate.breakdown], "signals": candidate.signals}  # fmt: skip
    basis = {"scoring": "s1", "company_scoring": "c1", "window_days": 60, "evidence_ids": [1]}
    assert explain_change(previous, candidate, old_basis=basis, new_basis=basis).reasons == ["no material change"]  # fmt: skip
    forced = explain_change(previous, candidate, old_basis=basis, new_basis=basis, forced=True)
    assert forced.reasons == ["recalculated on request; inputs unchanged"]
    assert forced.delta == 0
