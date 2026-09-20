"""The opportunity signal engine: deterministic signals → gaps → transparent score. No LLM.

Input: the latest analysis of every competitor page (``AnalysisFact``), the topic taxonomy,
your company profile and the scoring configuration. Output: one scored ``Candidate`` per
canonical top-level topic (plus core company topics no competitor covers), each with a
decomposable score, its gaps, a deterministic suggestion, and the pages it rests on.

Every number here is computed, never estimated by a model, and the same inputs always
give the same scores.

Dimensions (each a 0-1 value; positive weights are rescaled to total 100 points):

- momentum: smoothed growth of reliably dated items between the previous and the current
  window (60%) and the share of compared competitors growing on the topic (40%)
- strategic_fit: relevance to your company profile (core/adjacent topics, subtopics,
  description, keyword support); below ``min_strategic_fit`` a topic is never an opportunity
- audience_fit: whether your audiences read about the topic (their share of its pages)
- content_gap: the strongest weighted gap (topic, audience, intent, format, depth,
  freshness, differentiation)
- recency: how recently competitors published on it (half-life decay)
- saturation (subtracted): volume, competitor breadth, frequency and format variety,
  relieved when the existing coverage is weak (stale, shallow, fragmented, or missing
  your audience or valuable intents)
"""

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.domain.analysis import (  # fmt: skip
    ContentFormat,
    SearchIntent,
    TopicRef,
    TopicRole,
    TopicTrend,
    TrendDirection,
)
from app.domain.company import CompanyProfile
from app.domain.opportunities import (
    GapSignal,
    GapType,
    ScoreChange,
    ScoreComponent,
    ScoringConfig,
    Suggestion,
)
from app.services.labels import label_key
from app.services.relevance import (  # fmt: skip
    CONTAINS,
    StrategicFit,
    audience_matches,
    best_match,
    similar,
    strategic_fit,
)
from app.services.trends import AnalysisFact, TopicInfo, TrendEngine

DIMENSIONS = ("momentum", "strategic_fit", "audience_fit", "content_gap", "recency")
SHALLOW_WORDS, DEEP_WORDS = 200, 800  # median length below DEEP_WORDS counts as shallower
MAX_EVIDENCE_PAGES = 8


@dataclass
class Candidate:
    key: str
    topic_id: int | None
    label: str
    topic: TopicRef | None
    score: float
    breakdown: list[ScoreComponent]
    gaps: list[GapSignal]
    suggestion: Suggestion
    signals: dict[str, Any]
    strategic: StrategicFit
    trend: TopicTrend | None
    evidence_analysis_ids: list[int]
    aliases: tuple[str, ...] = ()
    related: list["Candidate"] = field(default_factory=list)
    rejected: str | None = None  # why it isn't an opportunity

    @property
    def labels(self) -> tuple[str, ...]:
        return (self.label, *self.aliases)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _shares(values: Sequence[str], total: int) -> dict[str, float]:
    counts = Counter(values)
    return {k: round(v / total, 4) for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))} if total else {}  # fmt: skip


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


class OpportunitySignalEngine:
    def __init__(
        self,
        facts: Sequence[AnalysisFact],
        topics: Mapping[int, TopicInfo],
        company: CompanyProfile,
        config: ScoringConfig,
        *,
        now: datetime,
        aliases: Mapping[int, Sequence[str]] | None = None,
    ) -> None:
        self._facts = list(facts)
        self._topics = topics
        self._aliases = aliases or {}
        self.company = company
        self.config = config
        self.now = now
        self.trends = TrendEngine(self._facts, topics, now=now, window_days=config.window_days)
        self.competitors = sorted({f.competitor for f in self._facts})
        weights = config.weights
        positive = sum(getattr(weights, d) for d in DIMENSIONS)
        if positive <= 0:
            raise ValueError("At least one positive scoring weight must be greater than 0")
        self._scale = 100 / positive

    # ── public API ───────────────────────────────────────────────────────────

    def candidates(self) -> list[Candidate]:
        """Every scored candidate (including rejected ones), best first."""
        by_topic: dict[int, list[tuple[AnalysisFact, TopicRole]]] = defaultdict(list)
        subtopics: dict[int, list[tuple[AnalysisFact, int]]] = defaultdict(list)
        for fact in self._facts:
            for link in fact.topics:
                info = self._topics.get(link.topic_id)
                if info is None:
                    continue
                if info.parent_id is None:
                    by_topic[link.topic_id].append((fact, link.role))
                else:
                    subtopics[info.parent_id].append((fact, link.topic_id))
        found = [self._topic_candidate(t, rows, subtopics.get(t, [])) for t, rows in by_topic.items()]  # fmt: skip
        found += self._core_topic_gaps(found)
        return sorted(found, key=lambda c: (-c.score, c.label.casefold()))

    def opportunities(self) -> tuple[list[Candidate], list[Candidate]]:
        """(qualified opportunities, deduplicated and capped; rejected candidates)."""
        qualified: list[Candidate] = []
        rejected: list[Candidate] = []
        for candidate in self.candidates():
            if candidate.rejected is None and candidate.score < self.config.min_score:
                candidate.rejected = f"score {candidate.score} is below the minimum {self.config.min_score}"  # fmt: skip
            if candidate.rejected:
                rejected.append(candidate)
                continue
            group = next((q for q in qualified if similar(q.labels, candidate.labels)), None)
            if group is not None:  # a near-duplicate of a better-scoring candidate
                candidate.rejected = f"near-duplicate of '{group.label}'"
                group.related.append(candidate)
                group.signals["related_topics"] = [r.label for r in group.related]
                rejected.append(candidate)
                continue
            qualified.append(candidate)
        for extra in qualified[self.config.max_opportunities :]:
            extra.rejected = f"beyond the top {self.config.max_opportunities}"
            rejected.append(extra)
        return qualified[: self.config.max_opportunities], rejected

    # ── candidates ───────────────────────────────────────────────────────────

    def _topic_candidate(
        self,
        topic_id: int,
        rows: list[tuple[AnalysisFact, TopicRole]],
        sub_rows: list[tuple[AnalysisFact, int]],
    ) -> Candidate:
        cfg = self.config
        info = self._topics[topic_id]
        facts = [fact for fact, _ in rows]
        items = len(facts)
        total_competitors = len(self.competitors)
        by_competitor = Counter(f.competitor for f in facts)
        trends = self.trends.topic_trends(topic_ids=[topic_id])
        trend = trends[0] if trends else None
        reliable = trend is not None and trend.trend is not TrendDirection.INSUFFICIENT_HISTORY
        per_competitor: dict[str, Counter[str | None]] = defaultdict(Counter)
        for fact in facts:
            per_competitor[fact.competitor][self.trends.bucket(fact)] += 1
        growing = sorted(c for c in self.trends.compared if per_competitor[c]["recent"] > per_competitor[c]["previous"])  # fmt: skip
        recent_all = sum(counter["recent"] for counter in per_competitor.values())
        dated = [f.published_at for f in facts if f.published_at is not None and f.published_at <= self.now]  # fmt: skip
        last = max(dated) if dated else None
        days_since = (self.now - last).days if last else None
        ages = [(self.now - d).days for d in dated]
        median_age = statistics.median(ages) if ages else None
        formats = _shares([f.content_format.value for f in facts], items)
        intents = _shares([f.intent.value for f in facts if f.intent], items)
        audience_labels = Counter(a for f in facts for a in dict.fromkeys(f.audiences))
        median_words = statistics.median([f.word_count for f in facts]) if facts else 0

        # Subtopics: which angles competitors take, and how thinly.
        sub_items: dict[int, set[int]] = defaultdict(set)
        sub_competitors: dict[int, set[str]] = defaultdict(set)
        for fact, sub_id in sub_rows:
            sub_items[sub_id].add(fact.analysis_id)
            sub_competitors[sub_id].add(fact.competitor)
        subtopic_stats: list[dict[str, Any]] = [
            {
                "slug": self._topics[s].slug,
                "name": self._topics[s].name,
                "items": len(ids),
                "competitors": sorted(sub_competitors[s]),
            }
            for s, ids in sub_items.items()
        ]
        subtopic_stats.sort(key=lambda s: (-int(s["items"]), str(s["name"])))

        labels = (info.name, *self._aliases.get(topic_id, ()))
        fit = strategic_fit(
            labels,
            self.company,
            subtopic_labels=[s["name"] for s in subtopic_stats],
            item_terms=[(*f.keywords, *f.key_themes) for f in facts],
        )
        audience_share = {
            a: round(
                sum(1 for f in facts if any(audience_matches(a, x) for x in f.audiences)) / items, 4
            )
            for a in self.company.target_audiences
        }
        best_audience_share = max(audience_share.values(), default=0.0)
        strength = min(1.0, items / cfg.min_items_for_gaps)

        gaps = self._gaps(
            items=items, covering=len(by_competitor), total_competitors=total_competitors,
            strength=strength, audience_share=audience_share, intents=intents, formats=formats,
            subtopics=subtopic_stats, median_words=median_words, median_age=median_age,
        )  # fmt: skip
        values = {
            "momentum": self._momentum(trend, reliable, len(growing)),
            "strategic_fit": fit.value,
            "audience_fit": _clamp(best_audience_share / cfg.audience_fit_share)
            if self.company.target_audiences
            else 0.5,
            "content_gap": max(
                (g.score * getattr(cfg.gap_weights, g.type.value) for g in gaps), default=0.0
            ),
            "recency": 0.5 ** (days_since / cfg.recency_half_life_days)
            if days_since is not None
            else 0.0,
        }
        relief = max((g.score for g in gaps if g.type in (GapType.FRESHNESS, GapType.DEPTH, GapType.AUDIENCE, GapType.INTENT)), default=0.0)  # fmt: skip
        saturation_parts = {
            "volume": _clamp(math.log1p(items) / math.log1p(cfg.saturation_reference_items)),
            "breadth": len(by_competitor) / total_competitors if total_competitors else 0.0,
            "frequency": _clamp(
                (recent_all / (cfg.window_days / 7)) / cfg.saturation_reference_per_week
            ),
            "variety": _clamp(len(formats) / 5),
        }
        raw_saturation = 0.35 * saturation_parts["volume"] + 0.30 * saturation_parts["breadth"] + 0.20 * saturation_parts["frequency"] + 0.15 * saturation_parts["variety"]  # fmt: skip
        saturation = raw_saturation * (1 - cfg.saturation_relief * relief)

        signals: dict[str, Any] = {
            "items": items,
            "competitors_total": total_competitors,
            "competitors_covering": len(by_competitor),
            "by_competitor": dict(sorted(by_competitor.items())),
            "coverage_ratio": round(len(by_competitor) / total_competitors, 4)
            if total_competitors
            else 0.0,
            "window_days": cfg.window_days,
            "recent": trend.recent if trend else 0,
            "previous": trend.previous if trend else 0,
            "growth_pct": round((trend.recent - trend.previous) / trend.previous * 100, 1)
            if trend and trend.previous
            else None,
            "trend": trend.trend.value if trend else None,
            "growth_reliable": reliable,
            "growing_competitors": growing,
            "compared_competitors": self.trends.compared,
            "recent_all_competitors": recent_all,
            "recent_per_week": round(recent_all / (cfg.window_days / 7), 2),
            "last_published_at": last.isoformat() if last else None,
            "days_since_last": days_since,
            "median_age_days": median_age,
            "median_words": median_words,
            "formats": formats,
            "intents": intents,
            "audiences": dict(audience_labels.most_common(8)),
            "company_audience_share": audience_share,
            "subtopics": subtopic_stats[:15],
            "strategic_fit": {"value": fit.value, "matches": list(fit.matches)},
            "saturation": {
                "raw": round(raw_saturation, 4),
                "relief": round(relief, 4),
                "effective": round(saturation, 4),
                **{k: round(v, 4) for k, v in saturation_parts.items()},
            },
            "related_topics": [],
        }
        candidate = self._scored(
            key=f"topic:{topic_id}", topic_id=topic_id, label=info.name, topic=info.ref(),
            values=values, saturation=saturation, gaps=gaps, signals=signals, fit=fit, trend=trend,
            evidence=self._evidence_pages(rows), aliases=tuple(self._aliases.get(topic_id, ())),
        )  # fmt: skip
        if fit.excluded_by:
            candidate.rejected = f"excluded by your company profile ('{fit.excluded_by}')"
        elif fit.value < cfg.min_strategic_fit:
            candidate.rejected = f"strategic fit {fit.value} is below the minimum {cfg.min_strategic_fit}"  # fmt: skip
        elif items < cfg.min_topic_items:
            candidate.rejected = f"only {items} competitor page(s); at least {cfg.min_topic_items} needed"  # fmt: skip
        return candidate

    def _core_topic_gaps(self, found: Sequence[Candidate]) -> list[Candidate]:
        """Core company topics no competitor covers: a topic gap, if the corpus is big
        enough for absence to mean something."""
        corpus = len(self._facts)
        if corpus < self.config.topic_gap_min_corpus_items or not self.competitors:
            return []
        gaps = []
        for core in self.company.core_topics:
            if any(best_match(c.labels, [core])[0] >= CONTAINS for c in found):
                continue
            fit = strategic_fit([core], self.company)
            gap = GapSignal(
                type=GapType.TOPIC, score=1.0,
                detail=f"none of the {len(self.competitors)} competitors covers it ({corpus} analyzed pages)",
                data={"competitors_covering": 0, "competitors_total": len(self.competitors), "corpus_items": corpus},
            )  # fmt: skip
            signals: dict[str, Any] = {
                "items": 0, "competitors_total": len(self.competitors), "competitors_covering": 0,
                "by_competitor": {}, "coverage_ratio": 0.0, "window_days": self.config.window_days,
                "recent": 0, "previous": 0, "growth_pct": None, "trend": None, "growth_reliable": False,
                "growing_competitors": [], "corpus_items": corpus, "company_core_topic": core,
                "strategic_fit": {"value": fit.value, "matches": list(fit.matches)},
                "saturation": {"raw": 0.0, "relief": 0.0, "effective": 0.0}, "related_topics": [],
            }  # fmt: skip
            values = {"momentum": 0.0, "strategic_fit": fit.value, "audience_fit": 0.5, "content_gap": 1.0 * self.config.gap_weights.topic, "recency": 0.0}  # fmt: skip
            gaps.append(
                self._scored(
                    key=f"core:{label_key(core)}",
                    topic_id=None,
                    label=core,
                    topic=None,
                    values=values,
                    saturation=0.0,
                    gaps=[gap],
                    signals=signals,
                    fit=fit,
                    trend=None,
                    evidence=[],
                )
            )
        return gaps

    # ── signals ──────────────────────────────────────────────────────────────

    def _momentum(self, trend: TopicTrend | None, reliable: bool, growing: int) -> float:
        """60% smoothed growth (log2 of (recent+1)/(previous+1): flat → 0.5, 4x → 1.0,
        1/4 → 0.0) and 40% the share of compared competitors growing on the topic. Without
        a trustworthy comparison: neutral growth and no breadth credit."""
        if trend is None or trend.recent + trend.previous == 0:
            return 0.0
        if not reliable:
            return 0.3
        growth = _clamp(0.5 + math.log2((trend.recent + 1) / (trend.previous + 1)) / 4)
        breadth = growing / max(len(self.trends.compared), 1)
        return round(0.6 * growth + 0.4 * breadth, 4)

    def _gaps(
        self,
        *,
        items: int,
        covering: int,
        total_competitors: int,
        strength: float,
        audience_share: dict[str, float],
        intents: dict[str, float],
        formats: dict[str, float],
        subtopics: list[dict[str, Any]],
        median_words: float,
        median_age: float | None,
    ) -> list[GapSignal]:
        cfg = self.config
        gaps: list[GapSignal] = []
        if total_competitors >= 2:
            topic_gap = 1 - covering / total_competitors
            gaps.append(GapSignal(type=GapType.TOPIC, score=round(topic_gap, 4), detail=f"covered by {covering} of {total_competitors} competitors", data={"competitors_covering": covering, "competitors_total": total_competitors}))  # fmt: skip
        if audience_share:
            best_audience, best = max(audience_share.items(), key=lambda kv: (kv[1], kv[0]))
            weakest, weakest_share = min(audience_share.items(), key=lambda kv: (kv[1], kv[0]))
            score = (1 - _clamp(best / cfg.audience_served_share)) * strength
            gaps.append(GapSignal(type=GapType.AUDIENCE, score=round(score, 4), detail=f"{weakest} are targeted by {_pct(weakest_share)} of {items} competitor pages (best served: {best_audience} {_pct(best)})", data={"company_audience_share": audience_share, "underserved": [a for a, s in audience_share.items() if s < cfg.audience_served_share]}))  # fmt: skip
        # Intent and format gaps: the *combined* share of the valuable intents (formats) is
        # what's underserved; the scarcest one is what the suggestion names.
        valuable_intents = [i.value for i in cfg.valuable_intents]
        if valuable_intents and items:
            combined = sum(intents.get(i, 0.0) for i in valuable_intents)
            scarce = sorted(valuable_intents, key=lambda i: (intents.get(i, 0.0), i))
            score = (1 - _clamp(combined / cfg.expected_share)) * strength
            gaps.append(GapSignal(type=GapType.INTENT, score=round(score, 4), detail=f"{' or '.join(valuable_intents)} intent in {_pct(combined)} of {items} competitor pages", data={"intents": intents, "valuable_share": round(combined, 4), "underserved": scarce}))  # fmt: skip
        valuable_formats = [f.value for f in (self.company.preferred_formats or cfg.valuable_formats)]  # fmt: skip
        if valuable_formats and items:
            combined = sum(formats.get(f, 0.0) for f in valuable_formats)
            scarce = sorted(valuable_formats, key=lambda f: (formats.get(f, 0.0), f))
            score = (1 - _clamp(combined / cfg.expected_share)) * strength
            names = ", ".join(f.replace("_", " ") for f in valuable_formats)
            gaps.append(GapSignal(type=GapType.FORMAT, score=round(score, 4), detail=f"{names} formats in {_pct(combined)} of {items} competitor pages", data={"formats": formats, "valuable_share": round(combined, 4), "underserved": scarce}))  # fmt: skip
        if items:
            thin = [s["name"] for s in subtopics if s["items"] == 1]
            fragmentation = len(thin) / len(subtopics) if len(subtopics) >= 3 else 0.0
            shallowness = _clamp((DEEP_WORDS - median_words) / (DEEP_WORDS - SHALLOW_WORDS))
            depth = max(fragmentation, shallowness) * strength
            detail = f"{len(thin)} of {len(subtopics)} subtopics touched by a single page" if fragmentation >= shallowness else f"median competitor page is {median_words:.0f} words"  # fmt: skip
            gaps.append(GapSignal(type=GapType.DEPTH, score=round(depth, 4), detail=detail, data={"thin_subtopics": thin[:10], "fragmentation": round(fragmentation, 4), "median_words": median_words}))  # fmt: skip
        if median_age is not None:
            freshness = _clamp((median_age - cfg.fresh_days) / (cfg.stale_days - cfg.fresh_days))
            gaps.append(GapSignal(type=GapType.FRESHNESS, score=round(freshness, 4), detail=f"median competitor page is {median_age:.0f} days old", data={"median_age_days": median_age}))  # fmt: skip
        if covering >= 2 and items >= cfg.min_items_for_gaps and formats:
            top_format, top_share = next(iter(formats.items()))
            differentiation = _clamp((top_share - 0.5) / 0.5)
            gaps.append(GapSignal(type=GapType.DIFFERENTIATION, score=round(differentiation, 4), detail=f"{_pct(top_share)} of competitor pages are {top_format.replace('_', ' ')}s", data={"dominant_format": top_format, "share": top_share}))  # fmt: skip
        return gaps

    def _evidence_pages(self, rows: list[tuple[AnalysisFact, TopicRole]]) -> list[int]:
        """The competitor pages that best represent the topic: primary first, newest first,
        at least one per covering competitor when possible."""
        ranked = sorted(
            rows,
            key=lambda row: (
                row[1] is not TopicRole.PRIMARY,
                -(row[0].published_at.timestamp() if row[0].published_at else 0),
                row[0].analysis_id,
            ),
        )
        chosen: list[int] = []
        seen: set[str] = set()
        for fact, _ in ranked:  # one per competitor first
            if fact.competitor not in seen:
                seen.add(fact.competitor)
                chosen.append(fact.analysis_id)
        for fact, _ in ranked:
            if len(chosen) >= MAX_EVIDENCE_PAGES:
                break
            if fact.analysis_id not in chosen:
                chosen.append(fact.analysis_id)
        return chosen[:MAX_EVIDENCE_PAGES]

    # ── scoring ──────────────────────────────────────────────────────────────

    def _scored(
        self,
        *,
        key: str,
        topic_id: int | None,
        label: str,
        topic: TopicRef | None,
        values: dict[str, float],
        saturation: float,
        gaps: list[GapSignal],
        signals: dict[str, Any],
        fit: StrategicFit,
        trend: TopicTrend | None,
        evidence: list[int],
        aliases: tuple[str, ...] = (),
    ) -> Candidate:
        weights = self.config.weights
        breakdown = []
        for dimension in DIMENSIONS:
            maximum = round(getattr(weights, dimension) * self._scale, 2)
            value = round(_clamp(values[dimension]), 4)
            breakdown.append(ScoreComponent(dimension=dimension, points=round(maximum * value, 2), max_points=maximum, value=value, detail=_dimension_detail(dimension, value, signals, gaps)))  # fmt: skip
        penalty = round(weights.saturation * _clamp(saturation), 2)
        breakdown.append(ScoreComponent(dimension="saturation", points=-penalty, max_points=weights.saturation, value=round(_clamp(saturation), 4), detail=_dimension_detail("saturation", saturation, signals, gaps)))  # fmt: skip
        score = round(_clamp(sum(c.points for c in breakdown), 0, 100), 1)
        weighted = [(g.score * getattr(self.config.gap_weights, g.type.value), g) for g in gaps]
        primary = max(weighted, key=lambda pair: pair[0], default=(0.0, None))
        return Candidate(
            key=key,
            topic_id=topic_id,
            label=label,
            topic=topic,
            score=score,
            breakdown=breakdown,
            gaps=gaps,
            suggestion=self._suggest(gaps, signals, primary[1] if primary[0] >= 0.3 else None),
            signals=signals,
            strategic=fit,
            trend=trend,
            evidence_analysis_ids=evidence,
            aliases=aliases,
        )

    def _suggest(self, gaps: list[GapSignal], signals: dict[str, Any], primary: GapSignal | None) -> Suggestion:  # fmt: skip
        by_type = {g.type: g for g in gaps}
        fmt = by_type.get(GapType.FORMAT)
        intent = by_type.get(GapType.INTENT)
        audience = by_type.get(GapType.AUDIENCE)
        # The valuable (or your preferred) format competitors use least on this topic; for a
        # topic nobody covers, your first preferred format.
        format_choice = (fmt.data["underserved"] or [None])[0] if fmt else None
        if format_choice is None and self.company.preferred_formats:
            format_choice = self.company.preferred_formats[0].value
        intent_choice = (
            (intent.data["underserved"] or [None])[0] if intent and intent.score >= 0.3 else None
        )
        shares: dict[str, float] = signals.get("company_audience_share", {})
        audience_choice = None
        if shares:
            audience_choice = (
                min(shares, key=lambda a: (shares[a], a))
                if audience and audience.score >= 0.3
                else max(shares, key=lambda a: (shares[a], a))
            )
        elif self.company.target_audiences:
            audience_choice = self.company.target_audiences[0]
        return Suggestion(
            format=ContentFormat(format_choice) if format_choice else None,
            audience=audience_choice,
            intent=SearchIntent(intent_choice) if intent_choice else None,
            primary_gap=primary.type if primary else None,
            reasons=_reasons(signals, gaps),
        )


# ── explanations (deterministic text with exact numbers) ─────────────────────


def _dimension_detail(dimension: str, value: float, signals: dict[str, Any], gaps: list[GapSignal]) -> str:  # fmt: skip
    if dimension == "momentum":
        if not signals.get("growth_reliable"):
            return f"{signals['recent']} recent vs {signals['previous']} previous items; history too short to compare growth" if signals["items"] else "no competitor activity"  # fmt: skip
        return f"{signals['recent']} items in the last {signals['window_days']} days vs {signals['previous']} before; growing at {len(signals['growing_competitors'])} competitor(s)"  # fmt: skip
    if dimension == "strategic_fit":
        return (
            "; ".join(signals["strategic_fit"]["matches"]) or "no match with your company profile"
        )
    if dimension == "audience_fit":
        shares = signals.get("company_audience_share") or {}
        return (
            ", ".join(f"{a} {_pct(s)}" for a, s in shares.items()) or "no competitor audience data"
        )
    if dimension == "content_gap":
        strongest = max(gaps, key=lambda g: g.score, default=None)
        return f"{strongest.type.value} gap: {strongest.detail}" if strongest else "no gap detected"
    if dimension == "recency":
        days = signals.get("days_since_last")
        return f"last competitor page {days} days ago" if days is not None else "no reliably dated competitor page"  # fmt: skip
    sat = signals["saturation"]
    return f"raw {sat['raw']:.2f}, relieved by weak coverage {sat['relief']:.2f} → {sat['effective']:.2f}"


def _reasons(signals: dict[str, Any], gaps: list[GapSignal]) -> list[str]:
    reasons = []
    growing = signals.get("growing_competitors") or []
    if len(growing) >= 2:
        reasons.append(f"{len(growing)} competitors increased publishing on this topic ({', '.join(growing)})")  # fmt: skip
    if signals.get("growth_pct") is not None and signals.get("growth_reliable"):
        reasons.append(f"topic growth: {signals['growth_pct']:+.0f}% ({signals['previous']} → {signals['recent']} items, {signals['window_days']}-day windows)")  # fmt: skip
    elif signals.get("recent") and not signals.get("previous") and signals.get("growth_reliable"):
        reasons.append(f"new activity: {signals['recent']} items in the last {signals['window_days']} days, none before")  # fmt: skip
    if signals.get("items"):
        reasons.append(f"covered by {signals['competitors_covering']} of {signals['competitors_total']} competitors ({signals['items']} pages)")  # fmt: skip
    for match in signals["strategic_fit"]["matches"][:1]:
        reasons.append(f"relevant to you: {match}")
    for gap in sorted(gaps, key=lambda g: -g.score):
        if gap.score >= 0.5:
            reasons.append(f"{gap.type.value} gap: {gap.detail}")
    return reasons


def explain_change(
    previous: Mapping[str, Any],
    current: "Candidate",
    *,
    old_basis: Mapping[str, Any],
    new_basis: Mapping[str, Any],
    forced: bool = False,
) -> ScoreChange:
    """Why the score moved: point change per dimension, with the signals that drove it,
    plus changes to the basis (company profile, scoring config, window, evidence pages).

    ``previous`` holds the stored assessment's ``score``, ``breakdown`` and ``signals``;
    a basis holds ``scoring``, ``company`` (the whole profile), ``company_scoring`` (its
    scoring fields), ``window_days`` and ``evidence_ids``.
    """
    profile_changed = old_basis.get("company_scoring") != new_basis.get("company_scoring")
    context_changed = "company" in old_basis and old_basis["company"] != new_basis.get("company")
    scoring_changed = old_basis.get("scoring") != new_basis.get("scoring")
    before = {c["dimension"]: c["points"] for c in previous["breakdown"]}
    old = previous["signals"]
    new = current.signals
    deltas = {c.dimension: round(c.points - before.get(c.dimension, 0.0), 2) for c in current.breakdown}  # fmt: skip
    drivers = {
        "momentum": [("recent items", "recent"), ("previous items", "previous")],
        "content_gap": [("competitors covering", "competitors_covering")],
        "recency": [("days since last competitor page", "days_since_last")],
        "saturation": [
            ("competitor pages", "items"),
            ("competitors covering", "competitors_covering"),
        ],
    }
    reasons = []
    for dimension, delta in sorted(deltas.items(), key=lambda kv: -abs(kv[1])):
        if abs(delta) < 0.5:
            continue
        parts = [f"{label} {old.get(k)} → {new.get(k)}" for label, k in drivers.get(dimension, []) if old.get(k) != new.get(k)]  # fmt: skip
        if dimension == "momentum" and len(old.get("growing_competitors", [])) != len(new.get("growing_competitors", [])):  # fmt: skip
            parts.append(f"growing competitors {len(old.get('growing_competitors', []))} → {len(new.get('growing_competitors', []))}")  # fmt: skip
        if dimension == "strategic_fit":
            parts.append(
                f"fit {old.get('strategic_fit', {}).get('value')} → {new['strategic_fit']['value']}"
            )
            if profile_changed:
                parts.append("company profile changed")
        reasons.append(f"{dimension.replace('_', ' ')} {delta:+.1f} pts" + (f" ({'; '.join(parts)})" if parts else ""))  # fmt: skip
    if scoring_changed:
        reasons.append("scoring configuration changed")
    if profile_changed and not any(r.startswith("strategic fit") for r in reasons):
        reasons.append("company profile changed")
    elif context_changed and not profile_changed:
        reasons.append("company profile changed (fields that don't affect scores)")
    if old_basis.get("window_days") != new_basis.get("window_days"):
        reasons.append(f"window {old_basis.get('window_days')} → {new_basis.get('window_days')} days")  # fmt: skip
    old_ids, new_ids = set(old_basis.get("evidence_ids", [])), set(new_basis.get("evidence_ids", []))  # fmt: skip
    if old_ids != new_ids:
        reasons.append(
            f"analysed pages: {len(new_ids - old_ids)} added, {len(old_ids - new_ids)} dropped"
        )
    fallback = "recalculated on request; inputs unchanged" if forced else "no material change"
    return ScoreChange(
        previous_score=previous["score"],
        score=current.score,
        delta=round(current.score - previous["score"], 1),
        dimensions=deltas,
        reasons=reasons or [fallback],
    )
