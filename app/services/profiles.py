"""Versioned competitor profiles: deterministic evidence pack → Gemini → grounding checks.

The model sees only this competitor's own analyzed pages (positioning pages first, then
recent editorial content), its pricing page excerpt, summarized changes, and statistics.
Every model-written statement must cite evidence ids; statements citing none are
dropped. Content-strategy facts in the profile are computed deterministically.

A new version is written only when the evidence changed (hash of the rendered evidence,
prompt version and model), unless forced.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from sqlalchemy import case, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db import analysis_queries
from app.db.models import (
    Competitor,
    CompetitorProfileSnapshot,
    ContentAnalysis,
    ContentItem,
    ContentVersion,
)
from app.db.session import SessionFactory
from app.domain.analysis import (
    EDITORIAL_TYPES,
    POSITIONING_TYPES,
    ChangeSummaryView,
    ContentQuality,
    LLMPurpose,
    RecentChange,
)
from app.domain.competitor_profile import (
    Claim,
    CompetitorProfile,
    EvidenceRef,
    FocusTopic,
    PricingTier,
)
from app.domain.content import ContentType
from app.domain.history import ItemStatus
from app.llm import LLMRequest
from app.prompts import competitor_profile as prompt
from app.services.digest import condense, neutralize
from app.services.llm_usage import BudgetedLLM
from app.services.trends import Dimension, TrendEngine

PROFILE_WINDOW_DAYS = 90
MAX_POSITIONING_EVIDENCE = 10
MAX_EDITORIAL_EVIDENCE = 8
MAX_CHANGE_EVIDENCE = 6
PRICING_EXCERPT_CHARS = 3_000
_TYPE_ORDER = {ContentType.HOMEPAGE: 0, ContentType.PRICING: 1, ContentType.PRODUCT: 2, ContentType.LANDING_PAGE: 3}  # fmt: skip


@dataclass
class ProfileResult:
    status: Literal["created", "unchanged", "skipped"]
    version: int | None = None
    detail: str | None = None
    dropped_claims: int = 0


@dataclass(frozen=True)
class _Evidence:
    ref: str
    evidence: EvidenceRef
    line: str


def _percent(value: float) -> str:
    return f"{value * 100:.0f}%"


async def _evidence_rows(
    session: AsyncSession, competitor_id: int
) -> list[tuple[ContentAnalysis, ContentItem, ContentVersion]]:
    latest = analysis_queries.latest_analysis_ids([competitor_id])
    base = (
        select(ContentAnalysis, ContentItem, ContentVersion)
        .join(latest, latest.c.id == ContentAnalysis.id)
        .join(ContentItem, ContentItem.id == ContentAnalysis.content_item_id)
        .join(ContentVersion, ContentVersion.id == ContentAnalysis.content_version_id)
        .where(
            ContentItem.status == ItemStatus.ACTIVE.value,
            ContentAnalysis.content_quality == ContentQuality.SUBSTANTIVE.value,
        )
    )
    type_rank = case(
        *((ContentItem.content_type == t.value, rank) for t, rank in _TYPE_ORDER.items()), else_=9
    )
    positioning = await session.execute(
        base.where(ContentItem.content_type.in_([t.value for t in POSITIONING_TYPES]))
        .order_by(type_rank, ContentVersion.word_count.desc(), ContentItem.id)
        .limit(MAX_POSITIONING_EVIDENCE)
    )
    editorial = await session.execute(
        base.where(
            ContentItem.content_type.in_([t.value for t in EDITORIAL_TYPES]),
            ContentItem.published_at.is_not(None),
        )
        .order_by(ContentItem.published_at.desc(), ContentItem.id.desc())
        .limit(MAX_EDITORIAL_EVIDENCE)
    )
    return [(a, i, v) for a, i, v in [*positioning, *editorial]]


def _evidence_line(ref: str, analysis: ContentAnalysis, item: ContentItem, version: ContentVersion) -> str:  # fmt: skip
    parts = [f"{ref} | {item.content_type} | {item.url} | observed {version.observed_at:%Y-%m-%d}"]
    if item.published_at:
        parts.append(f"  published: {item.published_at:%Y-%m-%d}")
    parts.append(f"  title: {item.title or '(none)'}")
    parts.append(f"  summary: {analysis.summary}")
    if analysis.primary_angle:
        parts.append(f"  angle: {analysis.primary_angle}")
    if analysis.positioning_claims:
        parts.append(f"  claims: {'; '.join(analysis.positioning_claims)}")
    if analysis.key_themes:
        parts.append(f"  themes: {'; '.join(analysis.key_themes)}")
    if analysis.target_audiences:
        parts.append(f"  audiences: {'; '.join(analysis.target_audiences)}")
    return neutralize("\n".join(parts))


def _change_line(ref: str, change: RecentChange, summary: ChangeSummaryView) -> str:
    return neutralize(
        f"{ref} | {change.content_type.value} | {change.url} | detected {change.detected_at:%Y-%m-%d} | "
        f"{summary.significance.value} | {summary.summary}"
    )


def _statistics(engine: TrendEngine, slug: str) -> list[str]:
    cadence = engine.cadence(competitor=slug)
    topics = engine.topic_trends(competitor=slug)[:8]
    lines = [
        f"- Analyzed pages: {engine.analyzed_items(slug)}",
        f"- Published in the last {cadence.window_days} days: {cadence.recent} "
        f"({cadence.per_week}/week); previous {cadence.window_days} days: {cadence.previous}; "
        f"without a reliable date: {cadence.undated}",
        "- Top topics: "
        + (
            ", ".join(f"{t.topic.name} {_percent(t.share)} ({t.trend.value})" for t in topics)
            or "none"
        ),
    ]
    dimensions: tuple[Dimension, ...] = ("format", "audience", "intent")
    for dimension in dimensions:
        shares = engine.mix(dimension, competitor=slug)[:6]
        if shares:
            lines.append(f"- {dimension.capitalize()}s: " + ", ".join(f"{s.value} {_percent(s.share)}" for s in shares))  # fmt: skip
    return lines


def _ground(out: prompt.CompetitorProfileOut, refs: dict[str, EvidenceRef]) -> tuple[dict[str, Any], int]:  # fmt: skip
    """Model output → profile fields, keeping only statements with valid evidence."""
    dropped = 0

    def cite(ids: list[str]) -> list[EvidenceRef]:
        seen: dict[int, EvidenceRef] = {}
        for raw in ids:
            ref = refs.get(raw.strip().upper())
            if ref is not None:
                seen.setdefault(ref.content_item_id, ref)
        return list(seen.values())

    def claim(value: prompt.ClaimOut | None) -> Claim | None:
        nonlocal dropped
        if value is None or not value.text:
            return None
        evidence = cite(value.evidence)
        if not evidence:
            dropped += 1
            return None
        return Claim(text=value.text, evidence=evidence)

    def claims(values: list[prompt.ClaimOut]) -> list[Claim]:
        return [c for c in (claim(v) for v in values) if c is not None]

    tiers = []
    for tier in out.pricing_tiers:
        evidence = cite(tier.evidence)
        if evidence:
            tiers.append(PricingTier(name=tier.name, price=tier.price, billing_period=tier.billing_period, highlights=tier.highlights, evidence=evidence))  # fmt: skip
        else:
            dropped += 1
    fields = {
        "tagline": claim(out.tagline),
        "description": claim(out.description),
        "positioning_statement": claim(out.positioning_statement),
        "target_audiences": claims(out.target_audiences),
        "value_propositions": claims(out.value_propositions),
        "key_features": claims(out.key_features),
        "differentiators": claims(out.differentiators),
        "pricing_model": claim(out.pricing_model),
        "pricing_tiers": tiers,
        "content_strategy": claim(out.content_strategy),
        "notable_changes": claims(out.notable_changes),
        "confidence": out.confidence,
    }
    return fields, dropped


async def refresh_profile(
    *,
    sessions: SessionFactory,
    llm: BudgetedLLM,
    settings: Settings,
    competitor: Competitor,
    run_id: int | None,
    now: datetime,
    force: bool = False,
) -> ProfileResult:
    async with sessions() as session:
        facts, topics = await analysis_queries.load_facts(session, competitor_ids=[competitor.id])
        if not facts:
            return ProfileResult("skipped", detail="no analyzed content yet")
        rows = await _evidence_rows(session, competitor.id)
        pricing_row = next((r for r in rows if r[1].content_type == ContentType.PRICING.value), None)  # fmt: skip
        changes = await analysis_queries.recent_changes(
            session,
            competitor_ids=[competitor.id],
            since=now - timedelta(days=PROFILE_WINDOW_DAYS),
            summarized_only=True,
            limit=MAX_CHANGE_EVIDENCE,
        )
        latest = await analysis_queries.latest_profile_row(session, competitor.id)

    evidence: list[_Evidence] = []
    for index, (analysis, item, version) in enumerate(rows, start=1):
        ref = f"E{index}"
        evidence.append(
            _Evidence(
                ref,
                EvidenceRef(
                    content_item_id=item.id,
                    url=item.url,
                    content_type=ContentType(item.content_type),
                    observed_at=version.observed_at,
                ),
                _evidence_line(ref, analysis, item, version),
            )
        )
    summarized = [(change, change.summary) for change in changes if change.summary is not None]
    for index, (change, summary) in enumerate(summarized, start=1):
        ref = f"C{index}"
        evidence.append(
            _Evidence(
                ref,
                EvidenceRef(
                    content_item_id=change.content_item_id,
                    url=change.url,
                    content_type=change.content_type,
                    observed_at=change.detected_at,
                ),
                _change_line(ref, change, summary),
            )
        )
    if not rows:
        return ProfileResult("skipped", detail="no substantive positioning or dated editorial pages analyzed yet")  # fmt: skip

    pricing = None
    if pricing_row is not None:
        ref = next(e.ref for e in evidence if e.evidence.content_item_id == pricing_row[1].id)
        excerpt, _ = condense(neutralize(pricing_row[2].text), PRICING_EXCERPT_CHARS)
        pricing = (ref, excerpt.replace("</pricing", "<\\/pricing"))
    engine = TrendEngine(facts, topics, now=now, window_days=PROFILE_WINDOW_DAYS)
    evidence_lines = [e.line for e in evidence if e.ref.startswith("E")]
    change_lines = [e.line for e in evidence if e.ref.startswith("C")]
    rendered = prompt.render(
        competitor=competitor.name,
        website=competitor.website,
        evidence=evidence_lines,
        changes=change_lines,
        pricing=pricing,
        statistics=_statistics(engine, competitor.slug),
    )
    # The hash covers the evidence, not the statistics (whose windows move daily): the
    # profile is regenerated when what it cites changes.
    fingerprint = "\n".join([prompt.VERSION, settings.synthesis_model, *evidence_lines, *change_lines, pricing[1] if pricing else ""])  # fmt: skip
    input_hash = hashlib.sha256(fingerprint.encode()).hexdigest()
    if latest is not None and latest.input_hash == input_hash and not force:
        return ProfileResult("unchanged", version=latest.version, detail="evidence unchanged since the last profile")  # fmt: skip

    response = await llm.structured(
        LLMRequest(
            prompt=rendered,
            system=prompt.SYSTEM,
            model=settings.synthesis_model,
            max_output_tokens=8_000,
            reasoning_effort=settings.synthesis_reasoning_effort,
        ),
        prompt.CompetitorProfileOut,
        purpose=LLMPurpose.COMPETITOR_PROFILE,
        prompt_version=prompt.VERSION,
        items=len(evidence),
    )
    fields, dropped = _ground(response.data, {e.ref: e.evidence for e in evidence})
    profile = CompetitorProfile(
        name=competitor.name,
        website=competitor.website,
        **fields,
        focus_topics=[
            FocusTopic(topic=t.topic, items=t.items, share=t.share, trend=t.trend)
            for t in engine.topic_trends(competitor=competitor.slug)[:8]
        ],
        messaging_themes=engine.themes(competitor=competitor.slug, types=POSITIONING_TYPES),
        formats=engine.mix("format", competitor=competitor.slug)[:8],
        audiences=engine.mix("audience", competitor=competitor.slug)[:8],
        cadence=engine.cadence(competitor=competitor.slug),
        evidence_items=len(evidence),
        unsupported_claims_dropped=dropped,
    )
    async with sessions() as session, session.begin():
        current = await analysis_queries.latest_profile_row(session, competitor.id)
        next_version = (current.version + 1) if current else 1
        session.add(
            CompetitorProfileSnapshot(
                competitor_id=competitor.id,
                version=next_version,
                run_id=run_id,
                model=response.raw.model,
                prompt_version=prompt.VERSION,
                input_hash=input_hash,
                profile=profile.model_dump(mode="json"),
                created_at=now,
            )
        )
    return ProfileResult("created", version=next_version, dropped_claims=dropped)
