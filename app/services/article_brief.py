"""The article brief (Phase 5): deterministic, built only from stored data. No LLM.

Inputs: the approved opportunity, the assessment it was briefed from (score breakdown,
gaps, deterministic suggestion and, if present, Gemini's interpretation from Phase 4),
that assessment's evidence rows (competitor pages and profiles), the company profile and
the competitor domains. The same inputs always give the same brief, and ``provenance``
records which input each choice came from. Length is a writing setting, not part of the
brief (ARTICLE_TARGET_WORDS).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    CompanyProfileVersion,
    Competitor,
    Opportunity,
    OpportunityAssessment,
    OpportunityEvidence,
)
from app.domain.analysis import ContentFormat, SearchIntent
from app.domain.articles import ArticleBrief, BriefCompany, BriefEvidence
from app.domain.company import CompanyProfile
from app.domain.opportunities import EvidenceKind, GapType, InterpretationStatus

BRIEF_VERSION = "article-brief/1"
MAX_KEY_POINTS = 8
MAX_EVIDENCE = 8
_INTERPRETED = (InterpretationStatus.OK.value, InterpretationStatus.REUSED.value)

_WEAKNESS = {
    GapType.TOPIC: "Few competitors cover it",
    GapType.AUDIENCE: "Your audience is underserved",
    GapType.INTENT: "Valuable search intents are rare",
    GapType.FORMAT: "Useful formats are rare",
    GapType.DEPTH: "Coverage is shallow or fragmented",
    GapType.FRESHNESS: "Coverage is dated",
    GapType.DIFFERENTIATION: "Competitors all take the same approach",
}
_STRATEGY = {
    GapType.TOPIC: "Be the thorough, practical resource on a subject few competitors cover.",
    GapType.AUDIENCE: "Write specifically for {audience}, whom competitors rarely address.",
    GapType.INTENT: "Serve the {intent} intent competitors neglect.",
    GapType.FORMAT: "Use a {format}, a format competitors rarely use on this topic.",
    GapType.DEPTH: "Go deeper than competitors, covering the subtopics they only touch on.",
    GapType.FRESHNESS: "Give an up-to-date treatment; most competitor coverage is dated.",
    GapType.DIFFERENTIATION: "Take a distinct angle; competitors all cover it the same way.",
}
_OUTCOME = {
    SearchIntent.INFORMATIONAL: "{audience} understand {topic} well enough to act on it, and see {company} as a credible guide.",
    SearchIntent.COMMERCIAL: "{audience} can evaluate their options for {topic} and tell when {product} is a good fit.",
    SearchIntent.COMPARISON: "{audience} can compare the main approaches to {topic} fairly, and see where {product} fits.",
    SearchIntent.TRANSACTIONAL: "{audience} know how to get started with {topic}, and how {product} helps them do it.",
    SearchIntent.NAVIGATIONAL: "{audience} find what they need about {topic} and {company} quickly.",
}


@dataclass(frozen=True)
class EvidenceRow:
    id: int
    kind: str
    competitor: str | None
    label: str
    data: dict[str, Any]


@dataclass(frozen=True)
class BriefInputs:
    opportunity_id: int
    opportunity_title: str
    topic: str
    assessment_id: int
    score: float
    gaps: list[dict[str, Any]]
    suggestion: dict[str, Any]
    signals: dict[str, Any]
    interpretation: dict[str, Any] | None
    evidence: list[EvidenceRow]
    company_profile_id: int
    company_profile_version: int
    company: CompanyProfile
    competitor_domains: list[str]


class BriefInputError(Exception):
    """The opportunity can't be briefed (unknown, never scored, or no company profile)."""


def domain_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host.removeprefix("www.")


def _gap_type(gap: dict[str, Any]) -> GapType | None:
    try:
        return GapType(str(gap.get("type")))
    except ValueError:
        return None


def build_brief(inputs: BriefInputs) -> ArticleBrief:
    """The brief for ``inputs``: identical inputs give an identical brief."""
    interp = inputs.interpretation or {}
    suggestion = inputs.suggestion
    company = inputs.company
    provenance: dict[str, str] = {}

    def pick(field: str, *options: tuple[str, Any]) -> Any:
        for source, value in options:
            if value:
                provenance[field] = source
                return value
        raise AssertionError(f"no value for {field}")  # the last option is always a default

    audience = pick(
        "target_audience",
        (
            "interpretation",
            interp.get("target_audience")
            if interp.get("target_audience") != "unspecified"
            else None,
        ),
        ("suggestion", suggestion.get("audience")),
        ("company profile", company.target_audiences[0] if company.target_audiences else None),
        ("default", "general readers"),
    )
    intent = SearchIntent(pick("search_intent", ("interpretation", interp.get("search_intent")), ("suggestion", suggestion.get("intent")), ("default", SearchIntent.INFORMATIONAL.value)))  # fmt: skip
    content_type = ContentFormat(pick("content_type", ("interpretation", interp.get("recommended_format")), ("suggestion", suggestion.get("format")), ("company profile", company.preferred_formats[0].value if company.preferred_formats else None), ("default", ContentFormat.GUIDE.value)))  # fmt: skip
    topic = inputs.topic
    gaps = sorted(
        ((g, t) for g in inputs.gaps if (t := _gap_type(g)) is not None),
        key=lambda pair: (-float(pair[0].get("score", 0)), pair[1].value),
    )
    strong = [(g, t) for g, t in gaps if float(g.get("score", 0)) >= 0.3]
    primary = GapType(suggestion["primary_gap"]) if suggestion.get("primary_gap") else (strong[0][1] if strong else None)  # fmt: skip
    fmt_name = content_type.value.replace("_", " ")
    template_angle = (
        {
            GapType.TOPIC: f"A thorough {fmt_name} on {topic} for {audience}: few competitors cover it.",
            GapType.AUDIENCE: f"{topic}, explained specifically for {audience}.",
            GapType.INTENT: f"{topic} from the {intent.value} angle competitors neglect.",
            GapType.FORMAT: f"A {fmt_name} on {topic}, a format competitors rarely use for it.",
            GapType.DEPTH: f"An in-depth treatment of {topic}, covering what competitors only touch on.",
            GapType.FRESHNESS: f"An up-to-date take on {topic}, where most coverage is dated.",
            GapType.DIFFERENTIATION: f"A distinct take on {topic}, where competitors all say the same thing.",
        }.get(primary, f"A practical {fmt_name} on {topic} for {audience}.")
        if primary
        else f"A practical {fmt_name} on {topic} for {audience}."
    )
    angle = pick("primary_angle", ("interpretation", interp.get("recommended_angle")), ("template (primary gap)", template_angle))  # fmt: skip
    strategies = [_STRATEGY[t].format(audience=audience, intent=intent.value, format=fmt_name) for _, t in strong[:3]]  # fmt: skip
    differentiation = pick("differentiation_strategy", ("interpretation", interp.get("differentiation_strategy")), ("template (gaps)", " ".join(strategies)), ("default", f"Be more practical and specific for {audience} than existing coverage."))  # fmt: skip
    title = pick("working_title", ("interpretation", interp.get("title")), ("opportunity", inputs.opportunity_title), ("default", topic))  # fmt: skip
    reasons = suggestion.get("reasons") or []
    why_now = interp.get("why_now") or (reasons[0] if reasons else None)
    provenance["why_now"] = "interpretation" if interp.get("why_now") else ("suggestion" if reasons else "none")  # fmt: skip
    product = company.products[0].name if company.products else company.name
    outcome = _OUTCOME[intent].format(audience=audience[:1].upper() + audience[1:], topic=topic, company=company.name, product=product)  # fmt: skip
    provenance["desired_outcome"] = "template (search intent)"

    key_points = _key_points(inputs, audience, intent, strong, company)
    provenance["key_points"] = "subtopics, gaps and company profile (deterministic)"
    weaknesses = [f"{_WEAKNESS[t]}: {g.get('detail')}" for g, t in strong]
    provenance["competitor_weaknesses"] = "assessment gaps (score ≥ 0.3)"

    content_rows = sorted((e for e in inputs.evidence if e.kind == EvidenceKind.CONTENT.value), key=lambda e: e.id)[:MAX_EVIDENCE]  # fmt: skip
    evidence = [
        BriefEvidence(
            evidence_id=e.id, competitor=e.competitor, url=e.data.get("url"), title=e.data.get("title"),
            published=(e.data.get("published_at") or "")[:10] or None, content_format=e.data.get("content_format"),
            audiences=list(e.data.get("target_audiences") or []), summary=e.data.get("summary"), angle=e.data.get("primary_angle"),
        )
        for e in content_rows
    ]  # fmt: skip
    positioning = [
        f"{e.competitor}: {e.data['positioning_statement']}"
        for e in sorted(inputs.evidence, key=lambda e: e.id)
        if e.kind == EvidenceKind.COMPETITOR_PROFILE.value and e.data.get("positioning_statement")
    ]
    avoid = [
        "Copying, closely paraphrasing or restructuring the competitor pages listed as context.",
        "Presenting competitor claims as facts: attribute them, or leave them out.",
        "Statistics, quotes, studies, customers, prices or product capabilities that aren't in the research or the company profile.",
        f"Claims about {company.name} beyond its company profile.",
        "Keyword stuffing and generic filler.",
    ]
    if company.excluded_topics:
        avoid.append(f"Excluded topics: {', '.join(company.excluded_topics)}.")
    return ArticleBrief(
        builder_version=BRIEF_VERSION,
        opportunity_id=inputs.opportunity_id,
        assessment_id=inputs.assessment_id,
        opportunity_score=inputs.score,
        company_profile_id=inputs.company_profile_id,
        company_profile_version=inputs.company_profile_version,
        topic=topic,
        working_title=title,
        target_audience=audience,
        search_intent=intent,
        primary_angle=angle,
        content_type=content_type,
        desired_outcome=outcome,
        why_now=why_now,
        key_points=key_points,
        competitor_weaknesses=weaknesses,
        differentiation_strategy=differentiation,
        evidence=evidence,
        competitor_positioning=positioning,
        company=BriefCompany(
            name=company.name,
            website=str(company.website) if company.website else None,
            description=company.description,
            products=[
                f"{p.name}: {p.description}" if p.description else p.name for p in company.products
            ],
            target_audiences=list(company.target_audiences),
            core_topics=list(company.core_topics),
            positioning=company.positioning,
            differentiators=list(company.differentiators),
            tone=company.tone,
        ),
        things_to_avoid=avoid,
        competitor_domains=sorted(set(inputs.competitor_domains)),
        provenance=dict(sorted(provenance.items())),
    )


def _key_points(
    inputs: BriefInputs,
    audience: str,
    intent: SearchIntent,
    strong: Sequence[tuple[dict[str, Any], GapType]],
    company: CompanyProfile,
) -> list[str]:
    points: list[str] = []
    subtopics = sorted(inputs.signals.get("subtopics") or [], key=lambda s: (-int(s.get("items", 0)), str(s.get("name"))))  # fmt: skip
    thin = {name for g, t in strong if t is GapType.DEPTH for name in g.get("data", {}).get("thin_subtopics", [])}  # fmt: skip
    for sub in subtopics[:4]:
        name = sub.get("name")
        if name and name not in thin:
            points.append(f"Cover {name}.")
    for name in sorted(thin)[:3]:
        points.append(f"Go deeper on {name}: competitors touch it only briefly.")
    kinds = {t for _, t in strong}
    if GapType.AUDIENCE in kinds or company.target_audiences:
        points.append(f"Address {audience} directly: their questions, constraints and vocabulary.")
    if GapType.INTENT in kinds or intent in (SearchIntent.COMMERCIAL, SearchIntent.COMPARISON):
        points.append("Help readers evaluate their options with clear criteria and trade-offs.")
    if company.core_topics:
        points.append(f"Connect the topic to {', '.join(company.core_topics[:3])} where it genuinely helps the reader.")  # fmt: skip
    return list(dict.fromkeys(points))[:MAX_KEY_POINTS] or [f"Explain {inputs.topic} clearly and practically."]  # fmt: skip


async def load_brief_inputs(
    session: AsyncSession,
    opportunity_id: int,
    *,
    assessment_id: int | None = None,
    company_profile_id: int | None = None,
) -> BriefInputs:
    """Everything the brief is built from. By default the opportunity's current assessment
    and the latest company profile; an existing article passes the ones it's pinned to."""
    opportunity = await session.get(Opportunity, opportunity_id)
    if opportunity is None:
        raise BriefInputError(f"Unknown opportunity {opportunity_id}")
    target = assessment_id or opportunity.current_assessment_id
    assessment = await session.get(OpportunityAssessment, target) if target else None
    if assessment is None or assessment.opportunity_id != opportunity_id:
        raise BriefInputError(f"Opportunity {opportunity_id} has no assessment to brief from: run `opportunities generate`")  # fmt: skip
    if company_profile_id is not None:
        profile_row = await session.get(CompanyProfileVersion, company_profile_id)
    else:
        profile_row = await session.scalar(select(CompanyProfileVersion).order_by(CompanyProfileVersion.version.desc()).limit(1))  # fmt: skip
    if profile_row is None:
        raise BriefInputError("No company profile: run `python -m app company import`")
    rows = await session.execute(
        select(OpportunityEvidence, Competitor.slug)
        .outerjoin(Competitor, Competitor.id == OpportunityEvidence.competitor_id)
        .where(OpportunityEvidence.assessment_id == assessment.id)
        .order_by(OpportunityEvidence.id)
    )
    evidence = [EvidenceRow(e.id, e.kind, slug, e.label, e.data) for e, slug in rows]
    websites = await session.scalars(select(Competitor.website))
    return BriefInputs(
        opportunity_id=opportunity.id,
        opportunity_title=opportunity.title,
        topic=opportunity.topic_label,
        assessment_id=assessment.id,
        score=assessment.score,
        gaps=list(assessment.gaps),
        suggestion=dict(assessment.suggestion),
        signals=dict(assessment.signals),
        interpretation=assessment.interpretation
        if assessment.interpretation_status in _INTERPRETED
        else None,
        evidence=evidence,
        company_profile_id=profile_row.id,
        company_profile_version=profile_row.version,
        company=profile_row.to_profile(),
        competitor_domains=sorted({d for w in websites if (d := domain_of(w))}),
    )
