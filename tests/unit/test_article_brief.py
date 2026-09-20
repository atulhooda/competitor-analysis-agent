"""The article brief is built deterministically from stored opportunity data (no LLM)."""

from typing import Any

from app.domain.analysis import ContentFormat, SearchIntent
from app.domain.company import CompanyProfile
from app.services.article_brief import BriefInputs, EvidenceRow, build_brief

COMPANY = CompanyProfile.model_validate(
    {
        "name": "Example Startup",
        "website": "https://startup.example",
        "description": "Helps founders deploy AI agents.",
        "products": [{"name": "Agent desk", "description": "An AI help desk."}],
        "target_audiences": ["founders"],
        "core_topics": ["AI agents"],
        "excluded_topics": ["Pricing"],
        "preferred_formats": ["tutorial"],
        "tone": "Plain",
    }
)
GAPS = [
    {
        "type": "freshness",
        "score": 0.8,
        "detail": "median competitor page is 400 days old",
        "data": {},
    },
    {
        "type": "depth",
        "score": 0.5,
        "detail": "2 of 3 subtopics touched by a single page",
        "data": {"thin_subtopics": ["Evaluation", "Costs"]},
    },
    {"type": "topic", "score": 0.1, "detail": "covered by 2 of 2 competitors", "data": {}},
]
EVIDENCE = [
    EvidenceRow(
        7,
        "content",
        "acme",
        "Page",
        {
            "url": "https://acme.test/b",
            "title": "B",
            "published_at": "2026-08-01T00:00:00+00:00",
            "content_format": "article",
            "summary": "S",
            "primary_angle": "A",
        },
    ),
    EvidenceRow(
        3,
        "content",
        "acme",
        "Page",
        {
            "url": "https://acme.test/a",
            "title": "A",
            "published_at": None,
            "content_format": "guide",
        },
    ),
    EvidenceRow(
        9,
        "competitor_profile",
        "acme",
        "acme profile v1",
        {"positioning_statement": "AI-first support."},
    ),
    EvidenceRow(10, "topic_metrics", None, "metrics", {}),
]


def inputs(**overrides: Any) -> BriefInputs:
    values: dict[str, Any] = {
        "opportunity_id": 1,
        "opportunity_title": "AI agents",
        "topic": "AI agents",
        "assessment_id": 5,
        "score": 72.2,
        "gaps": GAPS,
        "suggestion": {
            "format": "comparison",
            "audience": "founders",
            "intent": None,
            "primary_gap": "freshness",
            "reasons": ["topic growth: +50%"],
        },
        "signals": {
            "subtopics": [
                {"name": "Handoff", "items": 4},
                {"name": "Evaluation", "items": 1},
                {"name": "Costs", "items": 1},
            ]
        },
        "interpretation": None,
        "evidence": EVIDENCE,
        "company_profile_id": 2,
        "company_profile_version": 1,
        "company": COMPANY,
        "competitor_domains": ["acme.test", "acme.test"],
    }
    values.update(overrides)
    return BriefInputs(**values)


def test_the_same_inputs_give_the_same_brief() -> None:
    assert build_brief(inputs()) == build_brief(inputs())
    assert build_brief(inputs()).model_dump_json() == build_brief(inputs()).model_dump_json()


def test_without_an_interpretation_the_brief_uses_the_deterministic_suggestion() -> None:
    brief = build_brief(inputs())
    assert brief.working_title == "AI agents"
    assert (brief.target_audience, brief.content_type, brief.search_intent) == ("founders", ContentFormat.COMPARISON, SearchIntent.INFORMATIONAL)  # fmt: skip
    assert brief.primary_angle == "An up-to-date take on AI agents, where most coverage is dated."
    assert brief.why_now == "topic growth: +50%"
    assert brief.provenance["primary_angle"] == "template (primary gap)"
    assert brief.provenance["content_type"] == "suggestion"
    assert brief.provenance["search_intent"] == "default"
    assert brief.competitor_weaknesses == [
        "Coverage is dated: median competitor page is 400 days old",
        "Coverage is shallow or fragmented: 2 of 3 subtopics touched by a single page",
    ]  # the 0.1 topic gap is too weak to mention
    assert "Cover Handoff." in brief.key_points
    assert "Go deeper on Costs: competitors touch it only briefly." in brief.key_points
    assert brief.desired_outcome.startswith("Founders understand AI agents")


def test_an_interpretation_takes_precedence_and_is_credited() -> None:
    interpretation = {"title": "AI agents, compared", "recommended_angle": "Compare the options honestly.", "why_now": "Two competitors started covering it.", "target_audience": "support leads", "recommended_format": "guide", "search_intent": "commercial", "differentiation_strategy": "Show real trade-offs."}  # fmt: skip
    brief = build_brief(inputs(interpretation=interpretation))
    assert (brief.working_title, brief.primary_angle, brief.target_audience) == ("AI agents, compared", "Compare the options honestly.", "support leads")  # fmt: skip
    assert (brief.content_type, brief.search_intent) == (ContentFormat.GUIDE, SearchIntent.COMMERCIAL)  # fmt: skip
    assert brief.differentiation_strategy == "Show real trade-offs."
    assert {brief.provenance[k] for k in ("working_title", "primary_angle", "target_audience", "differentiation_strategy")} == {"interpretation"}  # fmt: skip
    assert "Agent desk" in brief.desired_outcome  # commercial intent names the product


def test_evidence_company_and_things_to_avoid_are_carried_over() -> None:
    brief = build_brief(inputs())
    assert [e.evidence_id for e in brief.evidence] == [3, 7]  # content rows only, in id order
    assert brief.evidence[1].published == "2026-08-01"
    assert brief.competitor_positioning == ["acme: AI-first support."]
    assert brief.competitor_domains == ["acme.test"]
    assert brief.company.products == ["Agent desk: An AI help desk."]
    assert brief.company.tone == "Plain"
    assert "Excluded topics: Pricing." in brief.things_to_avoid
    assert any("Copying" in item for item in brief.things_to_avoid)
    assert (brief.opportunity_id, brief.assessment_id, brief.company_profile_version) == (1, 5, 1)
