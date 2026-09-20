"""The SEO package (Phase 6): candidates from stored data, validation of Gemini's choices, and
the deterministic checks."""

from typing import Any

import pytest

from app.domain.articles import ArticleBrief, ArticleContent, SourceType
from app.prompts.seo import FAQOut, ImageOut, LinkChoiceOut, SEOOut
from app.services.article_brief import build_brief
from app.services.seo import (
    ExternalSource,
    InternalPage,
    SEOConfig,
    SEOInputs,
    accept_keyword,
    assemble,
    category_options,
    contains_keyword,
    external_candidates,
    heading_analysis,
    internal_candidates,
    keyword_candidates,
    keyword_density,
)
from tests.unit.test_article_brief import inputs as brief_inputs

CONFIG = SEOConfig(model="fake", reasoning_effort="low", title_max=60, description_min=70, description_max=160, max_density=0.03)  # fmt: skip
SOURCES = (
    ExternalSource(
        "S1",
        "https://standards.example.org/handoff",
        "Human handoff guidelines",
        SourceType.ORGANIZATION,
        2,
    ),
    ExternalSource(
        "S2", "https://acme.test/blog/agents", "Acme on agents", SourceType.COMPETITOR, 1
    ),
    ExternalSource("S3", "https://startup.example/blog", "Our blog", SourceType.COMPANY, 1),
)


def brief() -> ArticleBrief:
    return build_brief(brief_inputs())


def article(**overrides: Any) -> ArticleContent:
    data: dict[str, Any] = {
        "title": "AI agents for founders: a practical guide",
        "description": "How founders deploy AI agents.",
        "sections": [
            {
                "kind": "introduction",
                "heading": None,
                "blocks": [
                    {
                        "type": "paragraph",
                        "text": "AI agents can take routine tickets off a small team's plate [S1]. Start small.",
                    }
                ],
            },
            {
                "kind": "body",
                "heading": "What AI agents handle well",
                "blocks": [
                    {"type": "paragraph", "text": "Password resets and order status questions."}
                ],
            },
            {
                "kind": "body",
                "heading": "Designing the human handoff",
                "blocks": [
                    {"type": "subheading", "text": "Passing context"},
                    {"type": "paragraph", "text": "Pass the whole conversation."},
                ],
            },
            {
                "kind": "conclusion",
                "heading": "Getting started",
                "blocks": [{"type": "paragraph", "text": "Pick one queue this week."}],
            },
        ],
    }
    data.update(overrides)
    return ArticleContent.model_validate(data)


def inputs(**overrides: Any) -> SEOInputs:
    values: dict[str, Any] = {
        "brief": brief(),
        "subtopics": ("Human handoff", "Ticket automation"),
        "competitor_keywords": ("ai support agents", "ticket automation"),
        "company_topics": ("AI agents", "Automation"),
        "internal_pages": (
            InternalPage(
                "https://startup.example/blog/handoff",
                "How our agents hand over to people",
                ("Human handoff",),
            ),
            InternalPage("https://startup.example/careers", "Careers", ("Open roles",)),
        ),
        "sources": SOURCES,
    }
    values.update(overrides)
    return SEOInputs(**values)


def out(**overrides: Any) -> SEOOut:
    values: dict[str, Any] = {
        "primary_keyword": "AI agents",
        "primary_keyword_reason": "the topic",
        "secondary_keywords": ["human handoff", "ticket automation", "blockchain growth hacks"],
        "meta_title": "AI agents for founders: a practical guide",
        "meta_description": "How founders can use AI agents for support without losing customer trust, with a clear human handoff and a weekly review.",
        "slug": "AI Agents for Founders!",
        "faq": [
            FAQOut(question="What can AI agents handle?", answer="Routine tickets."),
            FAQOut(question="How big is the gain?", answer="Teams save 40% of their time."),
            FAQOut(question="Where to start?", answer="Pick one queue."),
            FAQOut(question="Who reviews?", answer="The team, weekly."),
        ],
        "internal_links": [
            LinkChoiceOut(candidate="L1", anchor_text="how our agents hand over", reason="related"),
            LinkChoiceOut(candidate="L7", anchor_text="made up", reason=""),
        ],
        "external_links": [
            LinkChoiceOut(candidate="X1", anchor_text="handoff guidelines", reason="the standard")
        ],
        "category": "automation",
        "tags": ["AI agents", "handoff", "crypto"],
        "image": ImageOut(
            concept="A founder reading a handed-over conversation",
            purpose="show the handoff",
            alt_text="x" * 200,
        ),
    }
    values.update(overrides)
    return SEOOut.model_validate(values)


def build(content: ArticleContent | None = None, **overrides: Any) -> Any:
    content = content or article()
    data = inputs()
    candidates = keyword_candidates(data, content)
    internal = internal_candidates(data.internal_pages, data, content)
    return assemble(out(**overrides), candidates, internal, external_candidates(data.sources), category_options(data), content, CONFIG)  # fmt: skip


# ── headings and keywords ────────────────────────────────────────────────────


def test_heading_analysis() -> None:
    analysis = heading_analysis(article())
    assert analysis.h1_count == 1
    assert analysis.h2 == ["What AI agents handle well", "Designing the human handoff", "Getting started"]  # fmt: skip
    assert analysis.h3 == ["Passing context"]
    assert analysis.hierarchy_ok
    assert analysis.duplicates == []
    broken = article(sections=[
        {"kind": "introduction", "heading": None, "blocks": [{"type": "subheading", "text": "Orphan"}, {"type": "paragraph", "text": "Text."}]},
        {"kind": "body", "heading": "Same heading", "blocks": [{"type": "paragraph", "text": "Text."}]},
        {"kind": "conclusion", "heading": "Same Heading", "blocks": [{"type": "paragraph", "text": "Text."}]},
    ])  # fmt: skip
    analysis = heading_analysis(broken)
    assert not analysis.hierarchy_ok
    assert analysis.duplicates == ["Same Heading"]
    assert any("has no H2 above it" in i for i in analysis.issues)


def test_keyword_candidates_carry_their_provenance() -> None:
    candidates = keyword_candidates(inputs(), article())
    top = candidates[0]
    assert top.keyword.casefold() == "ai agents"
    assert "opportunity topic" in top.sources
    assert "company topic" in top.sources
    by_keyword = {c.keyword.casefold(): c for c in candidates}
    assert by_keyword["ticket automation"].sources == ["competitor subtopic", "competitor keyword"]
    assert [c.score for c in candidates] == sorted((c.score for c in candidates), reverse=True)


def test_repeated_competitor_keywords_cannot_outweigh_the_topic() -> None:
    plain = article(title="A practical guide for founders")  # no keyword in the title
    candidates = keyword_candidates(inputs(competitor_keywords=("ticket automation",) * 40), plain)  # fmt: skip
    by_keyword = {c.keyword.casefold(): c.score for c in candidates}
    assert by_keyword["ai agents"] > by_keyword["ticket automation"]
    once = {c.keyword.casefold(): c.score for c in keyword_candidates(inputs(competitor_keywords=("ticket automation",)), plain)}  # fmt: skip
    assert by_keyword["ticket automation"] == once["ticket automation"] + 1  # the frequency bonus is capped  # fmt: skip


@pytest.mark.parametrize(
    ("keyword", "accepted"),
    [
        ("AI agents", True),  # a candidate
        ("ai agent", True),  # another form of it
        ("human handoff agents", True),  # reworded from candidate and heading words
        ("blockchain growth hacks", False),  # invented
        ("", False),
    ],
)
def test_keywords_must_come_from_the_candidates(keyword: str, accepted: bool) -> None:
    candidates = keyword_candidates(inputs(), article())
    result, evidence = accept_keyword(keyword, candidates, article())
    assert (result is not None) is accepted
    assert bool(evidence) is accepted


def test_contains_keyword_ignores_order_and_inflection() -> None:
    assert contains_keyword("The agents that support teams use", "support agent")
    assert not contains_keyword("The agents", "support agents")


def test_keyword_density_counts_whole_phrases() -> None:
    stuffed = article(sections=[{"kind": "body", "heading": "AI agents", "blocks": [{"type": "paragraph", "text": "AI agents AI agents AI agents help. AI agents again."}]}])  # fmt: skip
    assert (
        keyword_density(stuffed, "ai agents") == 0.4
    )  # 4 occurrences in 10 words (headings aside)


# ── links ────────────────────────────────────────────────────────────────────


def test_internal_links_only_to_related_stored_pages() -> None:
    pages = internal_candidates(inputs().internal_pages, inputs(), article())
    assert [p.url for p in pages] == [
        "https://startup.example/blog/handoff"
    ]  # careers isn't related


def test_external_links_only_to_research_sources_never_competitors() -> None:
    assert [s.label for s in external_candidates(SOURCES)] == ["S1"]


# ── the package ──────────────────────────────────────────────────────────────


def test_the_package_keeps_only_what_the_stored_data_supports() -> None:
    report = build()
    package = report.package
    assert package.primary_keyword == "AI agents"
    assert "opportunity topic" in package.primary_keyword_evidence
    assert "blockchain growth hacks" not in package.secondary_keywords
    assert len(package.secondary_keywords) >= 3
    assert package.slug == "ai-agents-for-founders"
    assert [link.url for link in package.internal_links] == ["https://startup.example/blog/handoff"]
    assert [link.url for link in package.external_links] == ["https://standards.example.org/handoff"]  # fmt: skip
    assert [f.question for f in package.faq] == ["What can AI agents handle?", "Where to start?", "Who reviews?"]  # the 40% answer is dropped  # fmt: skip
    assert package.category == "Automation"
    assert "crypto" not in package.tags
    assert package.image is not None
    assert len(package.image.alt_text) <= 125
    notes = " ".join(report.notes)
    for expected in ("blockchain growth hacks", "L7", "FAQ answer dropped", "alt text shortened"):
        assert expected in notes
    assert report.mandatory_missing == []
    assert 0 < report.score <= 1


def test_an_invented_primary_keyword_is_replaced_by_the_top_candidate() -> None:
    report = build(primary_keyword="quantum knitting", slug="")
    assert report.package.primary_keyword.casefold() == "ai agents"
    assert "rejected" in report.package.primary_keyword_reason
    assert report.package.slug == "ai-agents"


def test_an_unknown_category_falls_back_to_the_first_option() -> None:
    report = build(category="Gardening")
    assert report.package.category == brief().topic
    assert any("Gardening" in n for n in report.notes)


def test_checks_flag_lengths_placement_and_stuffing() -> None:
    long_title = "AI agents " + "for founders " * 10
    report = build(meta_title=long_title, meta_description="Too short.")
    checks = {c.name: c.passed for c in report.checks}
    assert not checks["meta_title_length"]
    assert not checks["meta_description_length"]
    assert checks["keyword_in_h1"]
    assert checks["keyword_in_introduction"]
    assert checks["keyword_in_h2"]
    assert checks["external_links"]
    stuffed = article(sections=[{"kind": "introduction", "heading": None, "blocks": [{"type": "paragraph", "text": "AI agents AI agents AI agents help teams. AI agents again and again."}]}, {"kind": "body", "heading": "AI agents", "blocks": [{"type": "paragraph", "text": "More text about support work."}]}, {"kind": "conclusion", "heading": "Next", "blocks": [{"type": "paragraph", "text": "Done."}]}])  # fmt: skip
    assert not {c.name: c.passed for c in build(stuffed).checks}["no_keyword_stuffing"]


def test_missing_mandatory_fields_are_reported() -> None:
    report = build(meta_description="   ", meta_title="")
    assert report.mandatory_missing == ["meta_title", "meta_description"]


def test_the_inputs_fingerprint_follows_the_stored_data() -> None:
    assert inputs().fingerprint() == inputs().fingerprint()
    assert inputs().fingerprint() != inputs(competitor_keywords=("other",)).fingerprint()
    assert inputs().fingerprint() != inputs(internal_pages=()).fingerprint()
