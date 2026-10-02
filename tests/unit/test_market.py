"""The company's market (country and places): written and titled for readers there, from the
topic idea to the article, its research and its search metadata. Pure functions; no LLM."""

from app.domain.company import CompanyMarket, CompanyProfile
from app.prompts import editorial
from app.prompts.article_common import company_block
from app.prompts.article_research import render_discover, render_follow_up
from app.services.article_brief import build_brief
from app.services.opportunities import company_lines
from tests.unit.test_article_brief import inputs as brief_inputs

PUNE = CompanyMarket(country="India", demonym="Indian", places=["Pune", " Kalyani  Nagar ", "Pune"], language="Indian English (British spelling)", currency="₹ (INR)")  # fmt: skip


def profile(market: CompanyMarket | None = PUNE) -> CompanyProfile:
    return CompanyProfile(name="Skin Essence", description="A dermatology clinic.", core_topics=["laser hair removal"], market=market)  # fmt: skip


def test_the_market_names_its_places_country_and_people() -> None:
    assert PUNE.places == ["Pune", "Kalyani Nagar"]
    assert PUNE.names == ("Pune", "Kalyani Nagar", "India", "Indian")
    assert CompanyMarket(country="India").names == ("India",)


def test_a_profile_without_a_market_reads_as_before() -> None:
    lines = company_lines(profile(None))
    assert not any(line.startswith("- market") for line in lines)
    # The digest a profile had before the market field existed, so stored versions still match.
    import hashlib
    import json

    old = profile(None).model_dump(mode="json", exclude={"market"})
    assert (
        profile(None).fingerprint
        == hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    )
    assert profile().fingerprint != profile(None).fingerprint


def test_topic_ideas_are_asked_for_the_market() -> None:
    lines = company_lines(profile())
    assert "- market: readers in India, above all in Pune, Kalyani Nagar (language: Indian English (British spelling); money in ₹ (INR))" in lines  # fmt: skip
    rendered = editorial.render(company=lines, excluded=[], covered=[], count=3)
    assert "readers in India, above all in Pune" in rendered
    assert (
        "the title and\nprimary keyword include it" not in editorial.SYSTEM
    )  # one rule, wrapped as a paragraph
    assert '"... in Pune"' in editorial.SYSTEM


def _brief(market: CompanyMarket | None) -> object:
    b = build_brief(brief_inputs())
    return b.model_copy(update={"company": b.company.model_copy(update={"market": market})})


def test_the_writer_is_told_who_the_readers_are() -> None:
    block = company_block(_brief(PUNE))  # type: ignore[arg-type]
    assert "Readers: people in India, above all Pune, Kalyani Nagar. Write for them:" in block
    assert "Indian English (British spelling); ₹ (INR) for any money" in block
    assert "name the country or a place (Pune, Kalyani Nagar)" in block
    assert "Readers:" not in company_block(_brief(None))  # type: ignore[arg-type]


def test_research_prefers_evidence_about_the_readers_country() -> None:
    b = _brief(PUNE)
    first = render_discover(b, max_questions=3, max_sources=5)  # type: ignore[arg-type]
    again = render_follow_up(b, questions=[], tried=[], max_sources=2)  # type: ignore[arg-type]
    for text in (first, again):
        assert "Readers are in India: prefer evidence about India" in text
    assert "Readers are in" not in render_discover(_brief(None), max_questions=3, max_sources=5)  # type: ignore[arg-type]
