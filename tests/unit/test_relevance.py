import pytest

from app.domain.company import CompanyProfile
from app.services.relevance import (
    CONTAINS,
    EXACT,
    OVERLAPS,
    audience_matches,
    match_strength,
    similar,
    stems,
    strategic_fit,
)


def company(**overrides: object) -> CompanyProfile:
    values: dict[str, object] = {
        "name": "Example",
        "description": "Workflow software for operations teams.",
        "products": ["Flowbot"],
        "target_audiences": ["SaaS founders", "developers"],
        "core_topics": ["AI", "Workflow automation"],
        "adjacent_topics": ["Customer support"],
        "excluded_topics": ["Politics"],
    }
    values.update(overrides)
    return CompanyProfile.model_validate(values)


def test_stems_ignore_filler_and_word_endings() -> None:
    assert stems("AI-powered workflow automation") == {"ai", "workf", "autom"}
    assert stems("Automating AI workflows") == {"autom", "ai", "workf"}
    assert stems("Marketing teams") == {"marke"}


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("AI agents", "ai-agents", EXACT),
        ("Workflow automation", "Automating workflows", EXACT),
        ("AI", "AI agents", CONTAINS),
        ("Customer support", "Support", CONTAINS),
        ("Data privacy compliance", "Privacy regulation compliance", OVERLAPS),
        ("Pricing", "Billing", 0.0),
    ],
)
def test_match_strength(a: str, b: str, expected: float) -> None:
    assert match_strength(a, b) == expected


def test_near_duplicate_topics() -> None:
    same = ["AI workflow automation", "Automating AI workflows", "AI-powered workflow automation"]
    assert all(similar([a], [b]) for a in same for b in same)
    assert not similar(["Workflow automation"], ["AI workflow automation"])  # narrower scope
    assert not similar(["AI"], ["AI agents"])
    assert similar(["Customer support"], ["Help desk", "customer-support"])  # via an alias


def test_audience_matching() -> None:
    assert audience_matches("SaaS founders", "founders")
    assert audience_matches("marketers", "marketing teams")
    assert not audience_matches("developers", "engineering leaders")


def test_strategic_fit_rules() -> None:
    profile = company()
    assert strategic_fit(["Workflow automation"], profile).value == EXACT
    assert strategic_fit(["AI agents"], profile).value == CONTAINS  # contains core topic "AI"
    assert strategic_fit(["Help desk", "Customer support"], profile).value == pytest.approx(
        0.6
    )  # adjacent
    fit = strategic_fit(["Hiring"], profile, subtopic_labels=["AI screening"])
    assert fit.value == 0.5
    assert "subtopic" in fit.matches[0]
    assert strategic_fit(["Operations"], profile).value == 0.35  # in the description
    assert strategic_fit(["Gardening"], profile).value == 0.0


def test_keyword_support_adds_up_to_a_fifth() -> None:
    fit = strategic_fit(["Gardening"], company(), item_terms=[["AI tools"], ["soil"], ["AI planning"], ["seeds"]])  # fmt: skip
    assert fit.value == pytest.approx(0.1)
    assert "2 of 4 competitor pages" in fit.matches[0]


def test_excluded_topics_win() -> None:
    fit = strategic_fit(["Political campaigns"], company())
    assert fit.value == 0.0
    assert fit.excluded_by == "Politics"
