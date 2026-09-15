import pytest

from app.services.numbers import numbers_in, strip_unverified, title_is_verified


def test_numbers_are_normalized() -> None:
    assert numbers_in("Growth of +62% over 1,200 pages; score 84.0 and 0.50") == {"62", "1200", "84", "0.5"}  # fmt: skip
    assert numbers_in("SOC 2 and GPT-4, published 2026-09-10") == {"2", "4", "2026", "9", "10"}
    assert numbers_in("no digits here") == set()


def test_sentences_with_unverified_numbers_are_removed() -> None:
    allowed = numbers_in("6 recent items vs 2 before; score 72.4")
    text = "Activity tripled from 2 to 6 items. Competitors grew 987% last week. Score 72.4 reflects it!"  # fmt: skip
    cleaned, removed = strip_unverified(text, allowed)
    assert cleaned == "Activity tripled from 2 to 6 items. Score 72.4 reflects it!"
    assert removed == 1
    assert strip_unverified("Only words here.", set()) == ("Only words here.", 0)


@pytest.mark.parametrize(
    ("title", "verified"),
    [
        ("7 ways to evaluate AI agents", True),  # a list count
        ("The 2026 guide to AI agents", True),  # a year
        ("AI agents: 6 new posts in 60 days", True),  # evidence numbers
        ("Why 73% of teams fail at AI agents", False),  # an invented statistic
        ("7% of founders get this wrong", False),  # small, but a percentage
        ("AI agents cut support costs 3x", False),  # a multiple
        ("AI agents: 5\u00d7 faster answers", False),  # a multiple, with the times sign
        ("1 checklist for AI agents", True),
        ("250 AI agent use cases", False),  # a count not in the evidence
        ("AI agents for founders", True),
    ],
)
def test_titles_may_not_invent_statistics(title: str, verified: bool) -> None:
    assert title_is_verified(title, numbers_in("6 recent items; 60-day windows")) is verified
