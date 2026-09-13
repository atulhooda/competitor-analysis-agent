from app.services.change_detection import detect_price_change, diff_texts, extract_prices

ARTICLE = """# Guide

Customer support teams are adopting AI agents.

The biggest gains come from automating password resets and refunds.

Teams that succeed start with a narrow scope."""


def test_identical_texts() -> None:
    diff = diff_texts(ARTICLE, ARTICLE, old_title="Guide", new_title="Guide")
    assert diff.similarity == 1.0
    assert diff.words_added == diff.words_removed == 0
    assert diff.is_minor


def test_a_one_word_edit_is_minor() -> None:
    edited = ARTICLE.replace("narrow scope", "small scope")
    diff = diff_texts(ARTICLE, edited, old_title="Guide", new_title="Guide")
    assert diff.lines_added == diff.lines_removed == 1
    assert diff.is_minor


def test_a_new_section_is_significant() -> None:
    extended = ARTICLE + "\n\n## Pricing\n\n" + "Vendors now charge per resolution instead of per seat. " * 5  # fmt: skip
    diff = diff_texts(ARTICLE, extended)
    assert not diff.is_minor
    assert diff.words_added > 40
    assert diff.words_removed == 0


def test_a_title_change_is_never_minor() -> None:
    diff = diff_texts(ARTICLE, ARTICLE, old_title="Guide", new_title="The 2026 Guide")
    assert diff.title_changed
    assert not diff.is_minor


def test_extract_prices() -> None:
    text = "Starter $29/mo. Growth $1,299.00 per year. EU: €49. UK: £ 12.50. Or 99 USD. Not 2026."
    assert extract_prices(text) == ["$1,299.00", "$29", "99USD", "£12.50", "€49"]


def test_price_change_detection() -> None:
    before = "Starter costs $29 per agent. Growth costs $79 per agent."
    after = "Starter costs $35 per agent. Growth costs $79 per agent."
    assert detect_price_change(before, before) is None
    change = detect_price_change(before, after)
    assert change is not None
    assert change.removed == ["$29"]
    assert change.added == ["$35"]
    assert change.as_details()["prices_after"] == ["$35", "$79"]
