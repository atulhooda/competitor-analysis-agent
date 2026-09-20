"""Deterministic handling of generated article content: citations, checks, slugs, preview."""

import pytest

from app.domain.articles import ArticleContent, BlockType, ContentBlock, ContentSection, SectionKind
from app.prompts.article_draft import ArticleContentOut
from app.services.article_content import (
    citations,
    clean_citations,
    from_output,
    number_issues,
    render_markdown,
    slugify,
    strip_markers,
    structural_problems,
    unique_slug,
    word_count,
)
from app.services.numbers import numbers_in

WORDS = " ".join(["word"] * 40)


def article(*paragraphs: str, items: tuple[str, ...] = (), title: str = "A title") -> ArticleContent:  # fmt: skip
    blocks = [ContentBlock(type=BlockType.PARAGRAPH, text=p) for p in paragraphs]
    if items:
        blocks.append(ContentBlock(type=BlockType.LIST, items=list(items)))
    return ArticleContent(
        title=title,
        description="What the reader gets.",
        sections=[
            ContentSection(
                kind=SectionKind.INTRODUCTION,
                blocks=[ContentBlock(type=BlockType.PARAGRAPH, text="Intro.")],
            ),
            ContentSection(kind=SectionKind.BODY, heading="Body", blocks=blocks),
        ],
    )


def test_markers_are_canonical_and_unknown_labels_are_removed_and_reported() -> None:
    content = article(
        "Handoffs keep context. [S1] Most tickets are routine [S2, S3; S9].",
        items=("Scope first [S3].", "Unknown [S7]."),
    )
    cleaned, issues = clean_citations(content, {"S1", "S2", "S3"})
    body = cleaned.sections[1].blocks
    assert body[0].text == "Handoffs keep context [S1]. Most tickets are routine [S2][S3]."
    assert body[1].items == ["Scope first [S3].", "Unknown."]
    assert sorted(i.detail for i in issues) == ["[S7] doesn't match any stored source", "[S9] doesn't match any stored source"]  # fmt: skip
    assert {i.kind for i in issues} == {"unknown_citation_removed"}


def test_each_claim_is_linked_to_its_sources() -> None:
    content, _ = clean_citations(
        article(
            "First claim [S1]. An uncited sentence. Second claim [S2][S1].",
            items=("Item claim [S2].",),
        ),
        {"S1", "S2"},
    )
    found = [(c.section, c.block, c.item, c.claim, c.labels) for c in citations(content)]
    assert found == [
        (1, 0, None, "First claim.", ["S1"]),
        (1, 0, None, "Second claim.", ["S2", "S1"]),
        (1, 1, 0, "Item claim.", ["S2"]),
    ]


def test_word_count_ignores_citation_markers() -> None:
    assert word_count(article("One two three [S1].")) == 4  # "Intro" + three words
    assert strip_markers("A claim [S1][S2].") == "A claim."


def test_numbers_missing_from_the_research_are_flagged_not_removed() -> None:
    content = article("Resolution rose 64% [S1]. Costs fell 37 percent.")
    issues = number_issues(content, numbers_in("resolved 64% of routine tickets"))
    assert [i.excerpt for i in issues] == ["Costs fell 37 percent."]
    assert "37" in issues[0].detail


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        (article(WORDS * 20, title=" "), "the title is empty"),
        (article("Too short."), "words; at least 100 are required"),
        (
            ArticleContent(
                title="T",
                description="",
                sections=[
                    ContentSection(
                        kind=SectionKind.INTRODUCTION,
                        blocks=[ContentBlock(type=BlockType.PARAGRAPH, text=WORDS * 5)],
                    )
                ],
            ),
            "at least 2 sections",
        ),
        (article(WORDS * 5, "Claim [S5]."), "citation(s) S5 don't match a stored source"),
        (
            ArticleContent(
                title="T",
                description="",
                sections=[
                    ContentSection(
                        kind=SectionKind.INTRODUCTION,
                        blocks=[ContentBlock(type=BlockType.PARAGRAPH, text=WORDS * 5)],
                    ),
                    ContentSection(
                        kind=SectionKind.BODY,
                        heading=None,
                        blocks=[ContentBlock(type=BlockType.LIST, items=[])],
                    ),
                ],
            ),
            "section 2 has no heading",
        ),
    ],
)
def test_structural_checks_before_completion(content: ArticleContent, problem: str) -> None:
    problems = structural_problems(content, min_words=100, labels={"S1"})
    assert any(problem in p for p in problems), problems


def test_a_sound_article_passes_the_checks() -> None:
    assert structural_problems(article(WORDS * 5, "Claim [S1]."), min_words=100, labels={"S1"}) == []  # fmt: skip


def test_model_output_is_normalized() -> None:
    out = ArticleContentOut.model_validate(
        {
            "title": " Title ",
            "description": "D",
            "sections": [
                {"kind": "introduction", "blocks": [{"type": "paragraph", "text": "  "}]},
                {
                    "kind": "body",
                    "heading": "H",
                    "blocks": [
                        {"type": "paragraph", "text": "Kept."},
                        {"type": "list", "items": []},
                    ],
                },
            ],
        }
    )
    content = from_output(out)
    assert content.title == "Title"
    assert [s.kind for s in content.sections] == [SectionKind.BODY]  # the empty intro is dropped
    assert [b.text for b in content.sections[0].blocks] == ["Kept."]


def test_slugs_are_safe_deterministic_and_unique() -> None:
    assert slugify("AI agents: the Practical Guide (2026)!") == "ai-agents-the-practical-guide-2026"
    assert slugify("Café déjà vu") == "cafe-deja-vu"
    assert slugify("!!!") == "article"
    assert len(slugify("word " * 60)) <= 80
    assert unique_slug("guide", set()) == "guide"
    assert unique_slug("guide", {"guide", "guide-2"}) == "guide-3"


def test_markdown_preview_numbers_references_and_lists_sources() -> None:
    content, _ = clean_citations(article("Claim one [S2]. Claim two [S1].", items=("Item [S2].",)), {"S1", "S2"})  # fmt: skip
    markdown = render_markdown(content, {"S1": ("Guide", "https://a.example/g"), "S2": (None, "https://b.example/r")})  # fmt: skip
    assert markdown.startswith("# A title\n")
    assert "## Body" in markdown
    assert "Claim one [1]. Claim two [2]." in markdown
    assert "- Item [1]." in markdown
    assert markdown.rstrip().endswith("2. Guide — https://a.example/g")
    assert "1. https://b.example/r — https://b.example/r" in markdown
