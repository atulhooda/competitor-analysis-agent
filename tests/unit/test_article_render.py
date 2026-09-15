"""The CMS-neutral renderer (Phase 7): structured article → safe HTML with numbered
citations, a sources section, validated links and the FAQ."""

from html.parser import HTMLParser
from typing import Any

from app.domain.articles import ArticleContent
from app.domain.quality import FAQItem, HeadingAnalysis, ImageSuggestion, LinkSuggestion, SEOPackage
from app.services.article_render import render_article, safe_url

SOURCES = {
    "S1": ("Human handoff guidelines", "https://standards.example.org/handoff"),
    "S2": ("Evaluating AI support agents", "https://research.example.edu/agents"),
    "S3": (None, "https://docs.example.org/guides/agents"),
}
INTERNAL = "https://startup.example/blog/handoff"


def content(**overrides: Any) -> ArticleContent:
    data: dict[str, Any] = {
        "title": "AI agents for founders",
        "description": "A practical guide.",
        "sections": [
            {"kind": "introduction", "heading": None, "blocks": [
                {"type": "paragraph", "text": "AI agents resolve routine tickets [S2]. Good handoffs matter [S1][S2]."},
            ]},
            {"kind": "body", "heading": "Designing the human handoff", "blocks": [
                {"type": "subheading", "text": "Passing context"},
                {"type": "paragraph", "text": "Pass the full conversation so customers never repeat themselves [S1]."},
                {"type": "list", "ordered": True, "items": ["Scope the agent first [S3].", "Review transcripts weekly."]},
                {"type": "list", "ordered": False, "items": ["Password resets", "Order status"]},
            ]},
            {"kind": "conclusion", "heading": "Getting started", "blocks": [
                {"type": "paragraph", "text": "Read our guide to the human handoff before you start."},
            ]},
        ],
    }  # fmt: skip
    data.update(overrides)
    return ArticleContent.model_validate(data)


def seo(**overrides: Any) -> SEOPackage:
    data: dict[str, Any] = {
        "primary_keyword": "ai agents",
        "primary_keyword_evidence": ["opportunity topic"],
        "primary_keyword_reason": "topic",
        "secondary_keywords": ["human handoff"],
        "meta_title": "AI agents for founders: a practical guide",
        "meta_description": "How founders can use AI agents for support without losing customer trust.",
        "slug": "AI Agents for Founders",
        "headings": HeadingAnalysis(
            h1="AI agents for founders",
            h1_count=1,
            h2=[],
            h3=[],
            hierarchy_ok=True,
            duplicates=[],
            issues=[],
        ),
        "faq": [FAQItem(question="What can AI agents handle?", answer="Routine tickets.")],
        "internal_links": [
            LinkSuggestion(
                anchor_text="guide to the human handoff",
                url=INTERNAL,
                title="Our handoff guide",
                reason="",
            )
        ],
        "external_links": [
            LinkSuggestion(
                anchor_text="full conversation",
                url="https://standards.example.org/handoff",
                title=None,
                reason="",
            )
        ],
        "category": "AI agents",
        "tags": ["ai agents", "handoff"],
        "image": ImageSuggestion(
            concept="A founder reading a handover", purpose="illustrate", alt_text="Founder reading"
        ),
    }
    data.update(overrides)
    return SEOPackage.model_validate(data)


def render(article: ArticleContent | None = None, package: SEOPackage | None = None, **kwargs: Any) -> Any:  # fmt: skip
    return render_article(article or content(), sources=kwargs.pop("sources", SOURCES), seo=package if package is not None else seo(), allowed_internal={INTERNAL}, allowed_external={url for _, url in SOURCES.values()}, **kwargs)  # fmt: skip


class _Tags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.attrs: list[tuple[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        self.attrs += attrs


def parsed(markup: str) -> _Tags:
    parser = _Tags()
    parser.feed(markup)
    return parser


def test_structure_is_rendered_as_html() -> None:
    doc = render()
    body = doc.body_html
    assert "<h1>" not in body  # the CMS shows the title
    assert "<h2>Designing the human handoff</h2>" in body
    assert "<h3>Passing context</h3>" in body
    assert "<ol><li>Scope the agent first" in body
    assert "<ul><li>Password resets</li><li>Order status</li></ul>" in body
    assert body.count("<p>") >= 3
    assert "<h1>AI agents for founders</h1>" in render(include_title=True).body_html
    assert set(parsed(body).tags) <= {"h2", "h3", "p", "ol", "ul", "li", "sup", "a"}


def test_citations_become_numbered_references_with_a_sources_section() -> None:
    doc = render()
    body = doc.body_html
    assert 'AI agents resolve routine tickets<sup class="citation"><a href="#source-1">[1]</a></sup>.' in body  # fmt: skip
    assert '<sup class="citation"><a href="#source-2">[2]</a><a href="#source-1">[1]</a></sup>' in body  # fmt: skip
    assert "[S1]" not in body
    assert "[S2]" not in body
    assert '<h2 id="sources">Sources</h2>' in body
    assert [(s.number, s.url) for s in doc.sources] == [(1, "https://research.example.edu/agents"), (2, "https://standards.example.org/handoff"), (3, "https://docs.example.org/guides/agents")]  # fmt: skip
    assert '<li id="source-3"><a href="https://docs.example.org/guides/agents" rel="noopener">docs.example.org</a></li>' in body  # untitled: the domain  # fmt: skip


def test_only_sources_this_version_cites_are_listed_and_unknown_labels_are_dropped() -> None:
    article = content(sections=[{"kind": "body", "heading": "One", "blocks": [{"type": "paragraph", "text": "A claim [S9]. Another [S1]."}]}])  # fmt: skip
    doc = render(article)
    assert doc.unknown_citations == ["S9"]
    assert [s.number for s in doc.sources] == [1]
    assert "S9" not in doc.body_html
    assert any("S9" in n for n in doc.notes)


def test_validated_links_are_placed_once_and_nothing_else_is_linked() -> None:
    doc = render()
    body = doc.body_html
    assert f'<a href="{INTERNAL}">guide to the human handoff</a>' in body
    assert '<a href="https://standards.example.org/handoff" rel="noopener">full conversation</a>' in body  # fmt: skip
    assert body.count(f'href="{INTERNAL}"') == 1
    assert all(link.placed for link in doc.links)
    assert "Related reading" not in body


def test_links_to_unvalidated_targets_are_left_out() -> None:
    package = seo(
        internal_links=[
            LinkSuggestion(
                anchor_text="Getting started",
                url="https://evil.example.net/phish",
                title=None,
                reason="",
            )
        ],
        external_links=[
            LinkSuggestion(
                anchor_text="routine tickets", url="javascript:alert(1)", title=None, reason=""
            )
        ],
    )
    doc = render(package=package)
    assert "evil.example.net" not in doc.body_html
    assert "javascript:" not in doc.body_html
    assert len([n for n in doc.notes if "not a validated target" in n]) == 2
    assert doc.links == []


def test_internal_links_that_dont_fit_go_to_related_reading() -> None:
    package = seo(internal_links=[LinkSuggestion(anchor_text="an anchor that isn't in the text", url=INTERNAL, title="Our handoff guide", reason="")])  # fmt: skip
    body = render(package=package).body_html
    assert f'<h2>Related reading</h2>\n<ul><li><a href="{INTERNAL}">Our handoff guide</a></li></ul>' in body  # fmt: skip


def test_faq_is_rendered() -> None:
    body = render().body_html
    assert "<h2>Frequently asked questions</h2>\n<h3>What can AI agents handle?</h3>\n<p>Routine tickets.</p>" in body  # fmt: skip
    assert "Frequently asked questions" not in render(package=seo(faq=[])).body_html


def test_every_piece_of_text_is_escaped() -> None:
    hostile = '<script>alert("x")</script> <img src=x onerror=alert(1)> "quotes" & ampersands'
    article = content(
        title=hostile,
        sections=[
            {"kind": "body", "heading": f"Heading {hostile}", "blocks": [
                {"type": "subheading", "text": hostile},
                {"type": "paragraph", "text": f"{hostile} [S1]."},
                {"type": "list", "ordered": False, "items": [hostile]},
            ]},
        ],
    )  # fmt: skip
    package = seo(faq=[FAQItem(question=hostile, answer=hostile)], internal_links=[], external_links=[])  # fmt: skip
    doc = render(article, package, include_title=True, sources={"S1": ('<b onclick="x">T</b>', "https://standards.example.org/handoff")})  # fmt: skip
    body = doc.body_html
    assert "<script" not in body
    assert "<img" not in body
    assert "<b " not in body
    assert "&lt;script&gt;" in body
    assert "&quot;quotes&quot; &amp; ampersands" in body
    tags = parsed(body)
    assert set(tags.tags) <= {"h1", "h2", "h3", "p", "ul", "ol", "li", "sup", "a"}
    assert not [name for name, _ in tags.attrs if name.startswith("on")]


def test_unsafe_source_urls_are_never_linked() -> None:
    doc = render(sources={"S1": ("x", "javascript:alert(1)"), "S2": ("y", "https://user:pw@example.org/")})  # fmt: skip
    assert "javascript:" not in doc.body_html
    assert "user:pw" not in doc.body_html
    assert {"S1", "S2"} <= set(doc.unknown_citations)


def test_metadata_carries_the_seo_package_and_image_suggestion() -> None:
    doc = render()
    assert doc.slug == "ai-agents-for-founders"
    assert doc.excerpt.startswith("How founders can use AI agents")
    assert doc.meta_title == "AI agents for founders: a practical guide"
    assert doc.category == "AI agents"
    assert doc.tags == ["ai agents", "handoff"]
    assert doc.image is not None
    assert doc.image.alt_text == "Founder reading"
    assert "<img" not in doc.body_html  # no image is invented
    assert doc.word_count > 20


def test_rendering_is_deterministic() -> None:
    assert render().content_hash == render().content_hash
    assert render().content_hash != render(package=seo(meta_description="A different description for the page.")).content_hash  # fmt: skip


def test_malformed_blocks_are_skipped() -> None:
    article = content(sections=[{"kind": "body", "heading": "One", "blocks": [{"type": "paragraph", "text": "   "}, {"type": "list", "ordered": False, "items": ["", "Kept"]}, {"type": "paragraph", "text": "Fine."}]}])  # fmt: skip
    body = render(article, seo(faq=[], internal_links=[], external_links=[])).body_html
    assert "<p></p>" not in body
    assert "<li></li>" not in body
    assert "<ul><li>Kept</li></ul>" in body


def test_safe_url() -> None:
    assert safe_url("https://example.org/a?b=c") == "https://example.org/a?b=c"
    for bad in ("javascript:alert(1)", "data:text/html,x", "ftp://example.org", "https://u:p@example.org", "//example.org", "not a url"):  # fmt: skip
        assert safe_url(bad) is None
