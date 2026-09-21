"""The site's content contract (Phase 8): MDX-safe escaping, the Markdown body, slugs,
categories, tags, frontmatter, link filtering, the CTA and byline, validation and the pull
request description. Pure functions; no network."""

from datetime import UTC, date, datetime
from typing import Any

import pytest
import yaml

from app.cms.github.mdx import (
    AUTOMATIC_CATEGORIES,
    CATEGORIES,
    SiteConfig,
    compose,
    cover_credit,
    filter_internal_links,
    frontmatter_text,
    jsx_attr,
    pr_body,
    reading_minutes,
    site_category,
    site_slug,
    site_tags,
    split_frontmatter,
    validate,
)
from app.domain.articles import ArticleContent, BlockType, ContentBlock, ContentSection, SectionKind
from app.domain.publishing import (
    RenderedCover,
    RenderedDocument,
    RenderedFAQ,
    RenderedLink,
    RenderedSource,
)
from app.services.article_markdown import MarkdownLink, mdx_escape, render_markdown_body

MARKER = "0123456789abcdef0123456789abcdef"
SITE = "https://www.engageoagency.com"


def config(**overrides: Any) -> SiteConfig:
    values: dict[str, Any] = {
        "site_url": SITE, "content_dir": "src/content/blog", "branch_prefix": "blog/",
        "author_name": "Engageo Team", "author_role": "AI Content", "author_initials": "EN", "author_linkedin": None,
        "cta_title": "See Engageo in action", "cta_body": 'Say "hello" to <5% no-shows & {real} results', "cta_label": "Book a demo", "cta_href": "/contact?intent=demo",
        "byline": "The Engageo Team builds *AI* for clinics.",
    }  # fmt: skip
    values.update(overrides)
    return SiteConfig(**values)


def document(**overrides: Any) -> RenderedDocument:
    values: dict[str, Any] = {
        "render_version": "render/1", "title": "Why Clinics Miss Calls: The 2026 Data", "slug": "Why Clinics Miss Calls -- The 2026 Data!",
        "excerpt": "A data-led look at missed calls.", "meta_title": "Missed calls", "primary_keyword": "missed calls",
        "category": "Industry Data (SEO)", "tags": ["Missed Calls", "clinics"],
        "body_html": "<p>x</p>\n",
        "body_markdown": "Intro ([Acme](https://acme.test/r)).\n\n## The numbers\n\nSee [our pricing](https://www.engageoagency.com/pricing/) and [the guide](https://www.engageoagency.com/blog/no-such-post) and [Acme](https://acme.test/r).\n",
        "headings": ["The numbers"], "secondary_keywords": ["patient calls", "clinic revenue"],
        "sources": [RenderedSource(number=1, title="Acme", url="https://acme.test/r")],
        "faq": [], "links": [RenderedLink(kind="internal", anchor_text="clinic playbook", url="https://www.engageoagency.com/blog/existing-post", placed=False), RenderedLink(kind="internal", anchor_text="nowhere", url="https://www.engageoagency.com/nowhere", placed=False)],
        "image": None, "word_count": 1200, "content_hash": "c" * 64,
        "article_id": 3, "version_id": 9, "content_type": "research", "quality_score": 88.0, "opportunity_title": "Missed calls cost",
    }  # fmt: skip
    values.update(overrides)
    return RenderedDocument(**values)


ALLOWED = {"/", "/pricing", "/contact", "/blog", "/blog/existing-post"}


# ── escaping and the body ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "escaped"),
    [
        ("Cost < 5% and {x} > y", "Cost &lt; 5% and &#123;x&#125; &gt; y"),
        ("import os", "&#105;mport os"),
        ("export default", "&#101;xport default"),
        ("1. first", "1\\. first"),
        ("- dash", "\\- dash"),
        ("# heading", "\\# heading"),
        ("> quote", "&gt; quote"),  # no blockquote: the bracket became a reference first
        ("[a](https://x.test) and ![i](u)", "\\[a\\](https&#58;//x.test) and !\\[i\\](u)"),
        (
            "go to https://evil.test/x or www.evil.test",
            "go to https&#58;//evil.test/x or www&#46;evil.test",
        ),
        ("mail me@x.test", "mail me&#64;x.test"),
        ("AT&T &amp; co", "AT&T &amp;amp; co"),
        ("`code` and *em* and _u_ and ~s~", "\\`code\\` and \\*em\\* and \\_u\\_ and \\~s\\~"),
        ("back\\slash", "back\\\\slash"),
    ],
)
def test_prose_can_never_become_markup(raw: str, escaped: str) -> None:
    assert mdx_escape(raw) == escaped


def test_the_body_keeps_the_house_structure() -> None:
    content = ArticleContent(
        title="T", description="D",
        sections=[
            ContentSection(kind=SectionKind.INTRODUCTION, heading="Introduction", blocks=[ContentBlock(type=BlockType.PARAGRAPH, text="Calls go unanswered after 7 PM [S1]. It costs money [S1][S2].")]),
            ContentSection(kind=SectionKind.BODY, heading="What it costs", blocks=[ContentBlock(type=BlockType.SUBHEADING, text="Per clinic"), ContentBlock(type=BlockType.LIST, items=["Lost bookings [S9]", "Staff time"], ordered=False), ContentBlock(type=BlockType.PARAGRAPH, text="Read our clinic playbook first.")]),
            ContentSection(kind=SectionKind.CONCLUSION, heading="What to do", blocks=[ContentBlock(type=BlockType.LIST, items=["Answer fast", "Follow up"], ordered=True)]),
        ],
    )  # fmt: skip
    links = [MarkdownLink("internal", "clinic playbook", "https://www.engageoagency.com/blog/existing-post", "Playbook")]  # fmt: skip
    body = render_markdown_body(content, sources={"S1": ("Acme report", "https://acme.test/r"), "S2": (None, "https://b.test/x y")}, links=links, faq=[RenderedFAQ(question="Is it worth it?", answer="Yes: 5 < 8.")])  # fmt: skip
    assert body.text.startswith(
        "Calls go unanswered after 7 PM ([Acme report](https://acme.test/r))."
    )
    assert "It costs money ([Acme report](https://acme.test/r), [b.test](https://b.test/x%20y))." in body.text  # fmt: skip
    assert (
        "\n## What it costs\n\n### Per clinic\n\n- Lost bookings\n- Staff time\n" in body.text
    )  # S9 dropped
    assert "[clinic playbook](https://www.engageoagency.com/blog/existing-post)" in body.text
    assert "\n1. Answer fast\n2. Follow up\n" in body.text
    assert "## Frequently asked questions\n\n### Is it worth it?\n\nYes: 5 &lt; 8." in body.text
    assert "# T" not in body.text
    assert "Introduction" not in body.text
    assert body.unknown_citations == ["S9"]
    assert body.headings == ["What it costs", "What to do", "Frequently asked questions"]
    assert links[0].placed


# ── slug, category, tags ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "slug"),
    [
        ("Why Clinics Miss Calls -- The 2026 Data!", "why-clinics-miss-calls-the-2026-data"),
        ("  --Leading_and_trailing--  ", "leading-and-trailing"),
        ("Café résumé naïve", "cafe-resume-naive"),
        ("a" * 100, "a" * 80),
        ("ai-receptionist-indian-clinics-guide", "ai-receptionist-indian-clinics-guide"),
    ],
)
def test_slugs_follow_the_site_rules(raw: str, slug: str) -> None:
    out = site_slug(raw)
    assert out == slug
    assert out == out.lower()
    assert "--" not in out
    assert not out.startswith("-")
    assert not out.endswith("-")


@pytest.mark.parametrize(
    ("content_type", "category"),
    [
        ("guide", "Playbook"),
        ("tutorial", "Playbook"),
        ("listicle", "Playbook"),
        ("article", "Playbook"),
        ("comparison", "Comparison"),
        ("research", "Industry Data"),
        ("announcement", "Playbook"),
        ("news", "Playbook"),
        ("product_page", "Playbook"),
        (None, "Playbook"),
        ("Research", "Industry Data"),
    ],
)
def test_categories_come_from_the_fixed_mapping_only(content_type: str | None, category: str) -> None:  # fmt: skip
    assert site_category(content_type) == category
    assert category in AUTOMATIC_CATEGORIES
    assert set(AUTOMATIC_CATEGORIES) == {"Playbook", "Industry Data", "Comparison"}
    assert {"Product", "Announcements", "Research Paper"} <= set(CATEGORIES) - set(AUTOMATIC_CATEGORIES)  # fmt: skip


def test_tags_are_lowercase_deduplicated_and_between_three_and_six() -> None:
    assert site_tags(["WhatsApp", "whatsapp ", "Clinic  Automation"], ["patient calls"], ["x", "y", "z"]) == ["whatsapp", "clinic automation", "patient calls"]  # fmt: skip
    assert site_tags(["a", "b", "c", "d", "e", "f", "g"]) == ["a", "b", "c", "d", "e", "f"]
    assert site_tags(["only one"]) == ["only one"]  # nothing invented
    assert site_tags(["x" * 60]) == ["x" * 40]


# ── frontmatter ──────────────────────────────────────────────────────────────


def test_frontmatter_is_single_quoted_and_round_trips() -> None:
    fields = {"title": "It's \"quoted\": yes", "description": "Line one\nline two", "publishedAt": "2026-09-16", "author": "Engageo Team", "category": "Playbook", "tags": ["a b", "it's"], "draft": False, "agentPublication": MARKER, "agentSource": "competitor-analysis-agent"}  # fmt: skip
    text = frontmatter_text(fields)
    assert text.startswith("---\ntitle: 'It''s \"quoted\": yes'\n")
    assert "publishedAt: '2026-09-16'" in text  # quoted: the site emits ISO metadata
    assert "draft: false" in text
    assert "tags:\n  - 'a b'\n  - 'it''s'\n" in text
    parsed, body = split_frontmatter(text + "\nBody.\n")
    assert parsed == {**fields, "description": "Line one line two"}
    assert body == "Body.\n"
    assert isinstance(yaml.safe_load(text.strip("-\n")), dict)


def test_reading_an_existing_post_is_lenient() -> None:
    fields, body = split_frontmatter("---\ntitle: 'Old'\npublishedAt: 2026-03-02\ndraft: false\n---\n\nHello.\n")  # fmt: skip
    assert fields is not None
    assert fields["title"] == "Old"
    assert body == "Hello.\n"
    assert split_frontmatter("no frontmatter") == (None, "no frontmatter")
    assert split_frontmatter("---\n: bad: [\n---\nx") == (None, "---\n: bad: [\n---\nx")


# ── links ────────────────────────────────────────────────────────────────────


def test_internal_links_become_site_paths_or_plain_text() -> None:
    body = "See [pricing](https://www.engageoagency.com/pricing/) and [a page](/nowhere) and [demo](https://www.engageoagency.com/contact?intent=demo) and [Acme](https://acme.test/r) and [other](https://other.test/x)."  # fmt: skip
    out, kept, dropped = filter_internal_links(body, config(), ALLOWED)
    assert out == "See [pricing](/pricing) and a page and [demo](/contact?intent=demo) and [Acme](https://acme.test/r) and [other](https://other.test/x)."  # fmt: skip
    assert kept == ["/pricing", "/contact?intent=demo"]
    assert dropped == ["/nowhere"]


def test_jsx_attributes_are_escaped() -> None:
    assert jsx_attr('Say "hi" <b>{x}</b> & more') == "Say &quot;hi&quot; &lt;b&gt;&#123;x&#125;&lt;/b&gt; &amp; more"  # fmt: skip


# ── the document ─────────────────────────────────────────────────────────────


def test_compose_produces_the_repository_file() -> None:
    mdx = compose(document(), marker=MARKER, config=config(), allowed_paths=ALLOWED, published_on=date(2026, 9, 16))  # fmt: skip
    assert mdx.slug == "why-clinics-miss-calls-the-2026-data"
    assert mdx.path == "src/content/blog/why-clinics-miss-calls-the-2026-data.mdx"
    assert mdx.branch == "blog/why-clinics-miss-calls-the-2026-data"
    assert mdx.expected_url == f"{SITE}/blog/why-clinics-miss-calls-the-2026-data"
    fields, body = split_frontmatter(mdx.text)
    assert fields == {
        "title": "Why Clinics Miss Calls: The 2026 Data", "description": "A data-led look at missed calls.", "publishedAt": "2026-09-16",
        "author": "Engageo Team", "authorRole": "AI Content", "authorInitials": "EN", "category": "Industry Data",
        "tags": ["missed calls", "clinics", "patient calls"], "draft": False, "agentPublication": MARKER, "agentSource": "competitor-analysis-agent",
    }  # fmt: skip
    assert body.startswith("Intro ([Acme](https://acme.test/r)).\n\n## The numbers\n\nSee [our pricing](/pricing) and the guide and [Acme](https://acme.test/r).\n")  # fmt: skip
    assert "## Related reading\n\n- [clinic playbook](/blog/existing-post)\n" in body
    assert "/nowhere" not in body
    assert "no-such-post" not in body
    assert '<BlogCTA\n  title="See Engageo in action"\n  body="Say &quot;hello&quot; to &lt;5% no-shows &amp; &#123;real&#125; results"\n  ctaLabel="Book a demo"\n  ctaHref="/contact?intent=demo"\n/>' in body  # fmt: skip
    assert body.rstrip().endswith("---\n\n*The Engageo Team builds AI for clinics.*")
    assert mdx.internal_links == ("/pricing", "/blog/existing-post")
    assert mdx.dropped_links == ("https://www.engageoagency.com/blog/no-such-post",)
    assert mdx.reading_minutes == reading_minutes(body)
    assert validate(mdx.text, marker=MARKER) == []


def test_an_update_keeps_the_original_date_and_adds_updated_at() -> None:
    mdx = compose(document(), marker=MARKER, config=config(), allowed_paths=ALLOWED, published_on=date(2026, 9, 1), updated_on=date(2026, 9, 16))  # fmt: skip
    fields, _ = split_frontmatter(mdx.text)
    assert fields is not None
    assert (fields["publishedAt"], fields["updatedAt"]) == ("2026-09-01", "2026-09-16")


def test_the_author_linkedin_is_only_written_when_configured() -> None:
    with_link = compose(document(), marker=MARKER, config=config(author_linkedin="https://www.linkedin.com/company/engageo"), allowed_paths=ALLOWED, published_on=date(2026, 9, 16))  # fmt: skip
    assert "authorLinkedin: 'https://www.linkedin.com/company/engageo'" in with_link.text


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (
            "---\ntitle: 'T'\npublishedAt: '2026-09-16'\ndescription: 'd'\ncategory: 'Playbook'\ntags: []\ndraft: false\nagentPublication: '%s'\n---\n\nimport x from 'y'\n\n<BlogCTA title=\"a\" />\n",
            "import/export",
        ),
        (
            "---\ntitle: 'T'\npublishedAt: '2026-09-16'\ndescription: 'd'\ncategory: 'Playbook'\ntags: []\ndraft: false\nagentPublication: '%s'\n---\n\nA {brace}.\n\n<BlogCTA title=\"a\" />\n",
            "brace",
        ),
        (
            "---\ntitle: 'T'\npublishedAt: '2026-09-16'\ndescription: 'd'\ncategory: 'Playbook'\ntags: []\ndraft: false\nagentPublication: '%s'\n---\n\nA <script>x</script>.\n\n<BlogCTA title=\"a\" />\n",
            "'<' before a letter",
        ),
        (
            "---\ntitle: 'T'\npublishedAt: '2026-09-16'\ndescription: 'd'\ncategory: 'Playbook'\ntags: []\ndraft: false\nagentPublication: '%s'\n---\n\n# H1\n\n<BlogCTA title=\"a\" />\n",
            "H1",
        ),
        (
            "---\ntitle: 'T'\npublishedAt: '2026-09-16'\ndescription: 'd'\ncategory: 'Playbook'\ntags: []\ndraft: false\nagentPublication: '%s'\n---\n\nNo call to action.\n",
            "exactly one <BlogCTA />",
        ),
        (
            "---\ntitle: 'T'\npublishedAt: '2026-09-16'\ndescription: 'd'\ncategory: 'Playbook'\ntags: []\ndraft: true\nagentPublication: '%s'\n---\n\n<BlogCTA title=\"a\" />\n",
            "draft must be false",
        ),
        (
            "---\ntitle: 'T'\npublishedAt: '2026-09-16'\ndescription: 'd'\ncategory: 'Playbook'\ntags: []\ndraft: false\nagentPublication: 'other'\n---\n\n<BlogCTA title=\"a\" />\n",
            "marker",
        ),
        (
            "---\ntitle: 'T'\npublishedAt: 'Sept 16'\ndescription: 'd'\ncategory: 'Newsy'\ntags: ['Upper']\ndraft: false\nagentPublication: '%s'\n---\n\n<BlogCTA title=\"a\" />\n",
            "YYYY-MM-DD",
        ),
        ("no frontmatter at all", "frontmatter is missing"),
    ],
)
def test_validation_catches_what_would_break_the_build_or_the_contract(text: str, problem: str) -> None:  # fmt: skip
    problems = validate(text % MARKER if "%s" in text else text, marker=MARKER)
    assert any(problem in p for p in problems), problems


# ── the cover image ──────────────────────────────────────────────────────────


def cover(**overrides: Any) -> RenderedCover:
    values: dict[str, Any] = {"filename": "missed-calls.png", "mime": "image/png", "alt": "Abstract editorial illustration about missed calls", "width": 1536, "height": 864, "sha256": "a" * 64}  # fmt: skip
    values.update(overrides)
    return RenderedCover(**values)


def test_a_document_without_a_cover_writes_no_cover_keys() -> None:
    mdx = compose(document(), marker=MARKER, config=config(), allowed_paths=ALLOWED, published_on=date(2026, 9, 16))  # fmt: skip
    fields, _ = split_frontmatter(mdx.text)
    assert fields is not None
    assert not {"coverImage", "coverWidth", "coverHeight"} & set(fields)
    assert mdx.cover_path is None


def test_the_cover_is_named_after_the_post_and_sits_between_tags_and_draft() -> None:
    mdx = compose(document(cover=cover()), marker=MARKER, config=config(), allowed_paths=ALLOWED, published_on=date(2026, 9, 16))  # fmt: skip
    assert mdx.cover_path == "public/blog/covers/why-clinics-miss-calls-the-2026-data.png"
    assert "\ntags:\n  - 'missed calls'\n  - 'clinics'\n  - 'patient calls'\ncoverImage: '/blog/covers/why-clinics-miss-calls-the-2026-data.png'\ncoverWidth: 1536\ncoverHeight: 864\ndraft: false\n" in mdx.text  # fmt: skip
    fields, _ = split_frontmatter(mdx.text)
    assert fields is not None
    assert fields["coverImage"] == "/blog/covers/why-clinics-miss-calls-the-2026-data.png"
    assert (fields["coverWidth"], fields["coverHeight"]) == (1536, 864)  # numbers, not strings
    assert validate(mdx.text, marker=MARKER) == []


def test_the_cover_directory_and_url_prefix_are_configuration() -> None:
    mdx = compose(document(cover=cover(filename="x.jpg", mime="image/jpeg")), marker=MARKER, config=config(cover_dir="static/img", cover_url_prefix="/img"), allowed_paths=ALLOWED, published_on=date(2026, 9, 16))  # fmt: skip
    assert mdx.cover_path == "static/img/why-clinics-miss-calls-the-2026-data.jpg"
    assert "coverImage: '/img/why-clinics-miss-calls-the-2026-data.jpg'" in mdx.text


def test_a_cover_of_unknown_size_is_referenced_without_dimensions() -> None:
    mdx = compose(document(cover=cover(width=None, height=None)), marker=MARKER, config=config(), allowed_paths=ALLOWED, published_on=date(2026, 9, 16))  # fmt: skip
    fields, _ = split_frontmatter(mdx.text)
    assert fields is not None
    assert "coverImage" in fields
    assert not {"coverWidth", "coverHeight"} & set(fields)
    assert validate(mdx.text, marker=MARKER) == []


@pytest.mark.parametrize(
    ("keys", "problem"),
    [
        ("coverImage: 'blog/covers/x.png'", "site-absolute image path"),
        ("coverImage: '/blog/covers/x.pdf'", "site-absolute image path"),
        ("coverImage: '/blog/covers/x.png'\ncoverWidth: '1536'", "positive number"),
        ("coverWidth: 1536", "coverWidth without a coverImage"),
    ],
)
def test_validation_rejects_a_cover_the_site_would_ignore(keys: str, problem: str) -> None:
    text = f"---\ntitle: 'T'\npublishedAt: '2026-09-16'\ndescription: 'd'\ncategory: 'Playbook'\ntags: []\n{keys}\ndraft: false\nagentPublication: '{MARKER}'\n---\n\n<BlogCTA title=\"a\" />\n"  # fmt: skip
    assert any(problem in p for p in validate(text, marker=MARKER)), validate(text, marker=MARKER)


def test_a_stock_photo_is_credited_in_the_pull_request_and_nowhere_in_the_file() -> None:
    photo = cover(source="pexels", credit="Grace Hopper", credit_url="https://www.pexels.com/@grace", source_url="https://www.pexels.com/photo/desk-1/")  # fmt: skip
    mdx = compose(document(cover=photo), marker=MARKER, config=config(), allowed_paths=ALLOWED, published_on=date(2026, 9, 16))  # fmt: skip
    body = pr_body(document(cover=photo), mdx, generated_at=datetime(2026, 9, 16, 6, 30, tzinfo=UTC))  # fmt: skip
    assert "- Cover photo: [Grace Hopper](https://www.pexels.com/@grace) on Pexels — https://www.pexels.com/photo/desk-1/" in body  # fmt: skip
    # The site reads no credit field: the frontmatter keeps exactly the keys it always had.
    fields, _ = split_frontmatter(mdx.text)
    assert fields is not None
    assert "Grace Hopper" not in mdx.text
    assert set(fields) & {"coverImage", "coverWidth", "coverHeight"} == {"coverImage", "coverWidth", "coverHeight"}  # fmt: skip
    assert validate(mdx.text, marker=MARKER) == []


def test_a_generated_illustration_gets_no_credit_line() -> None:
    mdx = compose(document(cover=cover()), marker=MARKER, config=config(), allowed_paths=ALLOWED, published_on=date(2026, 9, 16))  # fmt: skip
    assert "Cover photo" not in pr_body(document(cover=cover()), mdx, generated_at=datetime(2026, 9, 16, 6, 30, tzinfo=UTC))  # fmt: skip
    assert cover_credit(None) == ""


def test_a_photographers_name_can_never_become_markup_in_the_description() -> None:
    page = "https://www.pexels.com/photo/desk-1/"
    credit = cover_credit(cover(source="pexels", credit="Eve [x](http://evil.test) `id`", source_url=page))  # fmt: skip
    assert credit == f"- Cover photo: Eve x http://evil.test id on Pexels — {page}"
    assert not set("[]()`") & set(credit.removesuffix(page))  # no link but the source's own


def test_the_pull_request_description_carries_provenance_and_no_secret() -> None:
    mdx = compose(document(), marker=MARKER, config=config(), allowed_paths=ALLOWED, published_on=date(2026, 9, 16))  # fmt: skip
    body = pr_body(document(), mdx, generated_at=datetime(2026, 9, 16, 6, 30, tzinfo=UTC))
    for needle in ("Why Clinics Miss Calls: The 2026 Data", "`why-clinics-miss-calls-the-2026-data`", "88.0/100", "Missed calls cost", "Sources cited: 1", "[Acme](https://acme.test/r)", "2026-09-16 06:30 UTC", "competitor-analysis agent", mdx.expected_url, "Vercel preview"):  # fmt: skip
        assert needle in body
    assert MARKER not in body  # the marker stays in the file, not the description
    assert "token" not in body.lower()
    assert "GEMINI" not in body
