from datetime import UTC, datetime

from app.crawling.extract import extract_page
from app.crawling.html import decode_html, scan_html
from app.domain.content import DateSource
from tests.fakesite import BASE, HOME_HTML, article_html

GRAPH_PAGE = """<html lang="en-GB"><head><title>Guide</title>
<link rel="canonical" href="/guides/guide?utm_source=x">
<script type="application/ld+json">{"@context":"https://schema.org","@graph":[
 {"@type":"WebSite","name":"Acme"},
 {"@type":["Article","TechArticle"],"datePublished":"2026-08-20","dateModified":"2026-09-01",
  "author":[{"@type":"Person","name":"Alex"}]}]}</script>
<script type="application/ld+json">{ not valid json</script>
</head><body><a href="/a">A</a><a href="mailto:x@acme.test">mail</a></body></html>"""


def test_scan_html_reads_jsonld_graph_canonical_and_links() -> None:
    signals = scan_html(GRAPH_PAGE, f"{BASE}/guides/guide")
    assert signals.canonical == f"{BASE}/guides/guide"
    assert {"article", "techarticle", "website"} <= signals.jsonld_types
    assert signals.jsonld_published == datetime(2026, 8, 20, tzinfo=UTC)
    assert signals.jsonld_modified == datetime(2026, 9, 1, tzinfo=UTC)
    assert signals.jsonld_author == "Alex"
    assert signals.links == [f"{BASE}/a"]
    assert signals.lang == "en-GB"


def test_scan_html_finds_advertised_feeds() -> None:
    signals = scan_html(HOME_HTML, f"{BASE}/")
    assert signals.feed_links == [f"{BASE}/blog/feed.xml"]


def test_extract_article() -> None:
    html = article_html("/blog/x", "AI Agents Guide", published="2026-09-10T08:00:00Z")
    page = extract_page(html, f"{BASE}/blog/x")
    assert page.title == "AI Agents Guide"
    assert page.author == "Jane Doe"
    assert page.published_at == datetime(2026, 9, 10, 8, tzinfo=UTC)
    assert page.date_source is DateSource.STRUCTURED_DATA
    assert page.word_count > 100
    assert not page.is_thin
    assert "## Key takeaways" in page.text  # headings preserved as Markdown
    assert page.content_hash is not None
    assert len(page.content_hash) == 64


def test_meta_published_time_is_used_without_jsonld() -> None:
    html = article_html("/blog/y", "Pricing news", meta_published="2026-09-05T09:30:00Z")
    page = extract_page(html, f"{BASE}/blog/y")
    assert page.published_at == datetime(2026, 9, 5, 9, 30, tzinfo=UTC)
    assert page.date_source is DateSource.META


def test_content_hash_ignores_whitespace_only_changes() -> None:
    html = article_html("/blog/x", "Same", published="2026-09-10T08:00:00Z")
    first = extract_page(html, f"{BASE}/blog/x")
    second = extract_page(html.replace("</p>", "</p>\n\n   "), f"{BASE}/blog/x")
    assert first.content_hash == second.content_hash


def test_thin_page_is_flagged() -> None:
    page = extract_page("<html><body><div id='root'></div></body></html>", f"{BASE}/app")
    assert page.is_thin
    assert page.word_count == 0


def test_decode_html_charsets() -> None:
    latin = "Café".encode("latin-1")
    assert decode_html(latin, "text/html; charset=ISO-8859-1") == "Café"
    assert "Café" in decode_html(b'<meta charset="iso-8859-1">' + latin, None)
    assert decode_html("Café".encode(), None) == "Café"
    assert decode_html("Café".encode(), "text/html; charset=not-a-charset") == "Café"
