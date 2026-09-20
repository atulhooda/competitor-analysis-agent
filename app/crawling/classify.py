"""Deterministic, rule-based page classification. No LLM.

URL rules run first; structured data (JSON-LD / OpenGraph) and feed membership can
upgrade weak URL results to ``blog_post``. Every result carries a human-readable reason.
"""

import re
from collections.abc import Collection
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

from app.crawling.html import ARTICLE_TYPES
from app.domain.content import ContentType, DiscoverySource

_LOCALE = re.compile(r"^[a-z]{2}(?:[-_][a-z]{2,4})?$")
_YEAR = re.compile(r"^(?:19|20)\d{2}$")
_PAGINATION_PARAMS = frozenset({"page", "paged", "p"})


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


_LISTING = _words(
    "tag tags category categories author authors topic topics archive archives page search"
)
_LEGAL = _words(
    "legal privacy privacy-policy terms terms-of-service terms-of-use tos cookie-policy cookies "
    "gdpr dpa imprint impressum acceptable-use"
)
_CAREERS = _words("careers career jobs job join-us work-with-us hiring")
_DOCS = _words(
    "docs documentation help support kb knowledge-base developers developer api-reference manual"
)
_PRICING = _words("pricing plans price prices plans-and-pricing")
_CHANGELOG = _words("changelog release-notes releases whats-new product-updates")
_CASE_STUDY = _words(
    "case-studies case-study customers customer-stories customer-story success-stories"
)
_PRESS = _words("press newsroom press-releases news in-the-news media")
_RESOURCE = _words(
    "resources ebooks ebook whitepapers whitepaper guides guide webinars webinar reports templates "
    "template events podcast podcasts videos library academy learn glossary"
)
_BLOG = _words("blog blogs posts post articles article insights stories journal")
_PRODUCT = _words(
    "product products features feature platform solutions solution integrations integration "
    "use-cases use-case industries industry enterprise"
)
_LANDING = _words(
    "compare comparison vs alternatives alternative why demo get-started lp landing partners"
)
# Sections whose bare index page ("/blog") is a listing, not content.
_SECTIONS = _BLOG | _CASE_STUDY | _PRESS | _RESOURCE
_KEYWORD_RULES = (
    (ContentType.LEGAL, _LEGAL),
    (ContentType.CAREERS, _CAREERS),
    (ContentType.DOCS, _DOCS),
    (ContentType.PRICING, _PRICING),
    (ContentType.CHANGELOG, _CHANGELOG),
)
_DOCS_HOSTS = ("docs.", "help.", "support.", "developer.", "developers.", "kb.")


@dataclass(frozen=True)
class Classification:
    content_type: ContentType
    reason: str


def _segments(url: str) -> tuple[str, list[str]]:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    segments = [s for s in parts.path.lower().split("/") if s]
    if segments and _LOCALE.match(segments[0]) and len(segments) > 1:
        segments = segments[1:]  # /en-us/pricing → pricing
    return host, segments


def classify_url(url: str) -> Classification:
    host, segs = _segments(url)
    query = parse_qs(urlsplit(url).query)
    first = segs[0] if segs else ""

    if host.startswith(_DOCS_HOSTS):
        return Classification(ContentType.DOCS, f"host '{host}'")
    if host.startswith(("careers.", "jobs.")):
        return Classification(ContentType.CAREERS, f"host '{host}'")
    if host.startswith("blog.") and segs and not (set(segs) & _LISTING):
        segs = ["blog", *segs]  # blog.acme.com/post-slug behaves like acme.com/blog/post-slug
        first = "blog"
    if not segs:
        if host.startswith(("blog.", "news.")):
            return Classification(ContentType.LISTING, f"root of '{host}'")
        return Classification(ContentType.HOMEPAGE, "site root")

    if hit := set(segs) & _LISTING:
        return Classification(ContentType.LISTING, f"segment '{sorted(hit)[0]}'")
    if _PAGINATION_PARAMS & query.keys():
        return Classification(ContentType.LISTING, "pagination query")
    if len(segs) == 1 and first in _SECTIONS:
        return Classification(ContentType.LISTING, f"section index '/{first}'")
    # A content section in the first segment wins over keywords later in the path:
    # /blog/pricing is a blog post about pricing, not the pricing page.
    for section_type, sections in (
        (ContentType.CASE_STUDY, _CASE_STUDY),
        (ContentType.PRESS, _PRESS),
        (ContentType.BLOG_POST, _BLOG),
        (ContentType.RESOURCE, _RESOURCE),
    ):
        if first in sections:
            return Classification(section_type, f"under '/{first}/'")
    last = segs[-1]
    # The first segment is the site section, the last is the page itself, then anything
    # in between: /products/support/pricing is a pricing page, /help/pricing is docs.
    for candidates in ({first}, {last}, set(segs)):
        for rule_type, keywords in _KEYWORD_RULES:
            if hit := candidates & keywords:
                return Classification(rule_type, f"segment '{sorted(hit)[0]}'")
    if any(_YEAR.match(s) for s in segs[:-1]):
        return Classification(ContentType.BLOG_POST, "date-based path")
    if first in _PRODUCT:
        return Classification(ContentType.PRODUCT, f"under '/{first}/'")
    comparison_slug = (
        last.startswith(("why-", "vs-"))
        or "-vs-" in last
        or last.endswith(("-alternative", "-alternatives"))
    )
    if first in _LANDING or comparison_slug:
        return Classification(ContentType.LANDING_PAGE, "comparison/landing URL")
    if len(segs) == 1:
        return Classification(ContentType.LANDING_PAGE, "top-level page")
    return Classification(ContentType.OTHER, "no matching rule")


# URL results that page-level signals must not override.
_STRONG_URL_TYPES = frozenset(
    {
        ContentType.HOMEPAGE,
        ContentType.PRICING,
        ContentType.LISTING,
        ContentType.LEGAL,
        ContentType.CAREERS,
        ContentType.DOCS,
        ContentType.CHANGELOG,
        ContentType.CASE_STUDY,
        ContentType.PRESS,
        ContentType.BLOG_POST,
        ContentType.RESOURCE,
    }
)


def classify_page(
    url: str,
    *,
    discovered_via: Collection[DiscoverySource] = (),
    jsonld_types: Collection[str] = (),
    og_type: str | None = None,
) -> Classification:
    base = classify_url(url)
    if base.content_type in _STRONG_URL_TYPES:
        return base
    article_types = ARTICLE_TYPES & {t.lower() for t in jsonld_types}
    if article_types:
        return Classification(
            ContentType.BLOG_POST, f"structured data @type {sorted(article_types)[0]}"
        )
    if (og_type or "").lower() == "article":
        return Classification(ContentType.BLOG_POST, "og:type article")
    if DiscoverySource.FEED in discovered_via:
        return Classification(ContentType.BLOG_POST, "listed in the site's feed")
    return base
