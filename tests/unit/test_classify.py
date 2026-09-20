import pytest

from app.crawling.classify import classify_page, classify_url
from app.domain.content import ContentType, DiscoverySource

T = ContentType


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://acme.test/", T.HOMEPAGE),
        ("https://acme.test/pricing", T.PRICING),
        ("https://acme.test/en-us/pricing", T.PRICING),
        ("https://acme.test/products/support/pricing", T.PRICING),
        ("https://acme.test/blog/ai-agents", T.BLOG_POST),
        ("https://acme.test/blog/pricing", T.BLOG_POST),  # a post about pricing
        ("https://acme.test/2026/09/some-post", T.BLOG_POST),
        ("https://blog.acme.test/some-post", T.BLOG_POST),
        ("https://blog.acme.test/", T.LISTING),
        ("https://acme.test/blog", T.LISTING),
        ("https://acme.test/blog/tag/ai", T.LISTING),
        ("https://acme.test/blog/page/2", T.LISTING),
        ("https://acme.test/blog?page=2", T.LISTING),
        ("https://acme.test/customers/globex", T.CASE_STUDY),
        ("https://acme.test/case-studies/initech", T.CASE_STUDY),
        ("https://acme.test/press/series-b", T.PRESS),
        ("https://acme.test/changelog", T.CHANGELOG),
        ("https://acme.test/resources/ebooks/ai-guide", T.RESOURCE),
        ("https://acme.test/features/automation", T.PRODUCT),
        ("https://acme.test/compare/acme-vs-globex", T.LANDING_PAGE),
        ("https://acme.test/zendesk-alternative", T.LANDING_PAGE),
        ("https://acme.test/about", T.LANDING_PAGE),
        ("https://acme.test/careers", T.CAREERS),
        ("https://jobs.acme.test/listing/1", T.CAREERS),
        ("https://acme.test/legal/privacy", T.LEGAL),
        ("https://docs.acme.test/start", T.DOCS),
        ("https://acme.test/help/reset-password", T.DOCS),
        ("https://acme.test/misc/thing", T.OTHER),
    ],
)
def test_classify_url(url: str, expected: ContentType) -> None:
    result = classify_url(url)
    assert result.content_type is expected, result.reason
    assert result.reason


def test_structured_data_upgrades_weak_url_results() -> None:
    result = classify_page("https://acme.test/misc/thing", jsonld_types={"BlogPosting"})
    assert result.content_type is T.BLOG_POST
    assert "blogposting" in result.reason


def test_feed_membership_upgrades_weak_url_results() -> None:
    result = classify_page(
        "https://acme.test/ai-agents-guide", discovered_via={DiscoverySource.FEED}
    )
    assert result.content_type is T.BLOG_POST


def test_strong_url_results_are_not_overridden() -> None:
    result = classify_page("https://acme.test/pricing", jsonld_types={"Article"}, og_type="article")
    assert result.content_type is T.PRICING
