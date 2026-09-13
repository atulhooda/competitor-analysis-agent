import pytest

from app.crawling.urls import SiteScope, normalize_url, origin_of


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTPS://Acme.TEST/Blog/Post", "https://acme.test/Blog/Post"),
        ("https://acme.test", "https://acme.test/"),
        ("https://acme.test:443/a", "https://acme.test/a"),
        ("http://acme.test:8080/a", "http://acme.test:8080/a"),
        ("https://acme.test/a#section", "https://acme.test/a"),
        ("https://user:pw@acme.test/a", "https://acme.test/a"),
        ("https://acme.test/a?utm_source=x&id=7&gclid=abc", "https://acme.test/a?id=7"),
        ("https://acme.test/a?b=2&a=1", "https://acme.test/a?b=2&a=1"),  # untouched order
        ("https://bücher.example/p", "https://xn--bcher-kva.example/p"),
    ],
)
def test_normalize_url(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


def test_normalize_resolves_relative_urls() -> None:
    assert (
        normalize_url("../pricing", base="https://acme.test/blog/post")
        == "https://acme.test/pricing"
    )


@pytest.mark.parametrize(
    "raw", ["", "mailto:a@acme.test", "javascript:void(0)", "ftp://acme.test/f", "https://:80/"]
)
def test_normalize_rejects_non_crawlable(raw: str) -> None:
    assert normalize_url(raw) is None


def test_scope_covers_base_domain_and_subdomains_only() -> None:
    scope = SiteScope.from_urls(["https://www.acme.test/"], extra_domains=["acme-blog.test"])
    assert scope.contains("https://acme.test/x")
    assert scope.contains("https://blog.acme.test/x")
    assert scope.contains("https://acme-blog.test/y")
    assert not scope.contains("https://notacme.test/")
    assert not scope.contains("https://acme.test.evil.example/")


def test_origin_of() -> None:
    assert origin_of("https://acme.test:8443/a/b?c=1") == "https://acme.test:8443"
