"""Research safety and bookkeeping (no LLM, no network): URL screening, source typing, and
matching what the URL tool reports to what was requested."""

import pytest

from app.domain.articles import SourceType
from app.llm import RetrievedURL
from app.services.research import classify, match_retrievals, safe_public_url


async def public(host: str) -> list[str]:
    return ["93.184.216.34"]


async def unresolvable(host: str) -> list[str]:
    raise OSError("no such host")


async def private_dns(host: str) -> list[str]:
    return ["10.0.0.7"]


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("javascript:alert(1)", "not an http(s) URL (javascript)"),
        ("file:///etc/passwd", "not an http(s) URL (file)"),
        ("ftp://example.org/x", "not an http(s) URL (ftp)"),
        ("https://user:pw@example.org/", "the URL carries credentials"),
        ("https://example.org:8443/", "non-standard port 8443"),
        ("http://169.254.169.254/latest/meta-data", "unsafe destination"),
        ("http://127.0.0.1/admin", "unsafe destination"),
        ("http://localhost/", "unsafe destination"),
        ("https://intranet.internal/", "unsafe destination"),
        ("https://example.org/" + "a" * 2_100, "URL too long"),
    ],
)
async def test_unsafe_or_unusable_urls_are_rejected(url: str, reason: str) -> None:
    normalized, why = await safe_public_url(url, public)
    assert normalized is None
    assert why is not None
    assert why.startswith(reason)


async def test_urls_that_resolve_privately_or_not_at_all_are_rejected() -> None:
    assert (await safe_public_url("https://rebind.example/", private_dns))[1] == "unsafe destination (rebind.example resolves to non-public address 10.0.0.7)"  # fmt: skip
    assert (await safe_public_url("https://no-such-domain.example/", unresolvable)) == (None, "the domain does not resolve")  # fmt: skip


async def test_public_urls_are_normalized() -> None:
    assert await safe_public_url(" https://Docs.Example.org/Guide?utm_source=x#top ", public) == ("https://docs.example.org/Guide", None)  # fmt: skip


def test_your_site_and_competitors_are_recognized_by_domain() -> None:
    competitors = ["acme.test"]
    assert classify("https://blog.acme.test/post", SourceType.OFFICIAL_DOCS, competitors, "startup.example") is SourceType.COMPETITOR  # fmt: skip
    assert classify("https://www.startup.example/docs", SourceType.NEWS, competitors, "startup.example") is SourceType.COMPANY  # fmt: skip
    assert classify("https://notacme.test/", SourceType.RESEARCH, competitors, None) is SourceType.RESEARCH  # fmt: skip
    # The model can't label a page as the company's or a competitor's itself.
    assert classify("https://random.example/", SourceType.COMPETITOR, competitors, None) is SourceType.OTHER  # fmt: skip


def test_retrievals_are_matched_exactly_or_through_a_redirect() -> None:
    requested = [
        "https://docs.example.org/fake-page-that-does-not-exist",  # listed first on purpose
        "https://docs.example.org/our-work/report-cookie-banner-taskforce",
        "https://ok.example/page",
        "https://gone.example/x",
    ]
    results = [
        RetrievedURL(
            "https://docs.example.org/documents/report-of-the-cookie-banner-taskforce", "success"
        ),
        RetrievedURL("https://docs.example.org/fake-page-that-does-not-exist", "error"),
        RetrievedURL("https://docs.example.org/our-work/report-cookie-banner-taskforce", "error"),
        RetrievedURL("https://ok.example/page", "success"),
        RetrievedURL("https://gone.example/x", "paywall"),
    ]
    matched = match_retrievals(requested, results)
    # The redirect target goes to the request with the most similar path, not the first one.
    assert matched["https://docs.example.org/our-work/report-cookie-banner-taskforce"] == ("https://docs.example.org/documents/report-of-the-cookie-banner-taskforce", "success")  # fmt: skip
    assert matched["https://docs.example.org/fake-page-that-does-not-exist"] == ("https://docs.example.org/fake-page-that-does-not-exist", "error")  # fmt: skip
    assert matched["https://ok.example/page"] == ("https://ok.example/page", "success")
    assert matched["https://gone.example/x"][1] == "paywall"
    assert match_retrievals(["https://never.example/"], [])["https://never.example/"][1] == "not_retrieved"  # fmt: skip
