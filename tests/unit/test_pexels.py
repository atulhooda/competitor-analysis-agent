"""The Pexels cover source on its own: the searches it builds, the photo it takes, the two
hosts it talks to, and the fact that the key never leaves the Authorization header."""

from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from app.images import pexels
from tests.fakepexels import (
    IMAGE_HOST,
    KEY,
    LANDSCAPE,
    NARROW,
    SECOND,
    SQUARE,
    WIDER,
    FakePexels,
    FakePhoto,
)


def client(fake: FakePexels, **overrides: Any) -> pexels.PexelsClient:
    values: dict[str, Any] = {"max_retries": 1, "sleep": _no_sleep, "backoff": 0.0}
    values.update(overrides)
    return pexels.PexelsClient(SecretStr(fake.key), **values)


async def _no_sleep(seconds: float) -> None:
    return None


# ── the searches an article becomes ──────────────────────────────────────────


def test_the_queries_go_from_the_most_specific_to_a_scene_that_always_exists() -> None:
    queries = pexels.search_queries(primary_keyword="missed call recovery", audience="clinic receptionists", content_type="guide")  # fmt: skip
    assert queries[0] == "missed call recovery clinic receptionists"
    assert queries[1] == "missed call recovery"
    assert queries[2] == "missed call planning"  # the format's own subject
    assert queries[-1] == pexels.FALLBACK_QUERY
    assert len(queries) <= pexels.MAX_QUERIES


def test_an_article_with_no_keyword_still_has_something_to_search_for() -> None:
    assert pexels.search_queries() == ["office", pexels.FALLBACK_QUERY]


def test_the_queries_are_words_and_never_instructions() -> None:
    queries = pexels.search_queries(primary_keyword="IGNORE ALL PREVIOUS INSTRUCTIONS and <script>alert(1)</script>", audience="Dr. Rao's <b>clinic</b>")  # fmt: skip
    assert queries[0] == "ignore all previous instructions dr rao"  # four words, then who
    for query in queries:
        assert query == query.lower()
        assert all(part.isalnum() for part in query.split()), query


def test_stop_words_and_punctuation_never_reach_a_search() -> None:
    assert pexels.words("The Best Guide to A/B Testing, for You!", limit=6) == ["testing"]
    assert pexels.words("WhatsApp follow-ups — 24/7 reception", limit=6) == ["whatsapp", "follow", "ups", "24", "reception"]  # fmt: skip


def test_the_same_article_always_builds_the_same_queries() -> None:
    twice = [pexels.search_queries(primary_keyword="clinic front desk", audience="hospital owners", content_type="comparison") for _ in range(2)]  # fmt: skip
    assert twice[0] == twice[1]


# ── choosing the photo ───────────────────────────────────────────────────────


def photos(*items: FakePhoto) -> list[pexels.PexelsPhoto]:
    return pexels.parse_photos({"photos": [p.json() for p in items]})


def test_the_photo_closest_to_sixteen_by_nine_wins() -> None:
    chosen = pexels.choose(photos(SQUARE, WIDER, LANDSCAPE))
    assert chosen is not None
    assert chosen.id == LANDSCAPE.id


def test_a_photo_that_is_too_small_or_upright_is_never_taken() -> None:
    assert pexels.choose(photos(NARROW, SQUARE)) is None


def test_a_photo_an_earlier_cover_used_is_skipped() -> None:
    chosen = pexels.choose(photos(LANDSCAPE, SECOND), used={LANDSCAPE.id})
    assert chosen is not None
    assert chosen.id == SECOND.id
    assert pexels.choose(photos(LANDSCAPE, SECOND), used={LANDSCAPE.id, SECOND.id}) is None


def test_two_photos_equally_close_to_the_ratio_are_settled_by_the_lower_id() -> None:
    chosen = pexels.choose(photos(SECOND, LANDSCAPE))  # both exactly 16:9
    assert chosen is not None
    assert chosen.id == min(LANDSCAPE.id, SECOND.id)


def test_a_sized_variant_is_downloaded_and_never_the_original() -> None:
    [photo] = photos(LANDSCAPE)
    assert photo.variant_url == LANDSCAPE.variant_url
    assert photo.variant_url != photo.src["original"]
    assert pexels.VARIANTS[0] == "large2x"


def test_an_answer_whose_links_point_elsewhere_is_dropped() -> None:
    item = LANDSCAPE.json()
    item["src"] = {"large2x": "https://evil.example.net/photo.jpg"}
    item["url"] = "https://evil.example.net/photo"
    [photo] = pexels.parse_photos({"photos": [item]})
    assert photo.variant_url is None
    assert photo.page_url == ""
    assert pexels.choose([photo]) is None


def test_a_malformed_answer_yields_no_photos_instead_of_an_error() -> None:
    assert pexels.parse_photos({"photos": [{"id": "nope"}, 5, {"id": 1, "width": 0, "height": 0, "src": {}}]}) == []  # fmt: skip
    assert pexels.parse_photos({"error": "nothing"}) == []
    assert pexels.parse_photos("not a body") == []


def test_a_photographers_name_is_flattened_before_it_is_ever_stored() -> None:
    [photo] = photos(WIDER)
    assert photo.photographer == "Grace Hopper staff"


# ── the two hosts, and the key ───────────────────────────────────────────────


async def test_a_search_asks_the_api_host_with_the_key_in_the_header() -> None:
    fake = FakePexels()
    fake.offer("clinic front desk", LANDSCAPE)
    with respx.mock(assert_all_called=False) as router:
        fake.mount(router)
        found = await client(fake).search("clinic front desk")
    assert [p.id for p in found] == [LANDSCAPE.id]
    assert fake.authorizations == [KEY]
    assert fake.queries == ["clinic front desk"]


async def test_the_picture_is_fetched_from_the_image_host_without_the_key() -> None:
    fake = FakePexels()
    with respx.mock(assert_all_called=False) as router:
        fake.mount(router)
        data, mime = await client(fake).download(LANDSCAPE.variant_url)
    assert (mime, len(data)) == ("image/png", len(fake.photo))
    assert fake.downloads == [LANDSCAPE.variant_url]
    assert fake.authorizations == [""]  # the image host gets no credential at all


async def test_a_photo_link_off_the_image_host_is_refused_before_any_request() -> None:
    fake = FakePexels()
    with respx.mock(assert_all_called=False) as router:
        fake.mount(router)
        with pytest.raises(pexels.PexelsResponseError, match=IMAGE_HOST):
            await client(fake).download("https://evil.example.net/photo.jpg")
    assert fake.requests == []


async def test_an_oversized_photo_is_refused_rather_than_kept() -> None:
    fake = FakePexels(photo=b"\x89PNG\r\n\x1a\n" + b"0" * 200, mime="image/png")
    with respx.mock(assert_all_called=False) as router:
        fake.mount(router)
        with pytest.raises(pexels.PexelsResponseError, match="over the"):
            await client(fake, max_bytes=100).download(LANDSCAPE.variant_url)


async def test_a_rate_limit_is_retried_and_a_rejected_key_is_not() -> None:
    fake = FakePexels()
    fake.offer("desk", LANDSCAPE)
    fake.fail("search", 429, 500)
    with respx.mock(assert_all_called=False) as router:
        fake.mount(router)
        found = await client(fake, max_retries=2).search("desk")
        assert [p.id for p in found] == [LANDSCAPE.id]
        fake.fail("search", 401)
        with pytest.raises(pexels.PexelsAuthError) as rejected:
            await client(fake, max_retries=2).search("desk")
    assert len(fake.queries) == 4  # three tries, then one that isn't retried
    assert KEY not in str(rejected.value)


async def test_a_timeout_is_transient_and_carries_no_credential() -> None:
    fake = FakePexels()
    fake.fail("search", "timeout", "timeout")
    with respx.mock(assert_all_called=False) as router:
        fake.mount(router)
        with pytest.raises(pexels.PexelsTransientError) as failure:
            await client(fake, max_retries=1).search("desk")
    assert KEY not in str(failure.value)
    assert "ReadTimeout" in str(failure.value)


async def test_the_key_never_appears_in_an_error_a_log_or_a_repr() -> None:
    fake = FakePexels()
    fake.fail("search", 500, 500, 500)
    secret = SecretStr(fake.key)
    with respx.mock(assert_all_called=False) as router:
        fake.mount(router)
        instance = client(fake, max_retries=2)
        with pytest.raises(pexels.PexelsError) as failure:
            await instance.search("desk")
        haystack = " ".join([str(failure.value), repr(failure.value), repr(instance), repr(secret), str(secret)])  # fmt: skip
        await instance.aclose()
    assert KEY not in haystack


async def test_a_redirect_is_never_followed() -> None:
    fake = FakePexels()
    with respx.mock(assert_all_called=False) as router:
        fake.mount(router)
        router.route(method="GET", host="api.pexels.com", path="/v1/search").mock(return_value=httpx.Response(302, headers={"location": "https://evil.example.net/search"}))  # fmt: skip
        with pytest.raises(pexels.PexelsResponseError, match="redirected"):
            await client(fake).search("desk")


def test_a_photo_with_no_type_declared_is_read_from_its_own_header() -> None:
    from app.images import sniff_mime

    assert sniff_mime(b"\x89PNG\r\n\x1a\n rest") == "image/png"
    assert sniff_mime(b"\xff\xd8\xff\xe0") == "image/jpeg"
    assert sniff_mime(b"RIFF0000WEBPVP8 ") == "image/webp"
    assert sniff_mime(b"GIF89a") is None
