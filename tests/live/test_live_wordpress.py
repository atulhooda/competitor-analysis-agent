"""Opt-in: the WordPress adapter against a real WordPress site. It never publishes: it
creates one uniquely named test draft, reads it back, updates it, checks the content, and
moves it to the trash (only that draft, which it identifies by its marker).

    LIVE_WORDPRESS=1 uv run pytest -m cms_live tests/live/test_live_wordpress.py -s

Needs WORDPRESS_BASE_URL, WORDPRESS_USERNAME and WORDPRESS_APPLICATION_PASSWORD (environment
or .env), and LIVE_WORDPRESS=1 as an explicit go-ahead. Skipped otherwise. Use a staging
site or a dedicated test user if you can.
"""

import os
import uuid

import pytest

from app.cms import LazyCMS
from app.cms.base import TermResolution
from app.config import Settings
from app.domain.publishing import CMSPostStatus, RenderedDocument, TargetStatus

pytestmark = pytest.mark.cms_live

# Read at import, before the test fixtures isolate the environment.
_SETTINGS = Settings()
_GO = os.environ.get("LIVE_WORDPRESS") == "1"


async def test_real_wordpress_draft_round_trip() -> None:
    if not (_GO and _SETTINGS.cms_configured):
        pytest.skip("set LIVE_WORDPRESS=1 and the WORDPRESS_* settings to run against a real site")
    marker = uuid.uuid4().hex
    slug = f"cia-live-test-{marker[:10]}"
    doc = RenderedDocument(
        render_version="render/1", title=f"[test] competitor-analysis-agent {marker[:10]}", slug=slug,
        excerpt="A test draft from the competitor analysis agent's live test. Safe to delete.",
        meta_title="test", primary_keyword="test", category=None, tags=[],
        body_html="<p>A test draft. It is moved to the trash when the test ends.</p>\n",
        sources=[], faq=[], links=[], image=None, word_count=12, content_hash=marker,
    )  # fmt: skip
    cms = LazyCMS(_SETTINGS)
    publisher = cms.get()
    created_id: str | None = None
    try:
        check = await publisher.check()
        print(f"\n{check}")
        assert check.reachable, check.detail
        assert check.authenticated, check.detail
        assert check.can_create, check.detail

        payload = publisher.build_payload(doc, status=TargetStatus.DRAFT, terms=TermResolution(None, ()), marker=marker)  # fmt: skip
        created = await publisher.create_post(payload)
        created_id = created.external_id
        print(f"created draft {created.external_id}: {created.edit_url}")
        assert created.status is CMSPostStatus.DRAFT  # never published
        assert publisher.owns(created, marker)

        fetched = await publisher.get_post(created.external_id)
        assert fetched is not None
        assert fetched.status is CMSPostStatus.DRAFT
        found = await publisher.find_posts(slug=slug)
        assert [p.external_id for p in found] == [created.external_id]

        updated_doc = doc.model_copy(update={"body_html": "<p>Updated by the live test.</p>\n"})
        updated = await publisher.update_post(created.external_id, publisher.build_payload(updated_doc, status=TargetStatus.DRAFT, terms=TermResolution(None, ()), marker=marker))  # fmt: skip
        problems, warnings = publisher.verify(updated, publisher.build_payload(updated_doc, status=TargetStatus.DRAFT, terms=TermResolution(None, ()), marker=marker), status=TargetStatus.DRAFT, marker=marker)  # fmt: skip
        print(f"verify: problems={problems} warnings={warnings}")
        assert problems == []
        assert "Updated by the live test." in (updated.content or "")
    finally:
        if created_id is not None:  # clean up: only the draft this test created
            post = await publisher.get_post(created_id)
            if post is not None and publisher.owns(post, marker) and post.status is CMSPostStatus.DRAFT:  # fmt: skip
                await publisher._client.delete(f"wp/v2/posts/{int(created_id)}")  # type: ignore[attr-defined]  # moves it to the trash
                print(f"moved draft {created_id} to the trash")
        await cms.aclose()
