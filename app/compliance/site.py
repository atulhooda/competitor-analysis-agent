"""A post as the website stores it (``content/blog/<slug>.json`` + ``<slug>.mdx``), as a
``ComplianceInput``: so existing posts, and posts written by hand, are checked by the same
rules as the agent's."""

import re
from typing import Any

from app.compliance.rules import ComplianceInput

_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")


def from_site_post(fields: dict[str, Any], body: str) -> ComplianceInput:
    slug = str(fields.get("slug") or "")
    texts = [(f"the {name}", str(fields[key])) for key, name in (("title", "title"), ("slug", "slug"), ("metaTitle", "meta title (also the Open Graph and Twitter title)"), ("metaDescription", "meta description (also the Open Graph, Twitter and JSON-LD description)"), ("excerpt", "excerpt"), ("author", "author")) if fields.get(key)]  # fmt: skip
    urls = [("the post URL", f"/blog/{slug}")]
    hero = fields.get("heroImage") or {}
    if isinstance(hero, dict):
        if hero.get("alt"):
            texts.append(("the hero image alt text", str(hero["alt"])))
        if hero.get("src"):
            urls.append(("the hero image file name", str(hero["src"])))
    for key in ("relatedTreatments", "relatedConcerns"):
        urls += [(f"related link '{s}'", f"/{key.removeprefix('related').lower()}/{s}") for s in fields.get(key) or []]  # fmt: skip
    urls += [(f"link '{m.group(1)[:40]}'", m.group(2)) for m in _LINK.finditer(body)]
    for m in re.finditer(r"!\[([^\]]*)\]\(([^)\s]+)\)", body):
        texts.append(("an image's alt text", m.group(1)))
    reviewer = fields.get("reviewedBy")
    return ComplianceInput(fields=texts, body=body, urls=urls, author_box=str(reviewer) if reviewer else None)  # fmt: skip


__all__ = ["from_site_post"]
