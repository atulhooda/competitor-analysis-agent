"""The site's content contract (Phase 8): one ``src/content/blog/<slug>.mdx`` per post, as
the engageo-website repository expects it.

Discovered from the repository and kept here, not in the core:

- **Frontmatter** (``gray-matter``): ``title`` and ``publishedAt`` are required; dates are
  written as quoted ISO strings so the site emits correct published metadata; ``category``
  must be one of the site's six categories (an unknown value silently becomes Playbook
  there); ``tags`` are lowercase; ``draft`` must be ``false`` (a draft is invisible even on
  preview deployments; the pull request is the review state); ``coverImage`` (with
  ``coverWidth``/``coverHeight``, unquoted integers, which the detail page uses to render
  the cover at its natural aspect ratio) appears only when a cover was generated
  (PUBLISH_COVER_IMAGES); ``cardImage`` is never written. Two keys the site ignores carry
  ownership: ``agentPublication`` (this publication's marker) and ``agentSource``.
- **The slug is the filename.** The site derives it from the filename with its own rules,
  applied here first: lowercase ASCII, ``[a-z0-9-]``, no leading, trailing or double dash.
- **Category.** A fixed mapping from the brief's content format: comparisons →
  Comparison, research/data → Industry Data, everything else → Playbook. Product,
  Announcements and Research Paper are never chosen automatically.
- **Body.** The Markdown renderer's body (intro paragraphs first, H2/H3, lists, inline
  source links, FAQ), then a "Related reading" list of existing site pages, exactly one
  ``<BlogCTA />`` and a closing italic byline. The author, CTA and byline are configuration.
- **Safety.** Before a file is written, the whole document is checked: the frontmatter
  parses back to exactly what was meant, no line is an ESM statement, no ``{``, ``}`` or
  ``<`` survives outside the one component we emit, and that component is well-formed.
  The site's own build (the preview deployment) is the last gate.

A second layout, ``json_pair`` (the skin-essence repository), keeps a post's fields in
``<slug>.json`` and only its body in ``<slug>.mdx``: the site checks the JSON against its own
schema at build time (``metaTitle`` 10-80 characters, ``metaDescription`` 50-155, an ISO
``date``, a ``heroImage`` with its size) and renders the title, byline, related treatments
and concerns and its call to action itself, so the body carries none of them. The JSON
holds the ownership keys (``agentPublication``, ``agentSource``) and ``agentBody``, the
body file's SHA-256, so the one file that is looked up says whether the other changed.
"""

import hashlib
import json
import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from urllib.parse import urlsplit

import yaml

from app.domain.publishing import (
    CoverImageSource,
    RenderedCover,
    RenderedDocument,
    RenderedLink,
)
from app.services.article_markdown import link_text, link_url

CATEGORIES: tuple[str, ...] = ("Industry Data", "Comparison", "Playbook", "Product", "Announcements", "Research Paper")  # fmt: skip
AUTOMATIC_CATEGORIES: tuple[str, ...] = ("Playbook", "Industry Data", "Comparison")
DEFAULT_CATEGORY = "Playbook"
# The brief's content format (app.domain.analysis.ContentFormat) → the site's category.
CATEGORY_BY_FORMAT: dict[str, str] = {
    "guide": "Playbook",
    "tutorial": "Playbook",
    "listicle": "Playbook",
    "article": "Playbook",
    "opinion": "Playbook",
    "interview": "Playbook",
    "case_study": "Playbook",
    "comparison": "Comparison",
    "research": "Industry Data",
}
AGENT_SOURCE = "competitor-analysis-agent"
MIN_TAGS, MAX_TAGS, MAX_TAG_LENGTH = 3, 6, 40
MAX_SLUG_LENGTH = 80
RELATED_HEADING = "Related reading"
# The site's own order; coverImage sits between tags and draft, as its existing posts do.
FRONTMATTER_ORDER = ("title", "description", "publishedAt", "updatedAt", "author", "authorRole", "authorInitials", "authorLinkedin", "category", "tags", "coverImage", "coverWidth", "coverHeight", "draft", "agentPublication", "agentSource")  # fmt: skip
COVER_KEYS = ("coverImage", "coverWidth", "coverHeight")
# How a cover's source is named in the pull request. The site reads no credit field, so
# this is the only place a reviewer sees where the picture came from.
COVER_SOURCE_NAMES = {CoverImageSource.PEXELS.value: "Pexels"}
DEFAULT_COVER_DIR = "public/blog/covers"
DEFAULT_COVER_URL_PREFIX = "/blog/covers"
# Site paths that may carry a query (the site's own CTA targets).
QUERY_PATHS = frozenset({"/contact"})
_SLUG_ALLOWED = re.compile(r"[^a-z0-9-]+")
_MD_LINK = re.compile(r"\[((?:\\.|[^\]\\])*)\]\((\S+?)\)")
_ESM_LINE = re.compile(r"^(import|export)\s", re.MULTILINE)
_JSX_OPEN = re.compile(r"<(?=[A-Za-z/])")
_MARKER = re.compile(r"^[0-9a-f]{32}$")

FRONTMATTER, JSON_PAIR = "frontmatter", "json_pair"
# json_pair: the site's field order, then the agent's own keys (its schema drops unknown keys).
PAIR_ORDER = ("slug", "title", "metaTitle", "metaDescription", "date", "excerpt", "author", "reviewedBy", "readingMinutes", "heroImage", "relatedTreatments", "relatedConcerns", "agentPublication", "agentSource", "agentBody")  # fmt: skip
# Internal links to these sections become the post's related cards (field → path prefix).
PAIR_RELATED = (("relatedTreatments", "/treatments/"), ("relatedConcerns", "/concerns/"))
MAX_RELATED = 3
META_TITLE_RANGE, META_DESCRIPTION_RANGE = (10, 80), (50, 155)
# A post without a cover borrows the picture of its first related treatment.
TREATMENT_IMAGE, TREATMENT_IMAGE_SIZE = "/images/treatments/{slug}.webp", (1200, 900)
_PAIR_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class SiteConfig:
    """What the site expects around an article: fixed configuration, never model output."""

    site_url: str
    content_dir: str
    branch_prefix: str
    author_name: str
    author_role: str
    author_initials: str
    author_linkedin: str | None
    cta_title: str
    cta_body: str
    cta_label: str
    cta_href: str
    byline: str
    # Where a generated cover is committed, and what the frontmatter points at.
    cover_dir: str = DEFAULT_COVER_DIR
    cover_url_prefix: str = DEFAULT_COVER_URL_PREFIX
    categories: tuple[str, ...] = CATEGORIES
    query_paths: frozenset[str] = QUERY_PATHS
    # The blog index a new post must appear on, and the element holding its post cards.
    index_path: str = "/blog"
    index_container_id: str = "blog-posts"
    layout: str = FRONTMATTER  # or JSON_PAIR
    meta_title_suffix: str = ""  # json_pair: " | Skin Essence Pune"
    # What the live post must carry to count as published.
    published_marker: str = "article:published_time"

    @property
    def site_host(self) -> str:
        return (urlsplit(self.site_url).hostname or "").lower()


@dataclass(frozen=True)
class MDXDocument:
    slug: str
    path: str
    branch: str
    title: str
    frontmatter: dict[str, Any]
    text: str
    reading_minutes: int
    expected_url: str
    internal_links: tuple[str, ...]
    dropped_links: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)
    cover_path: str | None = None  # the image file in the repository, when there is a cover
    body_path: str | None = None  # json_pair: the body file next to the JSON
    body_text: str | None = None


# ── the pieces ───────────────────────────────────────────────────────────────


def site_slug(slug: str) -> str:
    """The site's slug rules (its ``slugify``), on an ASCII-folded slug."""
    import unicodedata

    folded = unicodedata.normalize("NFKD", slug).encode("ascii", "ignore").decode().lower().strip()
    out = re.sub(r"[\s_]+", "-", folded)
    out = _SLUG_ALLOWED.sub("", out)
    out = re.sub(r"-{2,}", "-", out).strip("-")
    if len(out) > MAX_SLUG_LENGTH:
        out = out[:MAX_SLUG_LENGTH].rsplit("-", 1)[0].strip("-") if "-" in out[:MAX_SLUG_LENGTH] else out[:MAX_SLUG_LENGTH]  # fmt: skip
    return out


def site_category(content_type: str | None, *, allowed: Collection[str] = CATEGORIES) -> str:
    """The fixed mapping; never a free choice of the model's."""
    category = CATEGORY_BY_FORMAT.get((content_type or "").strip().lower(), DEFAULT_CATEGORY)
    return category if category in allowed and category in AUTOMATIC_CATEGORIES else DEFAULT_CATEGORY  # fmt: skip


def site_tags(tags: Iterable[str], *extra: Iterable[str]) -> list[str]:
    """3-6 lowercase tags: the SEO tags, then keywords one by one until there are three."""
    out: list[str] = []
    for index, group in enumerate((tags, *extra)):
        for raw in group:
            if index and len(out) >= MIN_TAGS:
                break
            tag = " ".join(raw.lower().split())[:MAX_TAG_LENGTH].strip(" -")
            if tag and tag not in out:
                out.append(tag)
    return out[:MAX_TAGS]


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):  # coverWidth / coverHeight: the site reads them as numbers
        return str(value)
    text = str(value).replace("\n", " ").strip()
    return "'" + text.replace("'", "''") + "'"


def cover_extension(filename: str, mime: str) -> str:
    """The extension the cover file keeps in the repository, from its name or its type."""
    suffix = filename[filename.rfind(".") :].lower() if "." in filename else ""
    if suffix in (".png", ".jpg", ".jpeg", ".webp"):
        return suffix
    return {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime.split(";")[0].strip().lower(), ".png")  # fmt: skip


def frontmatter_text(fields: dict[str, Any]) -> str:
    """YAML in the site's own style: single-quoted strings, block lists, keys in the
    repository's order."""
    lines = ["---"]
    for key in FRONTMATTER_ORDER:
        if key not in fields:
            continue
        value = fields[key]
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines += [f"  - {yaml_scalar(item)}" for item in value]
        else:
            lines.append(f"{key}: {yaml_scalar(value)}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def split_frontmatter(text: str) -> tuple[dict[str, Any] | None, str]:
    """(frontmatter, body) of an MDX file; (None, text) when there is no frontmatter or
    it doesn't parse. Reading is lenient (dates may be unquoted in older files)."""
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---", 4)
    if end < 0:
        return None, text
    raw = text[4:end]
    body = text[end + 4 :].lstrip("\n")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return None, text
    return (data if isinstance(data, dict) else None), body


def reading_minutes(body: str) -> int:
    """The site's formula over the raw MDX body (component tags count as words)."""
    words = len(body.split())
    return max(1, -(-words // 225))


def jsx_attr(value: str) -> str:
    text = " ".join(str(value).split())
    for raw, ref in (("&", "&amp;"), ('"', "&quot;"), ("<", "&lt;"), (">", "&gt;"), ("{", "&#123;"), ("}", "&#125;")):  # fmt: skip
        text = text.replace(raw, ref)
    return text


def cta_block(config: SiteConfig) -> str:
    return "\n".join(
        [
            "<BlogCTA",
            f'  title="{jsx_attr(config.cta_title)}"',
            f'  body="{jsx_attr(config.cta_body)}"',
            f'  ctaLabel="{jsx_attr(config.cta_label)}"',
            f'  ctaHref="{jsx_attr(config.cta_href)}"',
            "/>",
        ]
    )


def site_path(url: str, config: SiteConfig, allowed_paths: Collection[str]) -> str | None:
    """A site-relative path for an internal link, if the page exists on the site."""
    parts = urlsplit(url.strip())
    if parts.scheme and (parts.hostname or "").lower() != config.site_host:
        return None
    if not parts.scheme and not url.startswith("/"):
        return None
    path = parts.path.rstrip("/") or "/"
    if path in allowed_paths:
        return path + (f"?{parts.query}" if parts.query and path in config.query_paths else "")
    return None


def filter_internal_links(body: str, config: SiteConfig, allowed_paths: Collection[str]) -> tuple[str, list[str], list[str]]:  # fmt: skip
    """Internal links become site-relative paths; ones to pages that don't exist on the
    site become plain text. External http(s) links are kept (they were validated)."""
    kept: list[str] = []
    dropped: list[str] = []

    def replace(match: re.Match[str]) -> str:
        text, url = match.group(1), match.group(2)
        parts = urlsplit(url)
        if parts.scheme in ("http", "https") and (parts.hostname or "").lower() != config.site_host:  # fmt: skip
            return match.group(0)
        path = site_path(url, config, allowed_paths)
        if path is None:
            dropped.append(url)
            return text
        kept.append(path)
        return f"[{text}]({path})"

    return _MD_LINK.sub(replace, body), kept, dropped


def related_reading(links: Iterable[RenderedLink], config: SiteConfig, allowed_paths: Collection[str], *, already: Collection[str]) -> tuple[str, list[str]]:  # fmt: skip
    items: list[tuple[str, str]] = []
    for link in links:
        if link.kind != "internal" or link.placed:
            continue
        path = site_path(link.url, config, allowed_paths)
        if path is None or path in already or any(p == path for _, p in items):
            continue
        items.append((link.anchor_text, path))
    if not items:
        return "", []
    text = f"## {RELATED_HEADING}\n\n" + "\n".join(f"- [{link_text(anchor)}]({link_url(path)})" for anchor, path in items)  # fmt: skip
    return text, [p for _, p in items]


# ── the document ─────────────────────────────────────────────────────────────


def compose(document: RenderedDocument, *, marker: str, config: SiteConfig, allowed_paths: Collection[str], published_on: date, updated_on: date | None = None) -> MDXDocument:  # fmt: skip
    """The complete MDX file for one article version."""
    if not _MARKER.match(marker):
        raise ValueError("the publication marker must be 32 hex characters")
    slug = site_slug(document.slug)
    if not slug:
        raise ValueError(f"no usable slug from {document.slug!r}")
    category = site_category(document.content_type, allowed=config.categories)
    tags = site_tags(document.tags, [document.primary_keyword], document.secondary_keywords)
    fields: dict[str, Any] = {
        "title": " ".join(document.title.split()),
        "description": " ".join(document.excerpt.split()),
        "publishedAt": published_on.isoformat(),
        "author": config.author_name,
        "authorRole": config.author_role,
        "authorInitials": config.author_initials,
        "category": category,
        "tags": tags,
        "draft": False,
        "agentPublication": marker,
        "agentSource": AGENT_SOURCE,
    }
    if updated_on is not None:
        fields["updatedAt"] = updated_on.isoformat()
    if config.author_linkedin:
        fields["authorLinkedin"] = config.author_linkedin
    cover_path: str | None = None
    if document.cover is not None:
        suffix = cover_extension(document.cover.filename, document.cover.mime)
        cover_path = f"{config.cover_dir}/{slug}{suffix}"
        fields["coverImage"] = f"{config.cover_url_prefix}/{slug}{suffix}"
        if document.cover.width and document.cover.height:
            fields["coverWidth"], fields["coverHeight"] = document.cover.width, document.cover.height  # fmt: skip
    body, kept, dropped = filter_internal_links(document.body_markdown, config, allowed_paths)
    related, related_paths = related_reading(document.links, config, allowed_paths, already=kept)
    parts = [body.strip(), related, cta_block(config), "---", f"*{jsx_attr(config.byline).replace('*', '')}*"]  # fmt: skip
    body_text = "\n\n".join(p for p in parts if p) + "\n"
    text = frontmatter_text(fields) + "\n" + body_text
    notes = [f"internal link left out (not on the site): {url}" for url in dropped]
    if len(tags) < MIN_TAGS:
        notes.append(f"only {len(tags)} tag(s) available (the site prefers 3-6)")
    return MDXDocument(
        slug=slug,
        path=f"{config.content_dir}/{slug}.mdx",
        branch=f"{config.branch_prefix}{slug}",
        title=fields["title"],
        frontmatter=fields,
        text=text,
        reading_minutes=reading_minutes(body_text),
        expected_url=f"{config.site_url}/blog/{slug}",
        internal_links=tuple(kept + related_paths),
        dropped_links=tuple(dropped),
        notes=tuple(notes),
        cover_path=cover_path,
    )


def body_digest(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def pair_json(fields: dict[str, Any]) -> str:
    """The post's JSON in the repository's style: two-space indent, the site's key order."""
    ordered = {k: fields[k] for k in PAIR_ORDER if k in fields}
    ordered.update({k: v for k, v in fields.items() if k not in ordered})
    return json.dumps(ordered, indent=2, ensure_ascii=False) + "\n"


def parse_fields(text: str, layout: str = FRONTMATTER) -> dict[str, Any] | None:
    """A post file's fields: its frontmatter, or (json_pair) its JSON object."""
    if layout != JSON_PAIR:
        return split_frontmatter(text)[0]
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _fit(text: str, limit: int) -> str:
    """``text`` cut at a word boundary to at most ``limit`` characters."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit + 1].rsplit(" ", 1)[0] if " " in text[:limit] else text[:limit]
    return cut.rstrip(" ,;:-\u2013\u2014")  # and en/em dashes


def pair_related(paths: Iterable[str]) -> dict[str, list[str]]:
    """Linked treatment and concern pages, as the slugs the post's related cards take."""
    related: dict[str, list[str]] = {name: [] for name, _ in PAIR_RELATED}
    for path in paths:
        for name, prefix in PAIR_RELATED:
            slug = path[len(prefix) :] if path.startswith(prefix) else ""
            if slug and _PAIR_SLUG.match(slug) and slug not in related[name] and len(related[name]) < MAX_RELATED:  # fmt: skip
                related[name].append(slug)
    return related


def compose_pair(document: RenderedDocument, *, marker: str, config: SiteConfig, allowed_paths: Collection[str], published_on: date) -> MDXDocument:  # fmt: skip
    """The JSON and the MDX body of one article version (the json_pair layout)."""
    if not _MARKER.match(marker):
        raise ValueError("the publication marker must be 32 hex characters")
    slug = site_slug(document.slug)
    if not slug:
        raise ValueError(f"no usable slug from {document.slug!r}")
    title = " ".join(document.title.split())
    suffix = config.meta_title_suffix
    if suffix and not suffix[0].isspace():  # an env var may lose its leading space
        suffix = " " + suffix
    meta_title = _fit(document.meta_title or title, META_TITLE_RANGE[1] - len(suffix)) + suffix
    description = _fit(document.excerpt, META_DESCRIPTION_RANGE[1])
    body, kept, dropped = filter_internal_links(document.body_markdown, config, allowed_paths)
    suggested = [p for link in document.links if link.kind == "internal" and (p := site_path(link.url, config, allowed_paths))]  # fmt: skip
    related = pair_related([*kept, *suggested])
    body_text = body.strip() + "\n"
    notes = [f"internal link left out (not on the site): {url}" for url in dropped]
    cover_path: str | None = None
    hero: dict[str, Any] | None = None
    if document.cover is not None and document.cover.width and document.cover.height:
        suffix_ext = cover_extension(document.cover.filename, document.cover.mime)
        cover_path = f"{config.cover_dir}/{slug}{suffix_ext}"
        hero = {"src": f"{config.cover_url_prefix}/{slug}{suffix_ext}", "alt": " ".join((document.cover.alt or title).split()), "width": document.cover.width, "height": document.cover.height}  # fmt: skip
    elif related["relatedTreatments"]:
        first = related["relatedTreatments"][0]
        width, height = TREATMENT_IMAGE_SIZE
        hero = {"src": TREATMENT_IMAGE.format(slug=first), "alt": title, "width": width, "height": height}  # fmt: skip
        notes.append(f"no cover photo: the post uses the picture of /treatments/{first}")
    fields: dict[str, Any] = {
        "slug": slug,
        "title": title,
        "metaTitle": meta_title,
        "metaDescription": description,
        "date": published_on.isoformat(),
        "excerpt": " ".join(document.excerpt.split()),
        "author": config.author_name,
        "readingMinutes": reading_minutes(body_text),
        **({"heroImage": hero} if hero else {}),
        **related,
        "agentPublication": marker,
        "agentSource": AGENT_SOURCE,
        "agentBody": body_digest(body_text),
    }
    return MDXDocument(
        slug=slug,
        path=f"{config.content_dir}/{slug}.json",
        branch=f"{config.branch_prefix}{slug}",
        title=title,
        frontmatter=fields,
        text=pair_json(fields),
        reading_minutes=fields["readingMinutes"],
        expected_url=f"{config.site_url}/blog/{slug}",
        internal_links=tuple(dict.fromkeys(kept)),
        dropped_links=tuple(dropped),
        notes=tuple(notes),
        cover_path=cover_path,
        body_path=f"{config.content_dir}/{slug}.mdx",
        body_text=body_text,
    )


def validate_pair(text: str, body: str, *, marker: str) -> list[str]:
    """Why the JSON or the body could break the site's build or the contract (the site's
    own schema, checked here first); empty when they can't."""
    fields = parse_fields(text, JSON_PAIR)
    if fields is None:
        return ["the post's JSON doesn't parse to an object"]
    problems: list[str] = []
    for key in ("slug", "title", "metaTitle", "metaDescription", "date", "excerpt", "author"):
        if not isinstance(fields.get(key), str) or not fields[key].strip():
            problems.append(f"json: {key} must be a non-empty string")
    if isinstance(fields.get("slug"), str) and not _PAIR_SLUG.match(fields["slug"]):
        problems.append("json: slug must be lowercase kebab-case")
    for key, (low, high) in (("metaTitle", META_TITLE_RANGE), ("metaDescription", META_DESCRIPTION_RANGE)):  # fmt: skip
        value = fields.get(key)
        if isinstance(value, str) and not low <= len(value) <= high:
            problems.append(
                f"json: {key} is {len(value)} characters (the site allows {low}-{high})"
            )
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(fields.get("date") or "")):
        problems.append("json: date must be YYYY-MM-DD")
    hero = fields.get("heroImage")
    if not isinstance(hero, dict):
        problems.append("json: heroImage is required (no cover photo, and no related treatment to borrow a picture from)")  # fmt: skip
    else:
        if not isinstance(hero.get("src"), str) or not hero["src"].startswith("/") or any(c.isspace() for c in hero["src"]):  # fmt: skip
            problems.append("json: heroImage.src must be a site-absolute path")
        if not isinstance(hero.get("alt"), str):
            problems.append("json: heroImage.alt must be a string")
        for key in ("width", "height"):
            value = hero.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                problems.append(f"json: heroImage.{key} must be a positive integer")
    for name, _ in PAIR_RELATED:
        value = fields.get(name, [])
        if not isinstance(value, list) or not all(isinstance(s, str) and _PAIR_SLUG.match(s) for s in value):  # fmt: skip
            problems.append(f"json: {name} must be a list of slugs")
    if fields.get("agentPublication") != marker:
        problems.append("json: agentPublication must be this publication's marker")
    if fields.get("agentBody") != body_digest(body):
        problems.append("json: agentBody doesn't match the body file")
    if body.startswith("---"):
        problems.append("body: starts with a frontmatter fence (the site would render it)")
    if _ESM_LINE.search(body):
        problems.append("body: a line starts with import/export (MDX would treat it as code)")
    if "{" in body or "}" in body:
        problems.append("body: a brace (MDX would evaluate it)")
    if _JSX_OPEN.search(body):
        problems.append("body: a '<' before a letter (MDX would read a tag)")
    if re.search(r"^# ", body, re.MULTILINE):
        problems.append("body: an H1 (the site renders the title)")
    return problems


def validate(text: str, *, marker: str, allowed_components: Collection[str] = ("BlogCTA",)) -> list[str]:  # fmt: skip
    """Why the file could break the site's build or the contract; empty when it can't."""
    problems: list[str] = []
    fields, body = split_frontmatter(text)
    if fields is None:
        return ["the frontmatter is missing or doesn't parse"]
    for key in ("title", "description", "publishedAt", "category"):
        if not isinstance(fields.get(key), str) or not fields[key].strip():
            problems.append(f"frontmatter: {key} must be a non-empty string")
    if isinstance(fields.get("publishedAt"), str):
        try:
            datetime.strptime(fields["publishedAt"], "%Y-%m-%d")
        except ValueError:
            problems.append("frontmatter: publishedAt must be YYYY-MM-DD")
    if fields.get("category") not in CATEGORIES:
        problems.append(f"frontmatter: category {fields.get('category')!r} isn't one of the site's")
    if fields.get("draft") is not False:
        problems.append("frontmatter: draft must be false (a draft is invisible even on previews)")
    if fields.get("agentPublication") != marker:
        problems.append("frontmatter: agentPublication must be this publication's marker")
    tags = fields.get("tags")
    if not isinstance(tags, list) or not all(isinstance(t, str) and t == t.lower() for t in tags):
        problems.append("frontmatter: tags must be a list of lowercase strings")
    problems += _cover_problems(fields)
    if _ESM_LINE.search(body):
        problems.append("body: a line starts with import/export (MDX would treat it as code)")
    stripped, opened = _without_components(body, allowed_components)
    if opened != 1:
        problems.append(f"body: expected exactly one <BlogCTA />, found {opened}")
    if "{" in stripped or "}" in stripped:
        problems.append("body: a brace outside a component (MDX would evaluate it)")
    if _JSX_OPEN.search(stripped):
        problems.append("body: a '<' before a letter outside a component (MDX would read a tag)")
    if re.search(r"^# ", body, re.MULTILINE):
        problems.append("body: an H1 (the site renders the title)")
    return problems


def _cover_problems(fields: dict[str, Any]) -> list[str]:
    """The optional cover keys: a site-absolute image path, and two positive integers the
    site reads as numbers (a quoted "1536" would be ignored there)."""
    cover = fields.get("coverImage")
    problems: list[str] = []
    if cover is not None and (not isinstance(cover, str) or not cover.startswith("/") or any(c.isspace() for c in cover) or not cover.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):  # fmt: skip
        problems.append("frontmatter: coverImage must be a site-absolute image path such as /blog/covers/<slug>.png")  # fmt: skip
    for key in ("coverWidth", "coverHeight"):
        value = fields.get(key)
        if value is None:
            continue
        if cover is None:
            problems.append(f"frontmatter: {key} without a coverImage")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            problems.append(f"frontmatter: {key} must be a positive number (unquoted)")
    return problems


def _without_components(body: str, allowed: Collection[str]) -> tuple[str, int]:
    """The body without the self-closing components we emit, and how many there were."""
    count = 0

    def drop(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return ""

    names = "|".join(re.escape(n) for n in allowed)
    stripped = re.sub(rf"<(?:{names})\b(?:\s+[A-Za-z]+=\"[^\"]*\")*\s*/>", drop, body)
    return stripped, count


def cover_credit(cover: RenderedCover | None) -> str:
    """The cover's provenance for the reviewer: who took the picture and where it came
    from. A generated illustration has no credit line; Pexels asks for none but gets one
    anyway. The photographer's name is someone else's text, so it is flattened to one
    plain line and the links are only the ones the source itself gave us."""
    if cover is None or cover.source == CoverImageSource.GEMINI.value:
        return ""
    who = " ".join(re.sub(r"[\[\]()`|]+", " ", cover.credit or "").split())[:100]
    where = COVER_SOURCE_NAMES.get(cover.source, cover.source)
    if not who:
        return f"- Cover photo: {where}" + (f" — {cover.source_url}" if cover.source_url else "")
    name = f"[{who}]({cover.credit_url})" if cover.credit_url else who
    photo = f" — {cover.source_url}" if cover.source_url else ""
    return f"- Cover photo: {name} on {where}{photo}"


def pr_body(document: RenderedDocument, mdx: MDXDocument, *, generated_at: datetime) -> str:
    """The pull request description: what the article is and where it came from. Never a
    credential; only what the document itself carries."""
    sources = [s for s in document.sources if s.url]
    lines = [
        f"**{mdx.title}**",
        "",
        f"- Slug: `{mdx.slug}` → {mdx.expected_url}",
        f"- Category: {mdx.frontmatter['category']} · Tags: {', '.join(mdx.frontmatter['tags']) or '—'}"
        if "category" in mdx.frontmatter
        else f"- Related treatments: {', '.join(mdx.frontmatter.get('relatedTreatments') or []) or '—'} · concerns: {', '.join(mdx.frontmatter.get('relatedConcerns') or []) or '—'}",
        f"- Quality score: {document.quality_score:.1f}/100"
        if document.quality_score is not None
        else (
            "- Quality score: none (written by a person, not scored by the agent)"
            if document.authored
            else "- Quality score: n/a"
        ),
        f"- Content opportunity: {document.opportunity_title or '—'}",
        f"- Article: #{document.article_id or '?'} (version {document.version_id or '?'}), {document.word_count} words, ~{mdx.reading_minutes} min read",
        f"- Sources cited: {len(sources)}",
    ]
    lines += [f"  - [{s.title}]({s.url})" for s in sources[:20]]
    if len(sources) > 20:
        lines.append(f"  - … and {len(sources) - 20} more")
    credit = cover_credit(document.cover)
    if credit:
        lines.append(credit)
    origin = (
        "Written by hand and imported into the competitor-analysis agent, which checked its "
        "length, its citations, its structure and this file against the site's contract, and "
        "then approved it. It was **not** fact-checked, originality-checked or scored by the "
        "agent's Gemini quality gates: check the claims and their sources yourself."
        if document.authored
        else "Generated by the competitor-analysis agent (Gemini article generation, "
        "fact-checked and scored by its quality gates, approved before publication)."
    )
    lines += [
        f"- {'Imported' if document.authored else 'Generated'}: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"{origin} The author byline is the configured team byline.",
        "",
        "Review the Vercel preview deployment of this branch before merging: the post appears "
        f"at `<preview-url>/blog/{mdx.slug}`. Merging (squash) publishes it at {mdx.expected_url}.",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "AGENT_SOURCE",
    "AUTOMATIC_CATEGORIES",
    "CATEGORIES",
    "CATEGORY_BY_FORMAT",
    "COVER_KEYS",
    "COVER_SOURCE_NAMES",
    "DEFAULT_CATEGORY",
    "DEFAULT_COVER_DIR",
    "DEFAULT_COVER_URL_PREFIX",
    "FRONTMATTER",
    "JSON_PAIR",
    "MDXDocument",
    "SiteConfig",
    "body_digest",
    "compose",
    "compose_pair",
    "cover_credit",
    "cover_extension",
    "cta_block",
    "filter_internal_links",
    "frontmatter_text",
    "pair_json",
    "pair_related",
    "parse_fields",
    "pr_body",
    "reading_minutes",
    "site_category",
    "site_path",
    "site_slug",
    "site_tags",
    "split_frontmatter",
    "validate",
    "validate_pair",
]
