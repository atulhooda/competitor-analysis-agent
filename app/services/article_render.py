"""CMS-neutral rendering (Phase 7): one validated article version → ``RenderedDocument``
(safe HTML plus its metadata), which a CMS adapter maps onto its own fields.

- **Safe HTML.** The article is structured text (sections, paragraphs, lists, subheadings),
  never HTML, so every piece of text is escaped and the tags come from this module only.
  A model can't inject markup or scripts.
- **Structure.** H1 is the title (left to the CMS, or included for previews), H2 the section
  headings, H3 the subheadings; paragraphs; ordered and unordered lists.
- **Citations.** ``[S#]`` markers become numbered references (``[1]``) linking to a Sources
  section that lists only the sources this version cites, with their stored URLs. Labels
  without a stored source are dropped and reported. No database id is published.
- **Links.** Only Phase 6's validated links, and only to allowed targets (stored pages of
  your site; the article's stored research sources); http(s) only. Each is placed inline on
  the first occurrence of its anchor text; internal links that don't fit are listed under
  "Related reading". No URL is ever added here.
- **FAQ** from the SEO package. **Images:** no image is generated or invented; the
  suggestion (concept, alt text) travels as metadata.
"""

import hashlib
import html
import json
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from app.domain.articles import ArticleContent, BlockType
from app.domain.publishing import RenderedDocument, RenderedFAQ, RenderedLink, RenderedSource
from app.domain.quality import SEOPackage
from app.services.article_content import slugify, word_count

RENDER_VERSION = "render/1"
# One or more adjacent markers: "[S1][S2]" or "[S1, S3]" become one group of references.
_MARKERS = re.compile(r"(?:\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]\s*)*\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]")
_LABEL = re.compile(r"S\d+")
_PUNCT_BEFORE = re.compile(r"\s+([.,;:!?])")


def safe_url(url: str) -> str | None:
    """An absolute http(s) URL without credentials, or None."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:  # fmt: skip
        return None
    return url.strip()


def _e(text: str) -> str:
    return html.escape(text, quote=True)


@dataclass
class _Link:
    kind: str
    anchor: str
    url: str
    title: str | None
    placed: bool = False


class _Renderer:
    def __init__(self, sources: Mapping[str, tuple[str | None, str]], links: list[_Link]) -> None:  # fmt: skip
        self.sources = sources
        self.numbers: dict[str, int] = {}
        self.unknown: list[str] = []
        self.links = links

    def text(self, raw: str, *, allow_links: bool) -> str:
        """Escaped inline HTML for a piece of article text: citations numbered, validated
        links placed on their first occurrence."""
        out: list[str] = []
        pos = 0
        for match in _MARKERS.finditer(raw):
            # "tickets [S2]." → "tickets<sup>[1]</sup>.": the reference sits on the word
            out.append(self._plain(raw[pos : match.start()].rstrip(), allow_links=allow_links))
            out.append(self._cite(match.group(0)))
            pos = match.end()
        out.append(self._plain(raw[pos:], allow_links=allow_links))
        return _PUNCT_BEFORE.sub(r"\1", "".join(out)).strip()

    def _cite(self, labels: str) -> str:
        refs = []
        for label in dict.fromkeys(_LABEL.findall(labels)):
            if label not in self.sources:
                if label not in self.unknown:
                    self.unknown.append(label)
                continue
            n = self.numbers.setdefault(label, len(self.numbers) + 1)
            refs.append(f'<a href="#source-{n}">[{n}]</a>')
        return f'<sup class="citation">{"".join(refs)}</sup>' if refs else ""

    def _plain(self, text: str, *, allow_links: bool) -> str:
        if not allow_links or not text:
            return _e(text)
        spans: list[tuple[int, int, _Link]] = []
        for link in self.links:
            if link.placed:
                continue
            match = re.search(rf"(?<!\w){re.escape(link.anchor)}(?!\w)", text, re.IGNORECASE)
            if match and not any(s < match.end() and match.start() < e for s, e, _ in spans):
                spans.append((match.start(), match.end(), link))
                link.placed = True
        out, pos = [], 0
        for start, end, link in sorted(spans, key=lambda s: s[0]):
            rel = ' rel="noopener"' if link.kind == "external" else ""
            out += [_e(text[pos:start]), f'<a href="{_e(link.url)}"{rel}>{_e(text[start:end])}</a>']  # fmt: skip
            pos = end
        out.append(_e(text[pos:]))
        return "".join(out)


def render_article(
    content: ArticleContent,
    *,
    sources: Mapping[str, tuple[str | None, str]],
    seo: SEOPackage | None,
    allowed_internal: Collection[str] = (),
    allowed_external: Collection[str] = (),
    include_title: bool = False,
) -> RenderedDocument:
    """``sources`` maps the labels this version cites to their stored (title, url)."""
    notes: list[str] = []
    links: list[_Link] = []
    if seo is not None:
        for kind, suggestions, allowed in (("internal", seo.internal_links, allowed_internal), ("external", seo.external_links, allowed_external)):  # fmt: skip
            for s in suggestions:
                url = safe_url(s.url)
                if url is None or url not in allowed:
                    notes.append(f"{kind} link to {s.url[:120]} left out: not a validated target")
                    continue
                if s.anchor_text.strip():
                    links.append(_Link(kind, s.anchor_text.strip(), url, s.title))
    r = _Renderer({label: (title, url) for label, (title, url) in sources.items() if safe_url(url)}, links)  # fmt: skip
    parts: list[str] = [f"<h1>{_e(content.title)}</h1>"] if include_title else []
    for section in content.sections:
        if section.heading:
            parts.append(f"<h2>{_e(section.heading)}</h2>")
        for block in section.blocks:
            if block.type is BlockType.SUBHEADING and block.text:
                parts.append(f"<h3>{r.text(block.text, allow_links=False)}</h3>")
            elif block.type is BlockType.LIST and block.items:
                tag = "ol" if block.ordered else "ul"
                items = "".join(f"<li>{r.text(item, allow_links=True)}</li>" for item in block.items if item.strip())  # fmt: skip
                parts.append(f"<{tag}>{items}</{tag}>")
            elif block.type is BlockType.PARAGRAPH and block.text and block.text.strip():
                parts.append(f"<p>{r.text(block.text, allow_links=True)}</p>")
    faq = [RenderedFAQ(question=f.question, answer=f.answer) for f in (seo.faq if seo else []) if f.question.strip() and f.answer.strip()]  # fmt: skip
    if faq:
        parts.append("<h2>Frequently asked questions</h2>")
        parts += [f"<h3>{_e(f.question)}</h3>\n<p>{_e(f.answer)}</p>" for f in faq]
    related = [link for link in links if link.kind == "internal" and not link.placed]
    if related:
        parts.append("<h2>Related reading</h2>")
        parts.append("<ul>" + "".join(f'<li><a href="{_e(link.url)}">{_e(link.title or link.anchor)}</a></li>' for link in related) + "</ul>")  # fmt: skip
    for link in links:
        if link.kind == "external" and not link.placed:
            notes.append(f"external link '{link.anchor}' left out: its anchor text isn't in the article")  # fmt: skip
    rendered_sources = []
    if r.numbers:
        parts.append('<h2 id="sources">Sources</h2>')
        entries = []
        for label, n in sorted(r.numbers.items(), key=lambda kv: kv[1]):
            title, url = r.sources[label]
            name = (title or "").strip() or (urlsplit(url).hostname or url)
            entries.append(f'<li id="source-{n}"><a href="{_e(url)}" rel="noopener">{_e(name)}</a></li>')  # fmt: skip
            rendered_sources.append(RenderedSource(number=n, title=name, url=url))
        parts.append("<ol>" + "".join(entries) + "</ol>")
    if r.unknown:
        notes.append(f"citation label(s) {', '.join(r.unknown)} have no stored source: left out")  # fmt: skip
    body = "\n".join(parts) + "\n"
    slug = slugify(seo.slug if seo and seo.slug else content.title)
    excerpt = (seo.meta_description if seo else "") or content.description
    category = seo.category if seo and seo.category else None
    tags = list(seo.tags) if seo else []
    fingerprint = {"title": content.title, "slug": slug, "excerpt": excerpt, "body": body, "category": category, "tags": tags}  # fmt: skip
    return RenderedDocument(
        render_version=RENDER_VERSION,
        title=content.title,
        slug=slug,
        excerpt=excerpt,
        meta_title=seo.meta_title if seo else content.title,
        primary_keyword=seo.primary_keyword if seo else "",
        category=category,
        tags=tags,
        body_html=body,
        sources=rendered_sources,
        faq=faq,
        links=[
            RenderedLink(kind=link.kind, anchor_text=link.anchor, url=link.url, placed=link.placed)
            for link in links
        ],
        image=seo.image if seo else None,
        word_count=word_count(content),
        content_hash=hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest(),
        unknown_citations=r.unknown,
        notes=notes,
    )


__all__ = ["RENDER_VERSION", "render_article", "safe_url"]
