"""Markdown rendering of an article version (Phase 8, for file-based publishing): the same
structure the HTML renderer produces, as Markdown that MDX compiles safely.

- **Escaping.** Article text is prose, never markup. Everything MDX or Markdown could read
  as something else is neutralized before any syntax of ours is added: ``<``, ``{`` and
  ``}`` (JSX), ``import``/``export`` at the start of a paragraph (ESM), brackets and
  backticks (links, code), list, heading, quote and rule markers at the start of a line,
  and bare URLs (GFM autolinks). A model can't smuggle a link, a component or an
  expression into the page, and generated prose can't break the site's build.
- **Citations.** ``[S#]`` markers become inline links to the cited source, right where
  the claim is made: ``... within 15 seconds ([Acme blog](https://...)).`` Labels without
  a stored source are dropped and reported. No separate Sources section.
- **Links.** Only Phase 6's validated links, placed on the first occurrence of their anchor
  text; internal links that don't fit are returned for a "Related reading" list.
- **Structure.** No H1 (the site renders the title); the introduction's paragraphs come
  first without a heading; sections are H2, subheadings H3; ordered and unordered lists;
  the FAQ as an H2 with one H3 per question.
"""

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from urllib.parse import quote, urlsplit

from app.domain.articles import ArticleContent, BlockType, SectionKind
from app.domain.publishing import RenderedFAQ

_MARKERS = re.compile(r"(?:\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]\s*)*\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]")
_LABEL = re.compile(r"S\d+")
_PUNCT_BEFORE = re.compile(r"\s+([.,;:!?])")
_ENTITY = re.compile(r"&(?=[A-Za-z#][A-Za-z0-9]*;)")
_BLOCK_START = re.compile(r"^(\s*)([#>*+=|-]|\d+[.)])(?=\s|$)")
_ESM = re.compile(r"^(\s*)(import|export)(?=\s)")
_AUTOLINK = re.compile(r"(?i)\b(https?|ftp|mailto):(?=//|\S)")
_WWW = re.compile(r"(?i)\bwww\.")
_EMAIL = re.compile(r"(?<=\S)@(?=\S)")
FAQ_HEADING = "Frequently asked questions"


def mdx_escape(text: str) -> str:
    """Prose that MDX and Markdown render as exactly this text."""
    out = text.replace("\\", "\\\\")
    out = _ENTITY.sub("&amp;", out)
    for raw, ref in (("<", "&lt;"), (">", "&gt;"), ("{", "&#123;"), ("}", "&#125;")):
        out = out.replace(raw, ref)
    for raw in ("[", "]", "`", "*", "_", "~"):
        out = out.replace(raw, "\\" + raw)
    out = _BLOCK_START.sub(_escape_block_start, out)
    out = _ESM.sub(lambda m: f"{m.group(1)}&#{ord(m.group(2)[0])};{m.group(2)[1:]}", out)
    out = _AUTOLINK.sub(lambda m: f"{m.group(1)}&#58;", out)
    out = _WWW.sub("www&#46;", out)
    return _EMAIL.sub("&#64;", out)


def _escape_block_start(match: re.Match[str]) -> str:
    indent, token = match.group(1), match.group(2)
    if token[0].isdigit():
        return f"{indent}{token[:-1]}\\{token[-1]}"  # "1. x" → "1\. x"
    return f"{indent}\\{token}"


def link_text(text: str) -> str:
    return mdx_escape(text).replace("\n", " ").strip()


def link_url(url: str) -> str:
    """A URL safe inside ``[text](url)``: spaces and parentheses percent-encoded."""
    return quote(url.strip(), safe=":/?#[]@!$&'*+,;=%-._~")


def source_name(title: str | None, url: str) -> str:
    name = " ".join((title or "").split())
    if not name:
        name = urlsplit(url).hostname or url
    return name if len(name) <= 70 else name[:67].rstrip() + "…"


@dataclass
class MarkdownLink:
    kind: str  # internal | external
    anchor: str
    url: str
    title: str | None
    placed: bool = False


@dataclass(frozen=True)
class MarkdownBody:
    text: str
    links: list[MarkdownLink]
    unknown_citations: list[str]
    headings: list[str]  # the H2 texts, in order (for verification)


class _Renderer:
    def __init__(self, sources: Mapping[str, tuple[str | None, str]], links: list[MarkdownLink]) -> None:  # fmt: skip
        self.sources = sources
        self.links = links
        self.unknown: list[str] = []

    def text(self, raw: str, *, allow_links: bool) -> str:
        out: list[str] = []
        pos = 0
        for match in _MARKERS.finditer(raw):
            out.append(self._plain(raw[pos : match.start()].rstrip(), allow_links=allow_links))
            cite = self._cite(match.group(0))
            if cite:
                out.append(" " + cite)
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
            title, url = self.sources[label]
            refs.append(f"[{link_text(source_name(title, url))}]({link_url(url)})")
        return f"({', '.join(refs)})" if refs else ""

    def _plain(self, text: str, *, allow_links: bool) -> str:
        if not text:
            return ""
        if not allow_links:
            return mdx_escape(text)
        spans: list[tuple[int, int, MarkdownLink]] = []
        for link in self.links:
            if link.placed:
                continue
            match = re.search(rf"(?<!\w){re.escape(link.anchor)}(?!\w)", text, re.IGNORECASE)
            if match and not any(s < match.end() and match.start() < e for s, e, _ in spans):
                spans.append((match.start(), match.end(), link))
                link.placed = True
        out, pos = [], 0
        for start, end, link in sorted(spans, key=lambda s: s[0]):
            out += [mdx_escape(text[pos:start]), f"[{link_text(text[start:end])}]({link_url(link.url)})"]  # fmt: skip
            pos = end
        out.append(mdx_escape(text[pos:]))
        return "".join(out)


def render_markdown_body(
    content: ArticleContent,
    *,
    sources: Mapping[str, tuple[str | None, str]],
    links: Collection[MarkdownLink] = (),
    faq: Collection[RenderedFAQ] = (),
) -> MarkdownBody:
    """The article body as MDX-safe Markdown. ``sources`` maps the labels this version
    cites to their stored (title, url); ``links`` are Phase 6's validated links (their
    ``placed`` flag is set here)."""
    link_list = list(links)
    r = _Renderer(dict(sources), link_list)
    parts: list[str] = []
    headings: list[str] = []
    for index, section in enumerate(content.sections):
        intro = index == 0 and section.kind is SectionKind.INTRODUCTION
        if section.heading and not intro:
            headings.append(section.heading)
            parts.append(f"## {mdx_escape(section.heading)}")
        for block in section.blocks:
            if block.type is BlockType.SUBHEADING and block.text:
                parts.append(f"### {r.text(block.text, allow_links=False)}")
            elif block.type is BlockType.LIST and block.items:
                items = [item for item in block.items if item.strip()]
                lines = [f"{f'{n}.' if block.ordered else '-'} {r.text(item, allow_links=True)}" for n, item in enumerate(items, start=1)]  # fmt: skip
                parts.append("\n".join(lines))
            elif block.type is BlockType.PARAGRAPH and block.text and block.text.strip():
                parts.append(r.text(block.text, allow_links=True))
    questions = [f for f in faq if f.question.strip() and f.answer.strip()]
    if questions:
        headings.append(FAQ_HEADING)
        parts.append(f"## {FAQ_HEADING}")
        for item in questions:
            parts.append(f"### {mdx_escape(' '.join(item.question.split()))}")
            parts.append(mdx_escape(" ".join(item.answer.split())))
    return MarkdownBody("\n\n".join(parts).strip() + "\n", link_list, r.unknown, headings)


__all__ = [
    "FAQ_HEADING",
    "MarkdownBody",
    "MarkdownLink",
    "link_text",
    "link_url",
    "mdx_escape",
    "render_markdown_body",
    "source_name",
]
