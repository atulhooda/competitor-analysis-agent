"""Deterministic handling of generated article content (Phase 5). No LLM.

- normalizes the model's structured output into ``ArticleContent``;
- canonicalizes citation markers ([S1, S3] → [S1][S3], placed before the sentence's final
  punctuation), removes labels that don't match a stored source, and records each removal;
- extracts claim → source citations per sentence, for storage and later fact-checking;
- counts words, flags sentences with numbers found in neither the research nor the brief,
  and runs the structural checks an article must pass before it is marked completed;
- builds slugs and a Markdown preview (a review aid, not a publishing format).
"""

import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping

from app.domain.articles import (
    ArticleContent,
    BlockType,
    Citation,
    ContentBlock,
    ContentIssue,
    ContentSection,
    SectionKind,
)
from app.prompts.article_draft import ArticleContentOut
from app.services.numbers import numbers_in

_MARKER = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]")
_LABEL = re.compile(r"S\d+")
# One or more markers right after sentence punctuation: "... hours. [S2]" → "... hours [S2]."
_TRAILING_MARKERS = re.compile(r"([.!?])((?:\s*\[S\d+\])+)")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[\w'\u2019-]+")  # \u2019: typographic apostrophe
MAX_SLUG_LENGTH = 80


def from_output(out: ArticleContentOut) -> ArticleContent:
    """The model's output as content: empty blocks and sections dropped, text trimmed."""
    sections = []
    for section in out.sections:
        blocks = []
        for block in section.blocks:
            if block.type is BlockType.LIST:
                if block.items:
                    blocks.append(ContentBlock(type=BlockType.LIST, items=block.items, ordered=block.ordered))  # fmt: skip
            elif block.text:
                blocks.append(ContentBlock(type=block.type, text=block.text))
        if blocks:
            sections.append(ContentSection(kind=section.kind, heading=section.heading, blocks=blocks))  # fmt: skip
    return ArticleContent(title=out.title.strip(), description=out.description.strip(), sections=sections)  # fmt: skip


def _map_texts(content: ArticleContent, fn: Callable[[str], str]) -> ArticleContent:
    sections = []
    for section in content.sections:
        blocks = [
            block.model_copy(
                update={
                    "text": fn(block.text) if block.text is not None else None,
                    "items": [fn(i) for i in block.items],
                }
            )
            for block in section.blocks
        ]
        sections.append(section.model_copy(update={"blocks": blocks}))
    return content.model_copy(update={"sections": sections})


def _texts(content: ArticleContent) -> Iterable[tuple[int, int, int | None, str]]:
    for s, section in enumerate(content.sections):
        for b, block in enumerate(section.blocks):
            if block.text is not None:
                yield s, b, None, block.text
            for i, item in enumerate(block.items):
                yield s, b, i, item


def text_blocks(content: ArticleContent) -> Iterable[tuple[int, int, int | None, str]]:
    """(section, block, list item or None, text) for every paragraph, list item and subheading."""
    return _texts(content)


def split_sentences(text: str) -> list[str]:
    return _SENTENCE.split(text)


def clean_citations(content: ArticleContent, valid: set[str]) -> tuple[ArticleContent, list[ContentIssue]]:  # fmt: skip
    """Canonical markers, with labels that aren't in ``valid`` removed and reported."""
    issues: list[ContentIssue] = []

    def rewrite(text: str) -> str:
        def marker(match: re.Match[str]) -> str:
            labels = list(dict.fromkeys(_LABEL.findall(match.group(1))))
            unknown = [label for label in labels if label not in valid]
            for label in unknown:
                issues.append(ContentIssue(kind="unknown_citation_removed", detail=f"[{label}] doesn't match any stored source", excerpt=text[:200]))  # fmt: skip
            return "".join(f"[{label}]" for label in labels if label in valid)

        text = _MARKER.sub(marker, text)
        text = _TRAILING_MARKERS.sub(lambda m: " " + m.group(2).strip() + m.group(1), text)
        return re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()

    return _map_texts(content, rewrite), issues


def strip_markers(text: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", _MARKER.sub("", text)).strip()


def citations(content: ArticleContent) -> list[Citation]:
    """Every sentence (or list item) that carries citation markers, with its labels."""
    found = []
    for s, b, i, text in _texts(content):
        for sentence in _SENTENCE.split(text):
            labels = list(dict.fromkeys(label for m in _MARKER.finditer(sentence) for label in _LABEL.findall(m.group(1))))  # fmt: skip
            if labels:
                found.append(Citation(section=s, block=b, item=i, claim=strip_markers(sentence), labels=labels))  # fmt: skip
    return found


def word_count(content: ArticleContent) -> int:
    return sum(len(_WORD.findall(strip_markers(text))) for *_, text in _texts(content))


def number_issues(content: ArticleContent, allowed: set[str]) -> list[ContentIssue]:
    """Sentences with a number that appears in neither the research nor the brief: flagged
    for review (Phase 6 checks facts), never silently kept or removed."""
    issues = []
    for *_, text in _texts(content):
        for sentence in _SENTENCE.split(strip_markers(text)):
            unknown = numbers_in(sentence) - allowed
            if unknown:
                issues.append(ContentIssue(kind="number_not_in_research", detail=f"number(s) {', '.join(sorted(unknown))} not found in the research or the brief", excerpt=sentence[:200]))  # fmt: skip
    return issues


def structural_problems(content: ArticleContent, *, min_words: int, labels: set[str]) -> list[str]:  # fmt: skip
    """What stops content from being an article: the baseline checked before completion
    (and, without the length check, before a draft is accepted). Not a quality score."""
    problems = []
    if not content.title.strip():
        problems.append("the title is empty")
    body = [s for s in content.sections if s.kind is SectionKind.BODY]
    if len(content.sections) < 2 or not body:
        problems.append(f"{len(content.sections)} section(s), {len(body)} of them body sections: at least 2 sections, with at least one body section, are required")  # fmt: skip
    for s, section in enumerate(content.sections):
        if section.kind is SectionKind.BODY and not (section.heading or "").strip():
            problems.append(f"section {s + 1} has no heading")
        for b, block in enumerate(section.blocks):
            if block.type is BlockType.LIST and not block.items:
                problems.append(f"section {s + 1}, block {b + 1}: empty list")
            elif block.type is not BlockType.LIST and not (block.text or "").strip():
                problems.append(f"section {s + 1}, block {b + 1}: empty {block.type.value}")
    words = word_count(content)
    if words < min_words:
        problems.append(f"{words} words; at least {min_words} are required")
    for citation in citations(content):
        missing = [label for label in citation.labels if label not in labels]
        if missing:
            problems.append(f"citation(s) {', '.join(missing)} don't match a stored source")
    return problems


def slugify(title: str) -> str:
    """A URL-safe slug: ASCII, lowercase, hyphens, at most 80 characters (never empty)."""
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")
    if len(slug) > MAX_SLUG_LENGTH:
        slug = slug[:MAX_SLUG_LENGTH].rsplit("-", 1)[0] or slug[:MAX_SLUG_LENGTH]
    return slug or "article"


def unique_slug(base: str, taken: set[str]) -> str:
    """``base``, or ``base-2``, ``base-3``, ... whichever isn't taken."""
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def to_markdown(content: ArticleContent) -> str:
    """The content as Markdown with its citation labels kept as written ([S1]), for prompts."""
    lines = [f"# {content.title}", ""]
    for section in content.sections:
        if section.heading:
            lines += [f"## {section.heading}", ""]
        for block in section.blocks:
            if block.type is BlockType.SUBHEADING:
                lines += [f"### {block.text or ''}", ""]
            elif block.type is BlockType.LIST:
                lines += [f"{f'{n}.' if block.ordered else '-'} {item}" for n, item in enumerate(block.items, start=1)] + [""]  # fmt: skip
            else:
                lines += [block.text or "", ""]
    return "\n".join(lines).rstrip() + "\n"


def render_markdown(content: ArticleContent, sources: Mapping[str, tuple[str | None, str]]) -> str:
    """A Markdown preview for review: markers become numbered references, listed at the end.
    ``sources`` maps a label to (title, url)."""
    order: dict[str, int] = {}

    def refs(text: str) -> str:
        def marker(match: re.Match[str]) -> str:
            numbers = [order.setdefault(label, len(order) + 1) for label in _LABEL.findall(match.group(1)) if label in sources]  # fmt: skip
            return "".join(f"[{n}]" for n in numbers)

        return _MARKER.sub(marker, text)

    lines = [f"# {content.title}", "", f"*{content.description}*", ""]
    for section in content.sections:
        if section.heading:
            lines += [f"## {section.heading}", ""]
        for block in section.blocks:
            if block.type is BlockType.SUBHEADING:
                lines += [f"### {refs(block.text or '')}", ""]
            elif block.type is BlockType.LIST:
                lines += [f"{f'{n}.' if block.ordered else '-'} {refs(item)}" for n, item in enumerate(block.items, start=1)] + [""]  # fmt: skip
            else:
                lines += [refs(block.text or ""), ""]
    if order:
        lines += ["## Sources", ""]
        for label, n in sorted(order.items(), key=lambda kv: kv[1]):
            title, url = sources[label]
            lines.append(f"{n}. {title or url} — {url}")
    return "\n".join(lines).rstrip() + "\n"
