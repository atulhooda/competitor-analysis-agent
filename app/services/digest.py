"""Deterministic preprocessing for the analyzer: captured version → bounded digest → batches.

Nothing is summarized by a model here. A digest is the page's facts (URL, page type,
title, description, reliable date, author, categories, tags, length, heading outline)
plus its main text, condensed extractively when it exceeds the per-item budget: the
opening is kept, then the first paragraph of each section, then the rest in order.
"""

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.domain.content import ContentType, DateSource

GAP = "[…]"
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)\s]*(?:\s+\"[^\"]*\")?\)")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_BLANK_LINES = re.compile(r"\n{3,}")
_DOCUMENT_TAG = re.compile(r"<(/?)\s*document", re.IGNORECASE)
_MAX_OUTLINE_LINES = 30
_MIN_SECTION_SHARE = 80  # characters per section (heading + lead) when spreading the budget


@dataclass(frozen=True)
class DigestSource:
    content_item_id: int
    content_version_id: int
    url: str
    content_type: ContentType
    title: str | None
    description: str | None
    author: str | None
    published_at: datetime | None  # the item's reliable publication date, if any
    published_at_source: DateSource | None
    categories: Sequence[str]
    tags: Sequence[str]
    headings: Sequence[Mapping[str, Any]]
    text: str
    word_count: int


@dataclass(frozen=True)
class Digest:
    content_item_id: int
    content_version_id: int
    url: str
    body: str
    source_chars: int
    truncated: bool

    @property
    def input_hash(self) -> str:
        return hashlib.sha256(self.body.encode()).hexdigest()

    def render(self, ref: str) -> str:
        return f'<document id="{ref}">\n{self.body}\n</document>'

    @property
    def rendered_size(self) -> int:
        return len(self.body) + 32


def neutralize(text: str) -> str:
    """Stop page text from opening or closing the <document> delimiters it is wrapped in."""
    return _DOCUMENT_TAG.sub(r"<\1_document", text)


def clean_markdown(text: str) -> str:
    """Drop images and link targets (kept: link text); normalize blank lines."""
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _TRAILING_SPACE.sub("\n", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


def _is_heading(block: str) -> bool:
    return block.startswith("#")


def _cut(block: str, limit: int) -> str:
    cut = block[:limit].rsplit(" ", 1)[0] if " " in block[:limit] else block[:limit]
    return cut.rstrip() + "…"


def condense(text: str, budget: int) -> tuple[str, bool]:
    """Fit ``text`` into ``budget`` characters. Returns (text, truncated).

    Breadth over depth, so topics from the whole page are represented: the opening
    (up to 40% of the budget), then every section's heading and the start of its first
    paragraph, spread evenly (an evenly spaced subset if there are too many sections),
    then any remaining whole paragraphs in page order.
    """
    text = clean_markdown(text)
    if len(text) <= budget:
        return text, False
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    usable = budget - budget // 10  # headroom for gap markers
    chosen: dict[int, str] = {}
    spent = 0

    def take(index: int, block: str) -> None:
        nonlocal spent
        chosen[index] = block
        spent += len(block) + 2

    # 1. The opening.
    opening = usable * 2 // 5 if len(blocks) > 1 else usable
    for index, block in enumerate(blocks):
        if _is_heading(block) or spent + len(block) + 2 > opening:
            break
        take(index, block)
    if not chosen:
        take(0, _cut(blocks[0], opening))
    # 2. Breadth: each section's heading and the start of its first paragraph.
    sections = [i for i, block in enumerate(blocks) if _is_heading(block) and i not in chosen]
    if sections:
        share = (usable - spent) // len(sections)
        if share < _MIN_SECTION_SHARE:
            count = max((usable - spent) // _MIN_SECTION_SHARE, 0)
            sections = [sections[k * len(sections) // count] for k in range(count)] if count else []  # fmt: skip
            share = _MIN_SECTION_SHARE
        for index in sections:
            heading = blocks[index][:share]
            take(index, heading)
            room = share - len(heading) - 2
            lead = index + 1
            if lead < len(blocks) and not _is_heading(blocks[lead]) and room >= _MIN_SECTION_SHARE // 2:  # fmt: skip
                take(lead, blocks[lead] if len(blocks[lead]) <= room else _cut(blocks[lead], room))
    # 3. Fill with whole paragraphs, in page order.
    for index, block in enumerate(blocks):
        if index not in chosen and spent + len(block) + 2 <= usable:
            take(index, block)
    parts: list[str] = []
    previous = -1
    for index in sorted(chosen):
        if index != previous + 1:
            parts.append(GAP)
        parts.append(chosen[index])
        previous = index
    if previous != len(blocks) - 1:
        parts.append(GAP)
    return "\n\n".join(parts)[:budget], True


def _outline(headings: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = []
    for heading in headings:
        level, heading_text = heading.get("level"), str(heading.get("text") or "").strip()
        if isinstance(level, int) and 1 <= level <= 3 and heading_text:
            lines.append(f"{'  ' * max(level - 2, 0)}- {heading_text[:120]}")
        if len(lines) >= _MAX_OUTLINE_LINES:
            break
    return lines


def build_digest(source: DigestSource, *, max_chars: int) -> Digest:
    header = [f"URL: {source.url}", f"Page type (from URL and markup): {source.content_type.value}"]
    if source.title:
        header.append(f"Title: {source.title.strip()[:300]}")
    if source.description:
        header.append(f"Description: {source.description.strip()[:500]}")
    if source.published_at and source.published_at_source:
        header.append(f"Published: {source.published_at:%Y-%m-%d}")
    if source.author:
        header.append(f"Author: {source.author.strip()[:120]}")
    if source.categories:
        header.append(f"Categories: {', '.join(source.categories[:10])}")
    if source.tags:
        header.append(f"Tags: {', '.join(source.tags[:15])}")
    header.append(f"Length: {source.word_count} words")
    outline = _outline(source.headings)
    if outline:
        header.append("Outline:")
        header.extend(outline)
    head = neutralize("\n".join(header))
    if len(head) > max_chars // 3:  # never let metadata crowd out the text
        head = head[: max_chars // 3].rsplit("\n", 1)[0]
    excerpt_budget = max(max_chars - len(head) - len("\nText:\n"), 0)
    excerpt, truncated = condense(neutralize(source.text), excerpt_budget)
    return Digest(
        content_item_id=source.content_item_id,
        content_version_id=source.content_version_id,
        url=source.url,
        body=f"{head}\nText:\n{excerpt}",
        source_chars=len(source.text),
        truncated=truncated,
    )


def batch_digests(digests: Sequence[Digest], *, max_items: int, max_chars: int) -> list[list[Digest]]:  # fmt: skip
    """Pack digests in order into batches of at most ``max_items`` and ``max_chars``."""
    batches: list[list[Digest]] = []
    current: list[Digest] = []
    size = 0
    for digest in digests:
        if current and (len(current) >= max_items or size + digest.rendered_size > max_chars):
            batches.append(current)
            current, size = [], 0
        current.append(digest)
        size += digest.rendered_size
    if current:
        batches.append(current)
    return batches


def estimate_tokens(chars: int) -> int:
    """Rough token estimate for budgeting before a call (≈4 characters per token)."""
    return math.ceil(chars / 4)
