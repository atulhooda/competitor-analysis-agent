"""Deterministic change detection between two captured versions of a page. No LLM."""

import difflib
import re
from dataclasses import dataclass

_WORD = re.compile(r"\w+")
# Currency amounts as written on pricing pages: $29, $1,299.00, €49, £12.50, ₹999, 29 USD.
_PRICE = re.compile(
    r"(?:[$€£¥₹]\s?\d[\d,]*(?:\.\d{1,2})?)"
    r"|(?:\b\d[\d,]*(?:\.\d{1,2})?\s?(?:USD|EUR|GBP|INR|AUD|CAD|JPY)\b)"
)
_MAX_DIFF_WORDS = 30_000  # beyond this, diff by lines only (word diffing gets slow)
_MINOR_SIMILARITY = 0.9
_MINOR_WORDS = 12


@dataclass(frozen=True)
class TextDiff:
    similarity: float  # 0..1, word-based
    words_added: int
    words_removed: int
    lines_added: int
    lines_removed: int
    title_changed: bool

    @property
    def is_minor(self) -> bool:
        """Small edits (a typo, a date, a counter) that don't change what the page says."""
        return (
            not self.title_changed
            and self.words_added + self.words_removed <= _MINOR_WORDS
            and self.similarity >= _MINOR_SIMILARITY
        )

    def as_details(self) -> dict[str, object]:
        return {
            "similarity": round(self.similarity, 4),
            "words_added": self.words_added,
            "words_removed": self.words_removed,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "title_changed": self.title_changed,
        }


def _changes(old: list[str], new: list[str]) -> tuple[float, list[str], list[str]]:
    """(similarity, added elements, removed elements) between two sequences."""
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    added: list[str] = []
    removed: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed.extend(old[i1:i2])
        if tag in ("replace", "insert"):
            added.extend(new[j1:j2])
    return matcher.ratio(), added, removed


def _count_words(lines: list[str]) -> int:
    return sum(len(_WORD.findall(line)) for line in lines)


def diff_texts(
    old_text: str, new_text: str, *, old_title: str | None = None, new_title: str | None = None
) -> TextDiff:
    """Word-level diff (trafilatura puts each paragraph on one line, so a line-level diff
    would count a one-word typo fix as a whole paragraph changing)."""
    old_lines = [line.strip() for line in old_text.splitlines() if line.strip()]
    new_lines = [line.strip() for line in new_text.splitlines() if line.strip()]
    _, lines_added, lines_removed = _changes(old_lines, new_lines)
    title_changed = (old_title or "").strip() != (new_title or "").strip()
    old_words, new_words = _WORD.findall(old_text), _WORD.findall(new_text)
    if len(old_words) > _MAX_DIFF_WORDS or len(new_words) > _MAX_DIFF_WORDS:
        similarity, added, removed = _changes(old_lines, new_lines)
        words_added, words_removed = _count_words(added), _count_words(removed)
    else:
        similarity, added, removed = _changes(old_words, new_words)
        words_added, words_removed = len(added), len(removed)
    return TextDiff(
        similarity=similarity,
        words_added=words_added,
        words_removed=words_removed,
        lines_added=len(lines_added),
        lines_removed=len(lines_removed),
        title_changed=title_changed,
    )


def extract_prices(text: str) -> list[str]:
    """Distinct currency amounts in the text, normalized (whitespace removed), sorted."""
    return sorted({re.sub(r"\s+", "", match) for match in _PRICE.findall(text)})


@dataclass(frozen=True)
class PriceChange:
    before: list[str]
    after: list[str]

    @property
    def added(self) -> list[str]:
        return sorted(set(self.after) - set(self.before))

    @property
    def removed(self) -> list[str]:
        return sorted(set(self.before) - set(self.after))

    def as_details(self) -> dict[str, object]:
        return {
            "prices_before": self.before,
            "prices_after": self.after,
            "added": self.added,
            "removed": self.removed,
        }


def detect_price_change(old_text: str, new_text: str) -> PriceChange | None:
    before, after = extract_prices(old_text), extract_prices(new_text)
    return PriceChange(before, after) if before != after else None
