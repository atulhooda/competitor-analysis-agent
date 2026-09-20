"""Keep model-written text from introducing numbers the evidence doesn't contain.

Opportunity metrics (counts, growth, frequencies, scores) are computed deterministically.
The model may quote them, but any sentence carrying a number that doesn't appear in the
evidence it was given is removed before the text is stored.
"""

import re

_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_LIST_COUNT = re.compile(r"(?:[1-9]|1\d|20)")  # "7 ways to …"
_YEAR = re.compile(r"(?:19[89]\d|20\d\d)")  # "the 2026 guide to …"
# A percentage or a multiple ("3x", "3\u00d7" with the times sign)
_STATISTIC_SUFFIX = re.compile(r"\s*(?:%|percent\b|x\b|\u00d7)", re.IGNORECASE)


def _normalize(number: str) -> str:
    """ "1,200" → "1200", "84.0" → "84", "0.50" → "0.5", "09" → "9"."""
    whole, _, fraction = number.replace(",", "").partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def numbers_in(text: str) -> set[str]:
    return {_normalize(n) for n in _NUMBER.findall(text)}


def title_is_verified(title: str, allowed: set[str]) -> bool:
    """A title may quote evidence numbers, a small list count ("7 ways to …") or a year.
    Any other number (a percentage, a multiple, a large count) would be an invented statistic.
    """
    for match in _NUMBER.finditer(title):
        number = _normalize(match.group())
        if number in allowed:
            continue
        if _STATISTIC_SUFFIX.match(title, match.end()):
            return False
        if not (_LIST_COUNT.fullmatch(number) or _YEAR.fullmatch(number)):
            return False
    return True


def strip_unverified(text: str, allowed: set[str]) -> tuple[str, int]:
    """``text`` without the sentences that contain numbers outside ``allowed``.
    Returns (text, sentences removed)."""
    kept, removed = [], 0
    for sentence in _SENTENCE.split(text.strip()):
        if numbers_in(sentence) - allowed:
            removed += 1
        else:
            kept.append(sentence)
    return " ".join(kept).strip(), removed
