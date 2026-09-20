"""Deterministic label normalization for topics, audiences and themes.

``label_key`` maps spellings of the same label to one matching key:
"AI Agents", "AI-agents", "ai agent" and "Artificial intelligence agents" all become
"ai agent". It is conservative on purpose: it unifies case, punctuation, a few standard
abbreviations and a trailing plural, but never guesses at synonyms ("agentic AI" vs
"AI agents"). Those are unified by the taxonomy prompt (reuse existing names), by
aliases, and by reviewed merges.
"""

import re
import unicodedata

# Longest phrases first; applied on word boundaries after punctuation is normalized.
_PHRASES: tuple[tuple[str, str], ...] = tuple(
    sorted(
        {
            "application programming interfaces": "api",
            "application programming interface": "api",
            "customer relationship management": "crm",
            "search engine optimization": "seo",
            "search engine optimisation": "seo",
            "large language models": "llm",
            "large language model": "llm",
            "artificial intelligence": "ai",
            "software as a service": "saas",
            "business to business": "b2b",
            "business to consumer": "b2c",
            "machine learning": "ml",
            "user experience": "ux",
            "user interface": "ui",
            "e commerce": "ecommerce",
            "gen ai": "genai",
        }.items(),
        key=lambda pair: -len(pair[0]),
    )
)
_LEADING_ARTICLES = frozenset({"the", "a", "an"})
# Words that end in "s" but aren't plurals (or whose singular reads wrong).
_INVARIANT = frozenset(
    {
        "aiops", "alias", "analytics", "atlas", "aws", "bias", "canvas", "cms", "devops",
        "ethics", "finops", "gitops", "genai", "https", "ios", "kubernetes", "llmops",
        "logistics", "macos", "mlops", "news", "ops", "paas", "iaas", "saas", "sales",
        "secops", "series", "sms", "species", "windows",
    }
)  # fmt: skip
# Plural acronyms the "-is" rule would otherwise keep (it protects "analysis", "basis").
_ACRONYM_PLURALS = {"apis": "api", "kpis": "kpi", "ais": "ai", "cis": "ci"}
_DASHES_AND_SLASHES = re.compile(r"[\u2010-\u2015_/\\\-]+")  # unicode dashes, _, /, \, -
_PUNCTUATION = re.compile(r"[^\w\s+#.]")
_LOOSE_DOTS = re.compile(r"(?<!\w)\.|\.(?!\w)")
_SLUG_CHARS = re.compile(r"[^a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")


def _singular(word: str) -> str:
    if word in _ACRONYM_PLURALS:
        return _ACRONYM_PLURALS[word]
    if (
        not word.isalpha()  # node.js, c++, b2b
        or word in _INVARIANT
        or len(word) <= 3
        or not word.endswith("s")
        or word.endswith(("ss", "us", "is", "ics"))
    ):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "ches", "shes", "xes", "zes")):
        return word[:-2]
    return word[:-1]


def label_key(label: str) -> str:
    """Matching key for a label ('' if the label has no letters or digits)."""
    text = unicodedata.normalize("NFKC", label).casefold().replace("&", " and ")
    text = _DASHES_AND_SLASHES.sub(" ", text)
    text = _PUNCTUATION.sub(" ", text)
    text = _LOOSE_DOTS.sub(" ", text)
    words = text.split()
    if words and words[0] in _LEADING_ARTICLES and len(words) > 1:
        words = words[1:]
    joined = f" {' '.join(words)} "
    for phrase, replacement in _PHRASES:
        joined = joined.replace(f" {phrase} ", f" {replacement} ")
    words = joined.split()
    if not words:
        return ""
    words[-1] = _singular(words[-1])
    return " ".join(words)


def clean_label(label: str, *, max_length: int = 80) -> str:
    """Display form: single spaces, no surrounding punctuation. Casing is kept as written
    (the analyzer is asked for sentence case; 'AI agents' must not become 'Ai agents')."""
    text = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", label)).strip(" .,;:-–—'\"")  # noqa: RUF001
    return text[:max_length].rstrip()


def slugify(text: str, *, max_length: int = 80) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = _SLUG_CHARS.sub("-", ascii_text.casefold()).strip("-")
    return slug[:max_length].rstrip("-") or "topic"
