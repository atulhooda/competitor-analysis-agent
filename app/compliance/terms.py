"""The Group A term list: prescription drug names (brand and generic) that may not appear
anywhere in a post, each with the class wording to use where safety needs it.

The file is plain text, one term per line, shared with the website's own check so both
repositories read the same list:

    # comment
    isotretinoin => a prescription oral acne medicine
    Botox => (no replacement: describe the treatment, not the product)
    steroid

A term starting ``re:`` is a regular expression (``re:lipo\\w*``). A term matches as a whole word or phrase, in any case, inside words joined by hyphens or
underscores too (``isotretinoin-guide``, ``botox_1.jpg``): slugs, file names and URLs are
checked like prose.
"""

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from app.core.errors import ConfigurationError

GENERIC_CLASS = "a prescription medicine your dermatologist may advise"


@dataclass(frozen=True)
class Term:
    term: str
    replacement: str | None  # the class wording; None: leave it out

    @property
    def pattern(self) -> re.Pattern[str]:
        if self.term.startswith("re:"):  # a regular expression, as the website's list has (lipo\w*)
            return re.compile(r"(?<![a-z0-9])(?:" + self.term[3:] + r")(?![a-z0-9])", re.IGNORECASE)
        words = [re.escape(w) for w in self.term.split()]
        return re.compile(
            r"(?<![a-z0-9])" + r"[\s_\-]+".join(words) + r"(?![a-z0-9])", re.IGNORECASE
        )

    @property
    def suggestion(self) -> str:
        if self.replacement:
            return f'say "{self.replacement}" instead'
        return "leave the product name out: describe the treatment, not the drug"


def parse_terms(text: str) -> list[Term]:
    terms: list[Term] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        term, _, replacement = (part.strip() for part in line.partition("=>"))
        if not term or term.casefold() in seen:
            continue
        seen.add(term.casefold())
        clean = replacement if replacement and not replacement.startswith("(") else None
        terms.append(Term(term, clean))
    return terms


@cache
def load_terms(path: Path) -> tuple[Term, ...]:
    if not path.is_file():
        raise ConfigurationError(f"No Group A term list at {path} (COMPLIANCE_TERMS_FILE)")
    terms = parse_terms(path.read_text(encoding="utf-8"))
    if not terms:
        raise ConfigurationError(f"The Group A term list {path} is empty")
    return tuple(terms)


__all__ = ["GENERIC_CLASS", "Term", "load_terms", "parse_terms"]
