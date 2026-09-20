"""Deterministic matching between labels: topics ↔ your company profile, audiences, and
near-duplicate topics. No LLM.

Labels are reduced to *stems*: ``label_key`` words, minus filler words, truncated to five
characters. So "automation" and "automating" match, "workflows" and "workflow" match, and
"marketers" matches "marketing teams". Matches are then measured as containment (one
label's stems are a subset of the other's) or Jaccard overlap.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.domain.company import CompanyProfile
from app.services.labels import label_key

_STEM_LENGTH = 5
_FILLER = frozenset(
    {
        "a", "an", "and", "as", "at", "based", "by", "driven", "for", "from", "in", "into",
        "of", "on", "or", "powered", "the", "to", "using", "via", "vs", "with", "team",
        "teams",
    }
)  # fmt: skip
_GENERIC = frozenset(  # too broad to make a topic relevant through the company description
    {"platf", "softw", "tool", "solut", "compa", "busin", "help", "servi", "produ", "user"}
)

EXACT, CONTAINS, OVERLAPS = 1.0, 0.85, 0.7
ADJACENT_FACTOR = 0.6
SUBTOPIC_SUPPORT = 0.5
DESCRIPTION_SUPPORT = 0.35
KEYWORD_BONUS = 0.2


def stems(label: str) -> frozenset[str]:
    return frozenset(
        word[:_STEM_LENGTH] for word in label_key(label).split() if word not in _FILLER
    )


def match_strength(a: str, b: str) -> float:
    """1.0 same label; 0.85 one contains the other; 0.7 Jaccard ≥ 0.5; else 0."""
    if label_key(a) == label_key(b):
        return EXACT
    sa, sb = stems(a), stems(b)
    if not sa or not sb:
        return 0.0
    if sa == sb:
        return EXACT
    if sa <= sb or sb <= sa:
        return CONTAINS
    if len(sa & sb) / len(sa | sb) >= 0.5:
        return OVERLAPS
    return 0.0


def best_match(labels: Iterable[str], targets: Iterable[str]) -> tuple[float, str | None]:
    best, found = 0.0, None
    targets = list(targets)
    for label in labels:
        for target in targets:
            strength = match_strength(label, target)
            if strength > best:
                best, found = strength, target
    return best, found


def similar(a: Sequence[str], b: Sequence[str], *, threshold: float = 0.75) -> bool:
    """Whether two topics (each given by its name and aliases) are near-duplicates:
    the same stems, or Jaccard ≥ ``threshold`` with at least two stems in common."""
    for x in a:
        for y in b:
            sx, sy = stems(x), stems(y)
            if not sx or not sy:
                continue
            if sx == sy:
                return True
            shared = sx & sy
            if len(shared) >= 2 and len(shared) / len(sx | sy) >= threshold:
                return True
    return False


def audience_matches(company_audience: str, content_audience: str) -> bool:
    ca, cb = stems(company_audience), stems(content_audience)
    return bool(ca and cb) and (ca <= cb or cb <= ca)


@dataclass(frozen=True)
class StrategicFit:
    value: float  # 0-1
    matches: tuple[str, ...]  # human-readable reasons, e.g. "core topic 'AI'"
    excluded_by: str | None = None


def strategic_fit(
    topic_labels: Sequence[str],
    company: CompanyProfile,
    *,
    subtopic_labels: Sequence[str] = (),
    item_terms: Sequence[Sequence[str]] = (),
) -> StrategicFit:
    """How relevant a topic is to your company, from the profile alone (deterministic).

    - core topic: 1.0 same, 0.85 contains, 0.7 overlaps; adjacent topics count 60%
    - a subtopic that is a core topic: 0.5
    - every stem of the topic appears in your description or products: 0.35
    - plus up to 0.2 for the share of the topic's pages whose keywords/themes match a core
      topic (``item_terms``: one list of terms per page)
    - an excluded topic that matches wins: the topic is excluded
    """
    excluded, excluded_by = best_match(topic_labels, company.excluded_topics)
    if excluded >= CONTAINS:
        return StrategicFit(0.0, (), excluded_by)
    reasons: list[str] = []
    core, core_hit = best_match(topic_labels, company.core_topics)
    adjacent, adjacent_hit = best_match(topic_labels, company.adjacent_topics)
    subtopic, subtopic_hit = best_match(subtopic_labels, company.core_topics)
    candidates = [
        (core, f"core topic '{core_hit}'"),
        (adjacent * ADJACENT_FACTOR, f"adjacent topic '{adjacent_hit}'"),
        (
            SUBTOPIC_SUPPORT if subtopic >= CONTAINS else 0.0,
            f"subtopic of core topic '{subtopic_hit}'",
        ),
    ]
    described = stems(" ".join([company.description, *(p.name for p in company.products), *(p.description or "" for p in company.products)]))  # fmt: skip
    topic_stems = set().union(*(stems(t) for t in topic_labels)) - _GENERIC
    if topic_stems and topic_stems <= described:
        candidates.append((DESCRIPTION_SUPPORT, "mentioned in your description/products"))
    base, reason = max(candidates, key=lambda pair: pair[0])
    if base > 0:
        reasons.append(reason)
    supported = sum(
        1 for terms in item_terms if best_match(terms, company.core_topics)[0] >= CONTAINS
    )
    bonus = KEYWORD_BONUS * supported / len(item_terms) if item_terms else 0.0
    if bonus > 0:
        reasons.append(f"{supported} of {len(item_terms)} competitor pages on it touch your core topics")  # fmt: skip
    return StrategicFit(round(min(1.0, base + bonus), 4), tuple(reasons))
