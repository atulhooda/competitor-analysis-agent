"""The quality decision (Phase 6): the combined score, the mandatory gates, the issues to fix
(in priority order) and best-version selection. Deterministic.

Combined score = sum(weight * component value), with the weights (QUALITY_WEIGHTS) rescaled to
total 100. The Gemini judge contributes one component (its rubric mean mapped to 0-1); every
other component is measured in code. An article is ``ready`` only if every gate passes.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

from app.config import Settings
from app.domain.opportunities import ScoreComponent
from app.domain.quality import (
    ClaimVerdict,
    FactCheckReport,
    Gate,
    JudgeReport,
    OriginalityReport,
    QualityAssessment,
    QualityIssue,
    QualityMetrics,
    SEOReport,
)

DECISION_VERSION = "quality-decision/1"
MAX_ISSUES = 30
_DETAIL = {
    "fact_support": "claims supported by their cited sources",
    "citation_coverage": "factual claims that cite a source",
    "originality": "distance from stored competitor and company pages",
    "structure": "structure checks passed",
    "readability": "Flesch reading ease, 20 → 0 and 60 → 1",
    "seo": "SEO checks passed",
    "gemini_judgment": "the Gemini rubric mean, 1 → 0 and 5 → 1",
}
_CONTENT_SEO_CHECKS = {"keyword_in_introduction", "keyword_in_h2", "no_duplicate_headings", "heading_hierarchy", "no_keyword_stuffing", "h2_count"}  # fmt: skip


@dataclass(frozen=True)
class QualityPolicy:
    weights: dict[str, float]
    min_score: float
    max_contradicted: int
    max_unsupported_ratio: float
    max_uncited: int

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            weights=dict(settings.quality_weights),
            min_score=settings.quality_min_score,
            max_contradicted=settings.quality_max_contradicted,
            max_unsupported_ratio=settings.quality_max_unsupported_ratio,
            max_uncited=settings.quality_max_uncited_claims,
        )

    def fingerprint_data(self) -> dict[str, Any]:
        return {"version": DECISION_VERSION, "weights": sorted(self.weights.items()), "min": self.min_score, "contradicted": self.max_contradicted, "unsupported": self.max_unsupported_ratio, "uncited": self.max_uncited}  # fmt: skip


def assess(
    metrics: QualityMetrics,
    judge: JudgeReport,
    fact_check: FactCheckReport,
    originality: OriginalityReport,
    seo: SEOReport,
    policy: QualityPolicy,
    *,
    max_overlap: float,
) -> QualityAssessment:
    values = {**metrics.values, "gemini_judgment": judge.value}
    total = sum(policy.weights.values())
    breakdown = []
    for name, weight in policy.weights.items():
        if weight <= 0:
            continue
        maximum = round(weight * 100 / total, 2)
        value = round(values.get(name, 0.0), 4)
        breakdown.append(ScoreComponent(dimension=name, points=round(maximum * value, 2), max_points=maximum, value=value, detail=_DETAIL.get(name, name)))  # fmt: skip
    overall = round(sum(c.points for c in breakdown), 1)
    fc = fact_check.metrics
    gates = [
        Gate(
            name="content_valid",
            passed=not metrics.structural_problems,
            detail="; ".join(metrics.structural_problems) or "well-formed",
        ),
        Gate(
            name="citation_integrity",
            passed=fc.integrity_ok,
            detail="; ".join(fc.integrity_problems) or "every citation resolves to a stored source",
        ),
        Gate(
            name="no_contradicted_claims",
            passed=fc.contradicted <= policy.max_contradicted,
            detail=f"{fc.contradicted} contradicted (max {policy.max_contradicted})",
        ),
        Gate(
            name="unsupported_claims",
            passed=fc.unsupported_claim_ratio <= policy.max_unsupported_ratio,
            detail=f"{fc.unsupported_claim_ratio:.0%} of cited claims unsupported (max {policy.max_unsupported_ratio:.0%})",
        ),
        Gate(
            name="uncited_claims",
            passed=fc.uncited_factual <= policy.max_uncited,
            detail=f"{fc.uncited_factual} factual claim(s) without a citation (max {policy.max_uncited})",
        ),
        Gate(
            name="originality",
            passed=not originality.severe,
            detail=f"highest passage similarity {originality.max_similarity:.0%} (severe at {max_overlap:.0%})",
        ),
        Gate(
            name="seo_fields",
            passed=not seo.mandatory_missing,
            detail=("missing: " + ", ".join(seo.mandatory_missing))
            if seo.mandatory_missing
            else "primary keyword, meta title, meta description and slug present",
        ),
        Gate(
            name="minimum_score",
            passed=overall >= policy.min_score,
            detail=f"{overall} (min {policy.min_score})",
        ),
    ]
    return QualityAssessment(
        overall_score=overall,
        breakdown=breakdown,
        gates=gates,
        passed=all(g.passed for g in gates),
        issues=issues(metrics, judge, fact_check, originality, seo),
    )


def issues(metrics: QualityMetrics, judge: JudgeReport, fact_check: FactCheckReport, originality: OriginalityReport, seo: SEOReport) -> list[QualityIssue]:  # fmt: skip
    """What a revision should fix, most serious first (the order the revision follows)."""
    found: list[tuple[int, dict[str, Any]]] = []
    # Claims: contradicted, then unsupported, then partially supported (less serious, same
    # priority), then citations that don't support an otherwise supported sentence.
    order = {ClaimVerdict.CONTRADICTED: 0, ClaimVerdict.UNSUPPORTED: 1, ClaimVerdict.PARTIAL: 2}
    for claim in sorted((c for c in fact_check.claims if c.verdict in order), key=lambda c: order[c.verdict]):  # fmt: skip
        worst = next((c for c in claim.checks if c.verdict is claim.verdict), None)
        if claim.verdict is ClaimVerdict.CONTRADICTED:
            found.append((1, {"kind": "contradicted_claim", "detail": f"contradicted by its source: {worst.explanation if worst else ''}", "excerpt": claim.claim, "section": claim.section, "source_label": worst.label if worst else None, "evidence": worst.evidence if worst else None}))  # fmt: skip
        elif claim.verdict is ClaimVerdict.UNSUPPORTED:
            found.append((2, {"kind": "unsupported_claim", "detail": f"its cited source doesn't support it: {worst.explanation if worst else ''}", "excerpt": claim.claim, "section": claim.section, "source_label": ", ".join(claim.labels)}))  # fmt: skip
        else:
            found.append((2, {"kind": "partially_supported_claim", "detail": f"goes further than its source: {worst.explanation if worst else ''}", "excerpt": claim.claim, "section": claim.section, "source_label": worst.label if worst else None, "evidence": worst.evidence if worst else None}))  # fmt: skip
    for claim in fact_check.claims:
        if claim.verdict in (ClaimVerdict.SUPPORTED, ClaimVerdict.PARTIAL):
            for check in claim.checks:
                if check.verdict in (ClaimVerdict.UNSUPPORTED, ClaimVerdict.CONTRADICTED):
                    found.append((4, {"kind": "citation_problem", "detail": f"[{check.label}] doesn't support this sentence ({check.verdict.value}): remove that citation", "excerpt": claim.claim, "section": claim.section, "source_label": check.label}))  # fmt: skip
    for uncited in fact_check.uncited:
        if uncited.verdict is ClaimVerdict.NEEDS_VERIFICATION:
            found.append((3, {"kind": "uncited_claim", "detail": f"a {uncited.claim_type} with no citation: {uncited.reason}", "excerpt": uncited.sentence, "section": uncited.section}))  # fmt: skip
    for problem in fact_check.metrics.integrity_problems:
        found.append((4, {"kind": "citation_integrity", "detail": problem}))
    for flag in originality.flagged:
        found.append((5, {"kind": "originality_overlap", "detail": f"{flag.similarity:.0%} similar to {flag.source_label} ({flag.url}); overlap: {flag.overlap_text[:200]}", "excerpt": flag.passage[:300], "section": flag.section}))  # fmt: skip
    for problem in metrics.structural_problems:
        found.append((6, {"kind": "structure", "detail": problem}))
    for problem in seo.package.headings.issues:
        found.append((6, {"kind": "structure", "detail": problem}))
    judged = {d.dimension: d for d in judge.dimensions}
    for dimension in ("search_intent_alignment", "strategic_alignment", "audience_value"):
        d = judged.get(dimension)
        if d is not None and d.score <= 3:
            found += [
                (7, {"kind": dimension, "detail": issue}) for issue in d.issues or [d.explanation]
            ]
    for seo_check in seo.checks:
        if not seo_check.passed and seo_check.name in _CONTENT_SEO_CHECKS:
            found.append((8, {"kind": f"seo_{seo_check.name}", "detail": f"{seo_check.name.replace('_', ' ')}: {seo_check.detail}"}))  # fmt: skip
    for dimension in ("clarity", "readability", "structure", "originality", "factual_support"):
        d = judged.get(dimension)
        if d is not None and d.score <= 3:
            priority = 9 if dimension in ("clarity", "readability") else 6 if dimension == "structure" else 5 if dimension == "originality" else 2  # fmt: skip
            found += [(priority, {"kind": f"judge_{dimension}", "detail": issue}) for issue in d.issues or [d.explanation]]  # fmt: skip
    if metrics.readability.get("flesch_reading_ease", 100.0) < 30:
        found.append((9, {"kind": "readability", "detail": f"Flesch reading ease {metrics.readability['flesch_reading_ease']}: shorten sentences and prefer simpler words"}))  # fmt: skip
    ordered = sorted(enumerate(found), key=lambda pair: (pair[1][0], pair[0]))[:MAX_ISSUES]
    return [QualityIssue(id=f"I{n}", priority=priority, **fields) for n, (_, (priority, fields)) in enumerate(ordered, start=1)]  # fmt: skip


@dataclass(frozen=True)
class Candidate:
    index: int  # order of validation: the first is the Phase 5 edit
    version_id: int
    assessment: QualityAssessment


def choose_best(candidates: Sequence[Candidate]) -> Candidate:
    """Passing versions first, then the highest score, then the earliest (the least rewritten)."""
    return max(
        candidates, key=lambda c: (c.assessment.passed, c.assessment.overall_score, -c.index)
    )
