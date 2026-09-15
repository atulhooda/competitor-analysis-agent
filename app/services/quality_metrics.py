"""Deterministic quality metrics (Phase 6). No LLM computes any number here.

Readability uses the Flesch reading-ease formula (Flesch, 1948):

    206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)

and the Flesch-Kincaid grade level: 0.39 * (words / sentences) + 11.8 * (syllables / words)
- 15.59. Syllables are counted with a vowel-group heuristic (a run of a, e, i, o, u, y is one
syllable; a final silent "e" doesn't count; every word has at least one), which is standard
for English text and needs no dictionary. Readability's 0-1 score maps reading ease 20 → 0
and 60 → 1 (linear): professional B2B prose typically scores 30-60.

The component values (0-1) feed the combined score:

- fact_support: (supported + half of partial) / cited claims (0.5 when there are no claims)
- citation_coverage: cited claims / (cited + uncited claims that need a source)
- originality: the originality report's score
- structure: share of the structure checks passed
- readability: the reading-ease mapping above
- seo: share of the SEO checks passed
"""

import re
import statistics

from app.domain.articles import ArticleContent, BlockType, SectionKind
from app.domain.quality import FactCheckReport, OriginalityReport, QualityMetrics, SEOReport
from app.services.article_content import (  # fmt: skip
    citations,
    split_sentences,
    strip_markers,
    structural_problems,
    text_blocks,
)
from app.services.seo import contains_keyword

METRICS_VERSION = "quality-metrics/1"
READABILITY_FLOOR, READABILITY_TARGET = 20.0, 60.0
LONG_PARAGRAPH_WORDS = 150
_WORD = re.compile(r"[A-Za-z0-9']+")
_VOWELS = re.compile(r"[aeiouy]+")


def syllables(word: str) -> int:
    word = word.lower().strip("'")
    if not word.isalpha():
        return 1
    count = len(_VOWELS.findall(word))
    if word.endswith("e") and not word.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(count, 1)


def readability(content: ArticleContent) -> dict[str, float]:
    words: list[str] = []
    sentences = 0
    for s, b, _, text in text_blocks(content):
        if content.sections[s].blocks[b].type is BlockType.SUBHEADING:
            continue
        plain = strip_markers(text)
        found = _WORD.findall(plain)
        if not found:
            continue
        words += found
        sentences += max(len([x for x in split_sentences(plain) if x.strip()]), 1)
    if not words or not sentences:
        return {"flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0, "words_per_sentence": 0.0, "syllables_per_word": 0.0}  # fmt: skip
    per_sentence = len(words) / sentences
    per_word = sum(syllables(w) for w in words) / len(words)
    return {
        "flesch_reading_ease": round(206.835 - 1.015 * per_sentence - 84.6 * per_word, 1),
        "flesch_kincaid_grade": round(0.39 * per_sentence + 11.8 * per_word - 15.59, 1),
        "words_per_sentence": round(per_sentence, 1),
        "syllables_per_word": round(per_word, 2),
    }


def compute(
    content: ArticleContent,
    fact_check: FactCheckReport,
    originality: OriginalityReport,
    seo: SEOReport,
    *,
    min_words: int,
    labels: set[str],
) -> QualityMetrics:
    headings = seo.package.headings
    paragraphs = [b for s in content.sections for b in s.blocks if b.type is BlockType.PARAGRAPH]  # fmt: skip
    paragraph_words = [len(_WORD.findall(strip_markers(b.text or ""))) for b in paragraphs]
    words = sum(len(_WORD.findall(strip_markers(t))) for *_, t in text_blocks(content))
    sentences = sum(len([x for x in split_sentences(strip_markers(t)) if x.strip()]) for *_, t in text_blocks(content))  # fmt: skip
    problems = structural_problems(content, min_words=min_words, labels=labels)
    structure = {
        "has_title": bool(content.title.strip()),
        "h1_count": headings.h1_count,
        "h2_count": len(headings.h2),
        "h3_count": len(headings.h3),
        "sections": len(content.sections),
        "has_introduction": any(s.kind is SectionKind.INTRODUCTION for s in content.sections),
        "has_conclusion": any(s.kind is SectionKind.CONCLUSION for s in content.sections),
        "hierarchy_ok": headings.hierarchy_ok,
        "duplicate_headings": len(headings.duplicates),
        "longest_paragraph_words": max(paragraph_words, default=0),
    }
    structure_checks = [
        structure["has_title"],
        structure["h1_count"] == 1,
        structure["h2_count"] >= 2,
        structure["hierarchy_ok"],
        structure["duplicate_headings"] == 0,
        structure["has_introduction"],
        structure["has_conclusion"],
        not problems,
        structure["longest_paragraph_words"] <= LONG_PARAGRAPH_WORDS,
    ]
    read = readability(content)
    marked = citations(content)
    fc = fact_check.metrics
    if fc.cited_claims:
        fact_support = (fc.supported + 0.5 * fc.partial) / fc.cited_claims
    else:
        fact_support = 0.0 if fc.factual_claims else 0.5
    seo_passed = {c.name: c.passed for c in seo.checks}
    keywords = [seo.package.primary_keyword, *seo.package.secondary_keywords]
    covered_h2 = sum(1 for h in headings.h2 if any(k and contains_keyword(h, k) for k in keywords))
    values = {
        "fact_support": round(fact_support, 4),
        "citation_coverage": fc.citation_coverage,
        "originality": originality.score,
        "structure": round(sum(structure_checks) / len(structure_checks), 4),
        "readability": round(
            min(
                max(
                    (read["flesch_reading_ease"] - READABILITY_FLOOR)
                    / (READABILITY_TARGET - READABILITY_FLOOR),
                    0.0,
                ),
                1.0,
            ),
            4,
        ),
        "seo": seo.score,
    }
    return QualityMetrics(
        structure=structure,
        length={
            "words": words,
            "paragraphs": len(paragraphs),
            "sentences": sentences,
            "lists": sum(1 for s in content.sections for b in s.blocks if b.type is BlockType.LIST),
            "average_paragraph_words": round(statistics.fmean(paragraph_words), 1)
            if paragraph_words
            else 0.0,
        },
        readability=read,
        citations={
            "citations": sum(len(c.labels) for c in marked),
            "cited_sentences": len(marked),
            "density_per_100_words": round(100 * sum(len(c.labels) for c in marked) / words, 2)
            if words
            else 0.0,
            "citation_coverage": fc.citation_coverage,
            "uncited_factual_claims": fc.uncited_factual,
            "integrity_ok": fc.integrity_ok,
        },
        claims={
            "cited_claims": fc.cited_claims,
            "supported": fc.supported,
            "partial": fc.partial,
            "unsupported": fc.unsupported,
            "contradicted": fc.contradicted,
            "supported_ratio": fc.supported_claim_ratio,
            "unsupported_ratio": fc.unsupported_claim_ratio,
            "contradicted_ratio": fc.contradicted_claim_ratio,
            "uncited_factual_ratio": fc.uncited_factual_claim_ratio,
        },
        originality={
            "max_similarity": originality.max_similarity,
            "avg_similarity": originality.avg_similarity,
            "overall_overlap": originality.overall_overlap,
            "flagged_passages": len(originality.flagged),
            "severe": originality.severe,
        },
        seo={
            "meta_title_length": len(seo.package.meta_title),
            "meta_description_length": len(seo.package.meta_description),
            "keyword_in_title": seo_passed.get("keyword_in_meta_title", False),
            "keyword_in_h1": seo_passed.get("keyword_in_h1", False),
            "keyword_in_introduction": seo_passed.get("keyword_in_introduction", False),
            "heading_keyword_coverage": round(covered_h2 / len(headings.h2), 4)
            if headings.h2
            else 0.0,
            "keyword_density": seo.keyword_density,
            "checks_passed": sum(c.passed for c in seo.checks),
            "checks_total": len(seo.checks),
        },
        structural_problems=problems,
        values=values,
    )


def judge_view(metrics: QualityMetrics) -> dict[str, object]:
    """The metrics the judge sees (no SEO, so an SEO change doesn't re-run the judge)."""
    return {"structure": metrics.structure, "length": metrics.length, "readability": metrics.readability, "citations": metrics.citations, "claims": metrics.claims, "originality": metrics.originality}  # fmt: skip
