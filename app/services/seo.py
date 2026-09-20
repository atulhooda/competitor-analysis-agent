"""The SEO package (Phase 6). Candidates come from stored data; Gemini chooses and words them;
code validates the result and runs the checks.

- Keyword candidates (deterministic, with provenance): the opportunity's topic (weight 4),
  its competitor subtopics (2 each), the keywords of the competitor pages behind it (1, plus
  0.25 for each further page using it, up to +1: frequency matters, but repetition can't
  outweigh the topic), and the company's core and adjacent topics (1), plus how prominently
  the article already uses them (title +2, a heading +1, body mentions up to +2).
- Gemini picks the primary keyword (a candidate, or a rewording of one using only
  candidate and heading words), secondary keywords, meta title and description, a slug,
  FAQ, links, category, tags and an image idea.
- Code rejects anything invented: keywords unrelated to the candidates, link ids that
  weren't offered (so every URL is a stored company page or research source), FAQ answers
  with numbers absent from the article, categories outside the options.
- Deterministic checks (lengths, keyword placement, heading hierarchy, keyword density:
  repetition is flagged as stuffing, never rewarded) give the SEO score.
"""

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

from app.config import Settings
from app.domain.analysis import LLMPurpose
from app.domain.articles import ArticleBrief, ArticleContent, BlockType, SectionKind, SourceType
from app.domain.quality import (
    FAQItem,
    HeadingAnalysis,
    ImageSuggestion,
    KeywordCandidate,
    LinkSuggestion,
    SEOCheck,
    SEOPackage,
    SEOReport,
)
from app.llm import LLMRequest, ReasoningEffort
from app.prompts import seo as prompt
from app.services.article_content import slugify, strip_markers, text_blocks, to_markdown
from app.services.checkpoints import digest
from app.services.labels import label_key
from app.services.llm_usage import BudgetedLLM
from app.services.numbers import numbers_in
from app.services.relevance import stems

MAX_CANDIDATES = 20
MAX_INTERNAL = 10
MAX_LINKS = 5
ALT_TEXT_MAX = 125
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class SEOConfig:
    model: str
    reasoning_effort: ReasoningEffort
    title_max: int
    description_min: int
    description_max: int
    max_density: float

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            model=settings.quality_model,
            reasoning_effort=settings.quality_reasoning_effort,
            title_max=settings.seo_title_max_chars,
            description_min=settings.seo_description_min_chars,
            description_max=settings.seo_description_max_chars,
            max_density=settings.seo_max_keyword_density,
        )

    def fingerprint_data(self) -> dict[str, Any]:
        return {"model": self.model, "reasoning": self.reasoning_effort, "title": self.title_max, "desc": [self.description_min, self.description_max], "density": self.max_density}  # fmt: skip


@dataclass(frozen=True)
class InternalPage:
    url: str
    title: str
    headings: tuple[str, ...]


@dataclass(frozen=True)
class ExternalSource:
    label: str
    url: str
    title: str | None
    source_type: SourceType
    facts: int


@dataclass(frozen=True)
class SEOInputs:
    brief: ArticleBrief
    subtopics: tuple[str, ...]
    competitor_keywords: tuple[str, ...]
    company_topics: tuple[str, ...]
    internal_pages: tuple[InternalPage, ...]
    sources: tuple[ExternalSource, ...]

    def fingerprint(self) -> str:
        return digest(
            {
                "topic": self.brief.topic, "intent": self.brief.search_intent.value, "audience": self.brief.target_audience,
                "subtopics": self.subtopics, "keywords": self.competitor_keywords, "company": self.company_topics,
                "internal": [(p.url, p.title) for p in self.internal_pages],
                "sources": [(s.label, s.url, s.source_type.value) for s in self.sources],
            }
        )  # fmt: skip


def _stems(text: str) -> frozenset[str]:
    return stems(text)


def contains_keyword(text: str, keyword: str) -> bool:
    """Every content word of the keyword appears in the text (order and inflection aside)."""
    wanted = _stems(keyword)
    return bool(wanted) and wanted <= _stems(text)


def heading_analysis(content: ArticleContent) -> HeadingAnalysis:
    h1 = content.title.strip() or None
    h2 = [s.heading.strip() for s in content.sections if s.heading and s.heading.strip()]
    h3: list[str] = []
    issues: list[str] = []
    hierarchy_ok = True
    for index, section in enumerate(content.sections):
        for block in section.blocks:
            if block.type is BlockType.SUBHEADING and block.text:
                h3.append(block.text.strip())
                if not (section.heading or "").strip():
                    hierarchy_ok = False
                    issues.append(f"H3 '{block.text.strip()[:60]}' has no H2 above it (section {index + 1})")  # fmt: skip
    seen: dict[str, str] = {}
    duplicates = []
    for heading in ([h1] if h1 else []) + h2 + h3:
        key = label_key(heading)
        if key in seen and heading not in duplicates:
            duplicates.append(heading)
        seen.setdefault(key, heading)
    if duplicates:
        issues.append(f"duplicate heading(s): {'; '.join(d[:60] for d in duplicates)}")
    for heading in h2 + h3:
        if len(heading) > 90:
            issues.append(f"heading longer than 90 characters: '{heading[:60]}…'")
    if not h1:
        issues.append("the article has no title (H1)")
    return HeadingAnalysis(h1=h1, h1_count=1 if h1 else 0, h2=h2, h3=h3, hierarchy_ok=hierarchy_ok, duplicates=duplicates, issues=issues)  # fmt: skip


def keyword_candidates(inputs: SEOInputs, content: ArticleContent) -> list[KeywordCandidate]:
    weights: dict[str, float] = defaultdict(float)
    sources: dict[str, list[str]] = defaultdict(list)
    forms: dict[str, str] = {}

    def add(keyword: str, weight: float, source: str) -> None:
        keyword = " ".join(keyword.split())
        key = label_key(keyword)
        if not key or len(keyword) > 80:
            return
        forms.setdefault(key, keyword.lower() if not any(c.isupper() for c in keyword[1:]) else keyword)  # fmt: skip
        weights[key] += weight
        if source not in sources[key]:
            sources[key].append(source)

    add(inputs.brief.topic, 4, "opportunity topic")
    for subtopic in dict.fromkeys(inputs.subtopics):
        add(subtopic, 2, "competitor subtopic")
    pages: dict[str, tuple[str, int]] = {}
    for keyword in inputs.competitor_keywords:
        key = label_key(keyword)
        pages[key] = (pages.get(key, (keyword, 0))[0], pages.get(key, (keyword, 0))[1] + 1)
    for keyword, count in pages.values():
        add(keyword, 1 + 0.25 * min(count - 1, 4), "competitor keyword")
    for topic in inputs.company_topics:
        add(topic, 1, "company topic")
    title = content.title
    h2 = " ".join(s.heading or "" for s in content.sections)
    body_tokens = _WORD.findall(" ".join(strip_markers(t) for *_, t in text_blocks(content)).lower())  # fmt: skip
    ranked = []
    for key, weight in weights.items():
        keyword = forms[key]
        bonus: float = (2 if contains_keyword(title, keyword) else 0) + (
            1 if contains_keyword(h2, keyword) else 0
        )
        bonus += min(_occurrences(body_tokens, keyword) * 0.5, 2)
        ranked.append(KeywordCandidate(keyword=keyword, score=round(weight + bonus, 2), sources=sources[key]))  # fmt: skip
    return sorted(ranked, key=lambda c: (-c.score, c.keyword))[:MAX_CANDIDATES]


def _occurrences(tokens: Sequence[str], keyword: str) -> int:
    phrase = _WORD.findall(keyword.lower())
    if not phrase:
        return 0
    n = len(phrase)
    return sum(1 for i in range(len(tokens) - n + 1) if tokens[i : i + n] == phrase)


def keyword_density(content: ArticleContent, keyword: str) -> float:
    tokens = _WORD.findall(" ".join(strip_markers(t) for *_, t in text_blocks(content)).lower())
    return round(_occurrences(tokens, keyword) / len(tokens), 4) if tokens else 0.0


def accept_keyword(keyword: str, candidates: Sequence[KeywordCandidate], content: ArticleContent) -> tuple[str | None, list[str]]:  # fmt: skip
    """The keyword if it's a candidate, or a rewording built only from candidate and heading
    words that overlaps a candidate; with the stored inputs it rests on. None if invented."""
    key = label_key(keyword)
    for candidate in candidates:
        if label_key(candidate.keyword) == key:
            return keyword.strip(), list(candidate.sources)
    wanted = _stems(keyword)
    if not wanted:
        return None, []
    pool: set[str] = set(_stems(content.title))
    for section in content.sections:
        pool |= _stems(section.heading or "")
    related = [c for c in candidates if _stems(c.keyword) & wanted]
    for candidate in candidates:
        pool |= _stems(candidate.keyword)
    if wanted <= pool and related:
        evidence = sorted({s for c in related for s in c.sources})
        return keyword.strip(), [*evidence, "reworded by Gemini from the candidates"]
    return None, []


def internal_candidates(pages: Sequence[InternalPage], inputs: SEOInputs, content: ArticleContent) -> list[InternalPage]:  # fmt: skip
    """Stored pages of your own site that relate to the article, most related first."""
    topic: set[str] = set(_stems(inputs.brief.topic)) | _stems(content.title)
    for section in content.sections:
        topic |= _stems(section.heading or "")
    for keyword in inputs.subtopics:
        topic |= _stems(keyword)
    scored = []
    for page in pages:
        words = _stems(page.title) | {s for h in page.headings for s in _stems(h)}
        overlap = len(words & topic)
        if overlap:
            scored.append((-overlap, page.url, page))
    return [page for *_, page in sorted(scored)][:MAX_INTERNAL]


def external_candidates(sources: Sequence[ExternalSource]) -> list[ExternalSource]:
    """Research sources worth linking: retrieved and authoritative; never competitors."""
    return [s for s in sources if s.source_type not in (SourceType.COMPETITOR, SourceType.COMPANY)]


def category_options(inputs: SEOInputs) -> list[str]:
    return list(dict.fromkeys([inputs.brief.topic, *inputs.company_topics]))


async def build_seo(llm: BudgetedLLM, inputs: SEOInputs, content: ArticleContent, config: SEOConfig) -> SEOReport:  # fmt: skip
    candidates = keyword_candidates(inputs, content)
    internal = internal_candidates(inputs.internal_pages, inputs, content)
    external = external_candidates(inputs.sources)
    categories = category_options(inputs)
    request = LLMRequest(
        prompt=prompt.render(
            topic=inputs.brief.topic,
            audience=inputs.brief.target_audience,
            intent=inputs.brief.search_intent.value,
            angle=inputs.brief.primary_angle,
            company=inputs.brief.company.name,
            article=to_markdown(content),
            keywords=[
                f"K{i} | {c.keyword} | score {c.score} | from: {', '.join(c.sources)}"
                for i, c in enumerate(candidates, start=1)
            ],
            internal=[f"L{i} | {p.title} | {p.url}" for i, p in enumerate(internal, start=1)],
            external=[
                f"X{i} | {s.title or s.url} | {s.source_type.value} | {s.url}"
                for i, s in enumerate(external, start=1)
            ],
            categories=categories,
        ),
        system=prompt.SYSTEM,
        model=config.model,
        max_output_tokens=prompt.MAX_OUTPUT_TOKENS,
        reasoning_effort=config.reasoning_effort,
    )
    response = await llm.structured(request, prompt.SEOOut, purpose=LLMPurpose.SEO_PACKAGE, prompt_version=prompt.VERSION)  # fmt: skip
    return assemble(response.data, candidates, internal, external, categories, content, config)


def assemble(
    out: prompt.SEOOut,
    candidates: Sequence[KeywordCandidate],
    internal: Sequence[InternalPage],
    external: Sequence[ExternalSource],
    categories: Sequence[str],
    content: ArticleContent,
    config: SEOConfig,
) -> SEOReport:
    """Validate the model's package against what it was offered, then check it."""
    notes: list[str] = []
    primary, evidence = accept_keyword(out.primary_keyword, candidates, content)
    reason = out.primary_keyword_reason
    if primary is None:
        fallback = candidates[0] if candidates else None
        notes.append(f"primary keyword '{out.primary_keyword}' isn't supported by the stored inputs; used the top candidate instead")  # fmt: skip
        primary = fallback.keyword if fallback else ""
        evidence = list(fallback.sources) if fallback else []
        reason = "top-scoring candidate (the proposed keyword was rejected)"
    secondary: list[str] = []
    for keyword in out.secondary_keywords:
        accepted, _ = accept_keyword(keyword, candidates, content)
        if accepted is None:
            notes.append(f"secondary keyword '{keyword}' rejected: not among the candidates")
        elif label_key(accepted) != label_key(primary) and label_key(accepted) not in {label_key(k) for k in secondary}:  # fmt: skip
            secondary.append(accepted)
    for candidate in candidates:  # at least three, from the candidates
        if len(secondary) >= 3:
            break
        if label_key(candidate.keyword) not in {label_key(k) for k in [primary, *secondary]}:
            secondary.append(candidate.keyword)
    article_text = " ".join(strip_markers(t) for *_, t in text_blocks(content))
    allowed_numbers = numbers_in(article_text + " " + content.title)
    faq = []
    for item in out.faq:
        if not item.question.strip() or not item.answer.strip():
            continue
        if numbers_in(item.answer) - allowed_numbers:
            notes.append(f"FAQ answer dropped: it has numbers the article doesn't ({item.question[:60]})")  # fmt: skip
            continue
        faq.append(FAQItem(question=item.question.strip(), answer=item.answer.strip()))
    internal_links = _links(out.internal_links, {f"L{i}": (p.url, p.title) for i, p in enumerate(internal, start=1)}, notes)  # fmt: skip
    external_links = _links(out.external_links, {f"X{i}": (s.url, s.title) for i, s in enumerate(external, start=1)}, notes)  # fmt: skip
    category = next((c for c in categories if label_key(c) == label_key(out.category)), None)
    if category is None:
        category = categories[0] if categories else ""
        if out.category:
            notes.append(f"category '{out.category}' isn't an option; used '{category}'")
    vocabulary = {s for c in candidates for s in _stems(c.keyword)} | {s for c in categories for s in _stems(c)}  # fmt: skip
    tags: list[str] = []
    for tag in out.tags:
        if _stems(tag) & vocabulary and label_key(tag) not in {label_key(t) for t in tags}:
            tags.append(tag.strip())
    for keyword in [primary, *secondary]:
        if len(tags) >= 3:
            break
        if keyword and label_key(keyword) not in {label_key(t) for t in tags}:
            tags.append(keyword)
    image = None
    if out.image and out.image.concept.strip() and out.image.alt_text.strip():
        alt = out.image.alt_text.strip()
        if len(alt) > ALT_TEXT_MAX:
            alt = alt[:ALT_TEXT_MAX].rsplit(" ", 1)[0]
            notes.append("image alt text shortened to 125 characters")
        image = ImageSuggestion(concept=out.image.concept.strip(), purpose=out.image.purpose.strip(), alt_text=alt)  # fmt: skip
    slug = slugify(out.slug or primary) if (out.slug or primary) else ""
    if len(slug) > 60:
        slug = slug[:60].rsplit("-", 1)[0]
    package = SEOPackage(
        primary_keyword=primary,
        primary_keyword_evidence=evidence,
        primary_keyword_reason=reason,
        secondary_keywords=secondary[:8],
        meta_title=out.meta_title.strip(),
        meta_description=out.meta_description.strip(),
        slug=slug,
        headings=heading_analysis(content),
        faq=faq[:6],
        internal_links=internal_links,
        external_links=external_links,
        category=category,
        tags=tags[:8],
        image=image,
    )
    checks, density, missing = seo_checks(package, content, config, has_sources=bool(external))
    score = round(sum(c.passed for c in checks) / len(checks), 4) if checks else 0.0
    return SEOReport(package=package, candidates=list(candidates), checks=checks, score=score, keyword_density=density, mandatory_missing=missing, notes=notes)  # fmt: skip


def _links(choices: Sequence[prompt.LinkChoiceOut], offered: dict[str, tuple[str, str | None]], notes: list[str]) -> list[LinkSuggestion]:  # fmt: skip
    links: list[LinkSuggestion] = []
    for choice in choices:
        target = offered.get(choice.candidate.strip().upper())
        if target is None:
            notes.append(f"link {choice.candidate} dropped: it wasn't offered (no invented URLs)")
            continue
        url, title = target
        if url in {link.url for link in links} or not choice.anchor_text.strip():
            continue
        links.append(LinkSuggestion(anchor_text=choice.anchor_text.strip(), url=url, title=title, reason=choice.reason))  # fmt: skip
    return links[:MAX_LINKS]


def seo_checks(package: SEOPackage, content: ArticleContent, config: SEOConfig, *, has_sources: bool) -> tuple[list[SEOCheck], float, list[str]]:  # fmt: skip
    keyword = package.primary_keyword
    intro = " ".join(strip_markers(b.text or " ".join(b.items)) for s in content.sections if s.kind is SectionKind.INTRODUCTION for b in s.blocks)  # fmt: skip
    density = keyword_density(content, keyword) if keyword else 0.0
    h = package.headings
    checks = [
        SEOCheck(name="primary_keyword", passed=bool(keyword), detail=keyword or "missing"),
        SEOCheck(
            name="keyword_in_meta_title",
            passed=bool(keyword) and contains_keyword(package.meta_title, keyword),
            detail=package.meta_title,
        ),
        SEOCheck(
            name="keyword_in_h1",
            passed=bool(keyword) and contains_keyword(h.h1 or "", keyword),
            detail=h.h1 or "no H1",
        ),
        SEOCheck(
            name="keyword_in_introduction",
            passed=bool(keyword) and contains_keyword(intro, keyword),
            detail="introduction",
        ),
        SEOCheck(
            name="keyword_in_h2",
            passed=bool(keyword) and any(contains_keyword(x, keyword) for x in h.h2),
            detail=f"{len(h.h2)} H2 heading(s)",
        ),
        SEOCheck(
            name="keyword_in_slug",
            passed=bool(keyword) and contains_keyword(package.slug.replace("-", " "), keyword),
            detail=package.slug,
        ),
        SEOCheck(
            name="meta_title_length",
            passed=0 < len(package.meta_title) <= config.title_max,
            detail=f"{len(package.meta_title)} of at most {config.title_max} characters",
        ),
        SEOCheck(
            name="meta_description_length",
            passed=config.description_min
            <= len(package.meta_description)
            <= config.description_max,
            detail=f"{len(package.meta_description)} characters ({config.description_min}-{config.description_max})",
        ),
        SEOCheck(
            name="heading_hierarchy",
            passed=h.hierarchy_ok,
            detail="; ".join(i for i in h.issues if "H3" in i) or "ok",
        ),
        SEOCheck(
            name="no_duplicate_headings",
            passed=not h.duplicates,
            detail="; ".join(h.duplicates) or "ok",
        ),
        SEOCheck(name="h2_count", passed=len(h.h2) >= 2, detail=f"{len(h.h2)} H2 heading(s)"),
        SEOCheck(
            name="no_keyword_stuffing",
            passed=density <= config.max_density,
            detail=f"density {density:.2%} (max {config.max_density:.0%})",
        ),
        SEOCheck(
            name="faq", passed=len(package.faq) >= 3, detail=f"{len(package.faq)} question(s)"
        ),
    ]
    if has_sources:
        checks.append(SEOCheck(name="external_links", passed=bool(package.external_links), detail=f"{len(package.external_links)} link(s)"))  # fmt: skip
    missing = [name for name, value in (("primary_keyword", keyword), ("meta_title", package.meta_title), ("meta_description", package.meta_description), ("slug", package.slug)) if not value.strip()]  # fmt: skip
    return checks, density, missing
