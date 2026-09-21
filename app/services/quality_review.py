"""The Gemini parts of Phase 6 that write or judge: the quality judge and the revision.

The judge scores the rubric (1-5 per dimension) with reasons; its 0-1 value is computed here.
The revision rewrites a version to fix listed issues; the result goes through the same
deterministic checks as a draft and is validated again like any other version.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

from app.config import Settings
from app.domain.analysis import LLMPurpose
from app.domain.articles import (  # fmt: skip
    ArticleBrief,
    ArticleContent,
    ArticleOutline,
    ContentIssue,
    ResearchResult,
)
from app.domain.quality import (
    ClaimVerdict,
    FactCheckReport,
    JudgeDimension,
    JudgeReport,
    OriginalityReport,
    QualityIssue,
)
from app.llm import LLMRequest, ReasoningEffort
from app.prompts import article_draft, quality_judge, revision
from app.prompts.article_common import (  # fmt: skip
    DRAFT_TAG,
    brief_block,
    company_block,
    fence,
    outline_block,
    research_block,
)
from app.services.article_content import (
    clean_citations,
    from_output,
    number_issues,
    structural_problems,
    to_markdown,
    word_count,
)
from app.services.article_writing import ContentRejectedError, WritingConfig, allowed_numbers
from app.services.llm_usage import BudgetedLLM

FINDINGS_TAG = "review_findings"


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    reasoning_effort: ReasoningEffort
    max_context_chars: int

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(model=settings.quality_model, reasoning_effort=settings.quality_reasoning_effort, max_context_chars=settings.article_max_context_chars)  # fmt: skip

    def fingerprint_data(self) -> dict[str, Any]:
        return {"model": self.model, "reasoning": self.reasoning_effort, "context": self.max_context_chars}  # fmt: skip


def fact_check_summary(report: FactCheckReport) -> str:
    m = report.metrics
    lines = [
        f"cited claims {m.cited_claims}: supported {m.supported}, partial {m.partial}, unsupported {m.unsupported}, contradicted {m.contradicted}",
        f"citation coverage {m.citation_coverage:.0%}; uncited claims needing a source: {m.uncited_factual}; citation integrity: {'ok' if m.integrity_ok else '; '.join(m.integrity_problems)}",
    ]
    for claim in [c for c in report.claims if c.verdict is not ClaimVerdict.SUPPORTED][:15]:
        lines.append(f"- {claim.verdict.value} [{', '.join(claim.labels)}]: {claim.claim[:200]}")
    for uncited in [u for u in report.uncited if u.verdict is ClaimVerdict.NEEDS_VERIFICATION][:10]:
        lines.append(f"- uncited ({uncited.claim_type}): {uncited.sentence[:200]}")
    return "\n".join(lines)


def originality_summary(report: OriginalityReport) -> str:
    lines = [f"highest passage similarity {report.max_similarity:.0%}, average {report.avg_similarity:.0%}, overall overlap {report.overall_overlap:.0%}, {len(report.flagged)} flagged passage(s) against {report.documents} stored pages"]  # fmt: skip
    lines += [f"- {f.similarity:.0%} like {f.source_label} ({f.url})" for f in report.flagged[:5]]
    return "\n".join(lines)


MODEL_OUTPUT_LIMIT = 65_536  # Gemini 3.x flash; the request is refused above it


def revision_output_tokens(target_words: int, words: int) -> int:
    """Room for a revision: it returns the whole article again, so the budget comes from the
    article in hand rather than the target length, with headroom for the JSON envelope, the
    list of changes and the model's own reasoning. Too little and the answer is cut off
    mid-article and thrown away (three retries, three wasted calls, the article stuck)."""
    draft_room = article_draft.max_output_tokens(max(target_words, words))
    return min(draft_room + 2_000 + max(target_words, words), MODEL_OUTPUT_LIMIT)


async def judge(llm: BudgetedLLM, *, brief: ArticleBrief, research: ResearchResult, content: ArticleContent, fact_check: FactCheckReport, originality: OriginalityReport, metrics: dict[str, Any], config: JudgeConfig, target_words: int) -> JudgeReport:  # fmt: skip
    request = LLMRequest(
        prompt=quality_judge.render(
            brief=brief_block(brief, target_words),
            research=research_block(research, config.max_context_chars // 2),
            article=fence(DRAFT_TAG, to_markdown(content)),
            fact_check=fact_check_summary(fact_check),
            originality=originality_summary(originality),
            metrics=json.dumps(metrics, sort_keys=True, default=str),
        ),
        system=quality_judge.SYSTEM,
        model=config.model,
        max_output_tokens=quality_judge.MAX_OUTPUT_TOKENS,
        reasoning_effort=config.reasoning_effort,
    )
    response = await llm.structured(request, quality_judge.JudgeOut, purpose=LLMPurpose.QUALITY_JUDGE, prompt_version=quality_judge.VERSION)  # fmt: skip
    return judge_report(response.data)


def judge_report(out: quality_judge.JudgeOut) -> JudgeReport:
    """The rubric scores, with a missing dimension counted as the worst score (1)."""
    given = {}
    for d in out.dimensions:
        key = d.dimension.strip().lower().replace(" ", "_")
        if key in quality_judge.DIMENSIONS and key not in given:
            given[key] = JudgeDimension(dimension=key, score=d.score, explanation=d.explanation, issues=d.issues)  # fmt: skip
    missing = [name for name in quality_judge.DIMENSIONS if name not in given]
    dimensions = [given.get(name) or JudgeDimension(dimension=name, score=1, explanation="not scored by the judge", issues=[]) for name in quality_judge.DIMENSIONS]  # fmt: skip
    value = sum((d.score - 1) / 4 for d in dimensions) / len(dimensions)
    return JudgeReport(dimensions=dimensions, summary=out.summary, value=round(value, 4), missing=missing)  # fmt: skip


@dataclass(frozen=True)
class Revised:
    content: ArticleContent
    changes: list[str]
    issues_addressed: list[str]
    issues: list[ContentIssue]
    model: str


def findings_block(issues: Sequence[QualityIssue], note: str | None) -> str:
    lines = []
    for issue in issues:
        where = f" (section {issue.section + 1})" if issue.section is not None else ""
        source = f" [{issue.source_label}]" if issue.source_label else ""
        lines.append(f"{issue.id} | priority {issue.priority} | {issue.kind}{where}{source}: {issue.detail}")  # fmt: skip
        if issue.excerpt:
            lines.append(f'   passage: "{issue.excerpt[:300]}"')
        if issue.evidence:
            lines.append(f'   source evidence: "{issue.evidence[:300]}"')
    if note:
        lines.append(f"Editor's request: {note}")
    return fence(FINDINGS_TAG, "\n".join(lines) or "(none)")


async def revise(llm: BudgetedLLM, *, brief: ArticleBrief, research: ResearchResult, outline: ArticleOutline | None, content: ArticleContent, issues: Sequence[QualityIssue], note: str | None, config: WritingConfig) -> Revised:  # fmt: skip
    request = LLMRequest(
        prompt=revision.render(
            brief=brief_block(brief, config.target_words),
            company=company_block(brief),
            outline=outline_block(outline) if outline else "Outline: (not stored)",
            research=research_block(research, config.max_context_chars),
            issues=findings_block(issues, note),
            article=fence(
                DRAFT_TAG, json.dumps(content.model_dump(mode="json"), ensure_ascii=False, indent=1)
            ),
            min_words=config.min_words,
            words=word_count(content),
        ),
        system=revision.SYSTEM,
        model=config.model,
        max_output_tokens=revision_output_tokens(config.target_words, word_count(content)),
        reasoning_effort=config.reasoning_effort,
    )
    response = await llm.structured(request, revision.RevisionOut, purpose=LLMPurpose.ARTICLE_REVISION, prompt_version=revision.VERSION)  # fmt: skip
    labels = {s.label for s in research.sources}
    revised, found = clean_citations(from_output(response.data.article), labels)
    problems = structural_problems(revised, min_words=config.min_words, labels=labels)
    if problems:
        raise ContentRejectedError("the revision isn't a usable article: " + "; ".join(problems))
    found += number_issues(revised, allowed_numbers(brief, research))
    known = {i.id for i in issues}
    addressed = [i for i in dict.fromkeys(x.strip().upper() for x in response.data.issues_addressed) if i in known]  # fmt: skip
    return Revised(revised, response.data.changes, addressed, found, response.raw.model)
