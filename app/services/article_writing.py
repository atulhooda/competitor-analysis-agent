"""The writing steps (Phase 5): outline, draft and editorial pass. One Gemini call each,
with no tools: the model sees only the brief, the company profile, the verified research
and (for competitive context) the stored summaries of competitor pages, never their text.

After each call the output goes through deterministic checks (``article_content``):
citation labels that don't match a stored source are removed and recorded, sentences with
numbers found in neither the research nor the brief are flagged, and structurally broken
output is rejected (the step fails, and a resume retries it).
"""

import json
from dataclasses import dataclass
from typing import Any, Self

from app.config import Settings
from app.core.errors import PermanentError
from app.domain.analysis import LLMPurpose
from app.domain.articles import (
    ArticleBrief,
    ArticleContent,
    ArticleOutline,
    ContentIssue,
    OutlineSection,
    ResearchResult,
)
from app.llm import LLMRequest, ReasoningEffort
from app.prompts import article_draft, article_edit, article_outline
from app.prompts.article_common import (
    DRAFT_TAG,
    brief_block,
    company_block,
    competitor_block,
    fence,
    outline_block,
    research_block,
)
from app.services.article_content import (
    clean_citations,
    from_output,
    number_issues,
    structural_problems,
)
from app.services.llm_usage import BudgetedLLM
from app.services.numbers import numbers_in


class ContentRejectedError(PermanentError):
    """Generated output isn't usable as an article (it's structurally broken)."""


@dataclass(frozen=True)
class WritingConfig:
    model: str
    reasoning_effort: ReasoningEffort
    target_words: int
    min_words: int
    max_context_chars: int

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            model=settings.writing_model,
            reasoning_effort=settings.writing_reasoning_effort,
            target_words=settings.article_target_words,
            min_words=settings.article_min_words,
            max_context_chars=settings.article_max_context_chars,
        )

    def fingerprint_data(self, *, final: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "model": self.model,
            "reasoning": self.reasoning_effort,
            "target_words": self.target_words,
            "max_context_chars": self.max_context_chars,
        }
        if final:  # only the edited version must reach the minimum length
            data["min_words"] = self.min_words
        return data


@dataclass(frozen=True)
class Written:
    content: ArticleContent
    issues: list[ContentIssue]
    changes: list[str]
    model: str


def allowed_numbers(brief: ArticleBrief, research: ResearchResult) -> set[str]:
    """Numbers the article may use: those in the brief, the company profile or the research."""
    parts = [brief.model_dump_json()]
    parts += [f"{f.statement} {f.excerpt or ''}" for f in research.facts]
    parts += [f"{s.title or ''} {s.published or ''}" for s in research.sources]
    return numbers_in("\n".join(parts))


def _labels(research: ResearchResult) -> set[str]:
    return {s.label for s in research.sources}


def _section(out: article_outline.OutlineSectionOut, labels: set[str], issues: list[ContentIssue], where: str) -> OutlineSection:  # fmt: skip
    kept = [label for label in dict.fromkeys(out.source_ids) if label in labels]
    for label in dict.fromkeys(out.source_ids):
        if label not in labels:
            issues.append(ContentIssue(kind="unknown_citation_removed", detail=f"outline {where} cited [{label}], which doesn't match any stored source"))  # fmt: skip
    return OutlineSection(heading=out.heading, purpose=out.purpose, key_points=out.key_points, source_ids=kept, audience_value=out.audience_value)  # fmt: skip


async def write_outline(llm: BudgetedLLM, brief: ArticleBrief, research: ResearchResult, config: WritingConfig) -> tuple[ArticleOutline, list[ContentIssue], str]:  # fmt: skip
    request = LLMRequest(
        prompt=article_outline.render(
            brief=brief_block(brief, config.target_words),
            company=company_block(brief),
            research=research_block(research, config.max_context_chars),
            competitors=competitor_block(brief),
        ),
        system=article_outline.SYSTEM,
        model=config.model,
        max_output_tokens=article_outline.MAX_OUTPUT_TOKENS,
        reasoning_effort=config.reasoning_effort,
    )
    response = await llm.structured(request, article_outline.OutlineOut, purpose=LLMPurpose.ARTICLE_OUTLINE, prompt_version=article_outline.VERSION)  # fmt: skip
    out, labels = response.data, _labels(research)
    issues: list[ContentIssue] = []
    outline = ArticleOutline(
        title=out.title,
        description=out.description,
        introduction=_section(out.introduction, labels, issues, "introduction"),
        sections=[
            _section(s, labels, issues, f"section {i}") for i, s in enumerate(out.sections, start=1)
        ],
        conclusion=_section(out.conclusion, labels, issues, "conclusion"),
    )
    if not outline.title.strip() or not outline.sections:
        raise ContentRejectedError("the outline has no title or no body sections")
    return outline, issues, response.raw.model


async def write_draft(llm: BudgetedLLM, brief: ArticleBrief, research: ResearchResult, outline: ArticleOutline, config: WritingConfig) -> Written:  # fmt: skip
    request = LLMRequest(
        prompt=article_draft.render(
            brief=brief_block(brief, config.target_words),
            company=company_block(brief),
            outline=outline_block(outline),
            research=research_block(research, config.max_context_chars),
            competitors=competitor_block(brief),
        ),
        system=article_draft.SYSTEM,
        model=config.model,
        max_output_tokens=article_draft.max_output_tokens(config.target_words),
        reasoning_effort=config.reasoning_effort,
    )
    response = await llm.structured(request, article_draft.ArticleContentOut, purpose=LLMPurpose.ARTICLE_DRAFT, prompt_version=article_draft.VERSION)  # fmt: skip
    labels = _labels(research)
    content, issues = clean_citations(from_output(response.data), labels)
    problems = structural_problems(content, min_words=0, labels=labels)
    if problems:
        raise ContentRejectedError("the draft isn't a usable article: " + "; ".join(problems))
    issues += number_issues(content, allowed_numbers(brief, research))
    return Written(content, issues, [], response.raw.model)


async def edit_draft(llm: BudgetedLLM, brief: ArticleBrief, research: ResearchResult, outline: ArticleOutline, draft: ArticleContent, config: WritingConfig) -> Written:  # fmt: skip
    request = LLMRequest(
        prompt=article_edit.render(
            brief=brief_block(brief, config.target_words),
            company=company_block(brief),
            outline=outline_block(outline),
            research=research_block(research, config.max_context_chars),
            draft=fence(
                DRAFT_TAG, json.dumps(draft.model_dump(mode="json"), ensure_ascii=False, indent=1)
            ),
        ),
        system=article_edit.SYSTEM,
        model=config.model,
        max_output_tokens=article_draft.max_output_tokens(config.target_words),
        reasoning_effort=config.reasoning_effort,
    )
    response = await llm.structured(request, article_edit.EditOut, purpose=LLMPurpose.ARTICLE_EDIT, prompt_version=article_edit.VERSION)  # fmt: skip
    labels = _labels(research)
    content, issues = clean_citations(from_output(response.data.article), labels)
    problems = structural_problems(content, min_words=config.min_words, labels=labels)
    if problems:
        raise ContentRejectedError("the edited article doesn't pass the completion checks: " + "; ".join(problems))  # fmt: skip
    issues += [
        ContentIssue(
            kind="editor_flag",
            detail=f"{flag.issue} ({flag.action}){': ' + flag.note if flag.note else ''}",
            excerpt=flag.excerpt or None,
        )
        for flag in response.data.flags
    ]
    issues += number_issues(content, allowed_numbers(brief, research))
    return Written(content, issues, response.data.changes, response.raw.model)
