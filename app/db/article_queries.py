"""Read queries for articles (Phase 5 drafts, with Phase 6 quality fields), shared by the API
and the CLI."""

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Article,
    ArticleCitation,
    ArticleQualityReport,
    ArticleSource,
    ArticleStepRun,
    ArticleVersion,
    CompanyProfileVersion,
    Opportunity,
    Run,
)
from app.domain.analysis import ContentFormat, SearchIntent
from app.domain.articles import (
    QUALITY_STEPS,
    STEP_ORDER,
    ArticleBrief,
    ArticleContent,
    ArticleDetail,
    ArticleOrigin,
    ArticleOutline,
    ArticleProgress,
    ArticleRunView,
    ArticleStatus,
    ArticleStep,
    ArticleStepView,
    ArticleSummary,
    CitationView,
    ContentIssue,
    ResearchFact,
    SourceType,
    SourceView,
    StepStatus,
    VersionDetail,
    VersionKind,
    VersionSummary,
)
from app.services.article_content import render_markdown

MAX_PAGE_SIZE = 200


def summary(article: Article) -> ArticleSummary:
    return ArticleSummary(
        id=article.id,
        opportunity_id=article.opportunity_id,
        attempt=article.attempt,
        origin=ArticleOrigin(article.origin),
        status=ArticleStatus(article.status),
        current_step=ArticleStep(article.current_step) if article.current_step else None,
        title=article.title,
        slug=article.slug,
        content_type=ContentFormat(article.content_type),
        target_audience=article.target_audience,
        search_intent=SearchIntent(article.search_intent) if article.search_intent else None,
        word_count=article.word_count,
        tokens_used=article.tokens_used,
        error=article.error,
        failed_step=ArticleStep(article.failed_step) if article.failed_step else None,
        created_at=article.created_at,
        updated_at=article.updated_at,
        completed_at=article.completed_at,
        quality_score=article.quality_score,
        revision_count=article.revision_count,
        recommended_version_id=article.recommended_version_id,
        validated_at=article.validated_at,
    )


async def list_articles(
    session: AsyncSession,
    *,
    statuses: Sequence[ArticleStatus] | None = None,
    opportunity_id: int | None = None,
    created_since: datetime | None = None,
    created_until: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ArticleSummary]:
    """Articles, newest first."""
    query = select(Article)
    if statuses:
        query = query.where(Article.status.in_([s.value for s in statuses]))
    if opportunity_id is not None:
        query = query.where(Article.opportunity_id == opportunity_id)
    if created_since:
        query = query.where(Article.created_at >= created_since)
    if created_until:
        query = query.where(Article.created_at < created_until)
    query = query.order_by(Article.created_at.desc(), Article.id.desc()).limit(min(limit, MAX_PAGE_SIZE)).offset(offset)  # fmt: skip
    return [summary(a) for a in await session.scalars(query)]


def _current_ids(article: Article) -> set[int]:
    return {i for i in (article.outline_version_id, article.draft_version_id, article.final_version_id, article.recommended_version_id) if i}  # fmt: skip


def content_version_id(article: Article) -> int | None:
    """The version an article's content comes from: Phase 6's recommended version once it
    has been validated, else the edited version (or the draft until then)."""
    return article.recommended_version_id or article.final_version_id or article.draft_version_id


async def _quality_step_ids(session: AsyncSession, article: Article) -> set[int]:
    """The checkpoint steps behind the recommended version's quality report."""
    if article.quality_report_id is None:
        return set()
    report = await session.get(ArticleQualityReport, article.quality_report_id)
    if report is None:
        return set()
    return {i for i in (report.fact_check_step_id, report.originality_step_id, report.seo_step_id, report.metrics_step_id, report.judge_step_id, report.decision_step_id) if i}  # fmt: skip


def _step_view(row: ArticleStepRun, article: Article, current_brief: int | None, quality_steps: set[int]) -> ArticleStepView:  # fmt: skip
    version_id = (row.output or {}).get("version_id") if row.step in (ArticleStep.OUTLINE.value, ArticleStep.DRAFT.value, ArticleStep.EDIT.value, ArticleStep.REVISION.value) else None  # fmt: skip
    current = (
        row.id == article.research_step_id
        or row.id == current_brief
        or row.id in quality_steps
        or (version_id is not None and version_id in _current_ids(article))
    )
    return ArticleStepView(
        id=row.id,
        step=ArticleStep(row.step),
        status=StepStatus(row.status),
        run_id=row.run_id,
        fingerprint=row.fingerprint,
        prompt_version=row.prompt_version,
        model=row.model,
        llm_calls=row.llm_calls,
        tokens=row.tokens,
        error=row.error,
        started_at=row.started_at,
        finished_at=row.finished_at,
        current=current,
    )


async def _latest_brief_step(session: AsyncSession, article_id: int) -> int | None:
    step_id: int | None = await session.scalar(
        select(ArticleStepRun.id)
        .where(
            ArticleStepRun.article_id == article_id,
            ArticleStepRun.step == ArticleStep.BRIEF.value,
            ArticleStepRun.status == StepStatus.SUCCEEDED.value,
        )
        .order_by(ArticleStepRun.id.desc())
        .limit(1)
    )
    return step_id


def _version_summary(v: ArticleVersion, current: set[int]) -> VersionSummary:
    return VersionSummary(
        id=v.id,
        kind=VersionKind(v.kind),
        number=v.number,
        step_id=v.step_id,
        parent_version_id=v.parent_version_id,
        title=v.title,
        word_count=v.word_count,
        issues=len(v.issues),
        prompt_version=v.prompt_version,
        model=v.model,
        authored=(v.prompt_version or "").startswith("import/"),
        created_at=v.created_at,
        current=v.id in current,
    )


async def _source_map(session: AsyncSession, step_id: int | None) -> dict[str, tuple[str | None, str]]:  # fmt: skip
    if step_id is None:
        return {}
    rows = await session.execute(select(ArticleSource.label, ArticleSource.title, ArticleSource.url).where(ArticleSource.step_id == step_id))  # fmt: skip
    return {label: (title, url) for label, title, url in rows}


async def get_article(session: AsyncSession, article_id: int, *, token_budget: int, include_markdown: bool = False) -> ArticleDetail | None:  # fmt: skip
    row = (
        await session.execute(
            select(Article, Opportunity.title, Opportunity.status, CompanyProfileVersion.version)
            .join(Opportunity, Opportunity.id == Article.opportunity_id)
            .join(CompanyProfileVersion, CompanyProfileVersion.id == Article.company_profile_id)
            .where(Article.id == article_id)
        )
    ).first()
    if row is None:
        return None
    article, opportunity_title, opportunity_status, profile_version = row
    current_brief = await _latest_brief_step(session, article_id)
    step_rows = list(await session.scalars(select(ArticleStepRun).where(ArticleStepRun.article_id == article_id).order_by(ArticleStepRun.id)))  # fmt: skip
    latest: dict[str, ArticleStepRun] = {}
    for step_row in step_rows:
        latest[step_row.step] = step_row
    quality_steps = await _quality_step_ids(session, article)
    steps = [_step_view(latest[s.value], article, current_brief, quality_steps) for s in (*STEP_ORDER, *QUALITY_STEPS, ArticleStep.REVISION) if s.value in latest]  # fmt: skip
    runs = await session.scalars(select(Run).where(Run.article_id == article_id).order_by(Run.id))
    version_id = content_version_id(article)
    content_version = await session.get(ArticleVersion, version_id) if version_id else None
    outline_version = await session.get(ArticleVersion, article.outline_version_id) if article.outline_version_id else None  # fmt: skip
    sources = await session.scalar(select(func.count()).select_from(ArticleSource).where(ArticleSource.step_id == article.research_step_id)) if article.research_step_id else 0  # fmt: skip
    done = [ArticleStep.BRIEF] if current_brief else []
    done += [step for step, pointer in ((ArticleStep.RESEARCH, article.research_step_id), (ArticleStep.OUTLINE, article.outline_version_id), (ArticleStep.DRAFT, article.draft_version_id), (ArticleStep.EDIT, article.final_version_id)) if pointer]  # fmt: skip
    content = ArticleContent.model_validate(content_version.content) if content_version else None
    markdown = None
    if include_markdown and content is not None:
        markdown = render_markdown(content, await _source_map(session, article.research_step_id))
    return ArticleDetail(
        **summary(article).model_dump(),
        opportunity_title=opportunity_title,
        opportunity_status=opportunity_status,
        assessment_id=article.assessment_id,
        company_profile_version=profile_version,
        description=article.description,
        angle=article.angle,
        brief=ArticleBrief.model_validate(article.brief),
        progress=ArticleProgress(
            completed_steps=done, percent=round(100 * len(done) / len(STEP_ORDER))
        ),
        token_budget=token_budget,
        steps=steps,
        runs=[
            ArticleRunView(
                id=r.id,
                status=r.status,
                trigger=r.trigger,
                created_at=r.created_at,
                finished_at=r.finished_at,
                error=r.error,
                summary=r.summary,
            )
            for r in runs
        ],
        content=content,
        content_version=_version_summary(content_version, _current_ids(article))
        if content_version
        else None,
        outline=ArticleOutline.model_validate(outline_version.content) if outline_version else None,
        issues=[ContentIssue.model_validate(i) for i in content_version.issues]
        if content_version
        else [],
        sources=int(sources or 0),
        markdown=markdown,
    )


async def get_sources(session: AsyncSession, article_id: int, *, include_all: bool = False) -> list[SourceView] | None:  # fmt: skip
    """The current research's sources (every research run's with ``include_all``)."""
    article = await session.get(Article, article_id)
    if article is None:
        return None
    query = select(ArticleSource).where(ArticleSource.article_id == article_id)
    if not include_all:
        query = query.where(ArticleSource.step_id == article.research_step_id)
    content_version = content_version_id(article)
    counts: dict[int, int] = {}
    if content_version:
        counts = {sid: n for sid, n in await session.execute(select(ArticleCitation.source_id, func.count()).where(ArticleCitation.version_id == content_version).group_by(ArticleCitation.source_id))}  # fmt: skip
    rows = await session.scalars(query.order_by(ArticleSource.step_id.desc(), ArticleSource.id))
    return [
        SourceView(
            id=s.id, step_id=s.step_id, current=s.step_id == article.research_step_id, label=s.label, url=s.url,
            requested_url=s.requested_url, domain=s.domain, title=s.title, publisher=s.publisher, published=s.published,
            source_type=SourceType(s.source_type), relevance=s.relevance, attribution_required=s.attribution_required,
            excerpt=s.excerpt, facts=[ResearchFact.model_validate(f) for f in s.facts], retrieved_at=s.retrieved_at,
            citations=counts.get(s.id, 0),
        )
        for s in rows
    ]  # fmt: skip


async def list_versions(session: AsyncSession, article_id: int) -> list[VersionSummary] | None:
    article = await session.get(Article, article_id)
    if article is None:
        return None
    rows = await session.scalars(select(ArticleVersion).where(ArticleVersion.article_id == article_id).order_by(ArticleVersion.id))  # fmt: skip
    return [_version_summary(v, _current_ids(article)) for v in rows]


async def get_version(session: AsyncSession, article_id: int, version_id: int, *, include_markdown: bool = False) -> VersionDetail | None:  # fmt: skip
    article = await session.get(Article, article_id)
    version = await session.get(ArticleVersion, version_id)
    if article is None or version is None or version.article_id != article_id:
        return None
    rows = await session.execute(
        select(ArticleCitation, ArticleSource.label, ArticleSource.url, ArticleSource.step_id)
        .join(ArticleSource, ArticleSource.id == ArticleCitation.source_id)
        .where(ArticleCitation.version_id == version_id)
        .order_by(ArticleCitation.id)
    )
    cited = [(CitationView(source_id=c.source_id, label=label, url=url, claim=c.claim, section=c.section_index, block=c.block_index, item=c.item_index), step) for c, label, url, step in rows]  # fmt: skip
    markdown = None
    if include_markdown and version.kind != VersionKind.OUTLINE.value:
        step = cited[0][1] if cited else article.research_step_id
        markdown = render_markdown(ArticleContent.model_validate(version.content), await _source_map(session, step))  # fmt: skip
    return VersionDetail(
        **_version_summary(version, _current_ids(article)).model_dump(),
        content=version.content,
        reason=version.reason,
        issues_addressed=list(version.issues_addressed or []),
        issue_details=[ContentIssue.model_validate(i) for i in version.issues],
        changes=list(version.changes),
        citations=[c for c, _ in cited],
        markdown=markdown,
    )


async def list_steps(session: AsyncSession, article_id: int) -> list[ArticleStepView] | None:
    article = await session.get(Article, article_id)
    if article is None:
        return None
    current_brief = await _latest_brief_step(session, article_id)
    quality_steps = await _quality_step_ids(session, article)
    rows = await session.scalars(select(ArticleStepRun).where(ArticleStepRun.article_id == article_id).order_by(ArticleStepRun.id))  # fmt: skip
    return [_step_view(r, article, current_brief, quality_steps) for r in rows]


__all__ = [
    "MAX_PAGE_SIZE",
    "content_version_id",
    "get_article",
    "get_sources",
    "get_version",
    "list_articles",
    "list_steps",
    "list_versions",
    "summary",
]
