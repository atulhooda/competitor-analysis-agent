"""Read queries for article quality (Phase 6), shared by the API and the CLI.

Every view defaults to the article's recommended version (the one its quality report is
about); pass ``version_id`` to read an earlier or rejected version's results instead.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Article,
    ArticleClaimCheck,
    ArticleQualityReport,
    ArticleSource,
    ArticleStepRun,
    ArticleVersion,
)
from app.domain.articles import ArticleStep, StepStatus, VersionKind
from app.domain.opportunities import ScoreComponent
from app.domain.quality import (
    ClaimCheckView,
    ClaimKind,
    ClaimVerdict,
    FactCheckReport,
    FactCheckView,
    Gate,
    JudgeReport,
    OriginalityReport,
    OriginalityView,
    QualityIssue,
    QualityMetrics,
    QualityOverview,
    QualityReportView,
    RevisionView,
    SEOReport,
    SEOView,
    VersionScore,
)

_REPORT_STEP = {
    ArticleStep.FACT_CHECK: "fact_check_step_id",
    ArticleStep.ORIGINALITY: "originality_step_id",
    ArticleStep.SEO: "seo_step_id",
    ArticleStep.METRICS: "metrics_step_id",
    ArticleStep.JUDGE: "judge_step_id",
    ArticleStep.DECISION: "decision_step_id",
}
_VALIDATED_KINDS = (VersionKind.FINAL.value, VersionKind.REVISION.value)


class UnknownVersionError(LookupError):
    pass


async def _report(session: AsyncSession, article: Article, version_id: int | None) -> ArticleQualityReport | None:  # fmt: skip
    """The recommended version's report, or the latest report for ``version_id``."""
    current = await session.get(ArticleQualityReport, article.quality_report_id) if article.quality_report_id else None  # fmt: skip
    if version_id is None or (current is not None and current.version_id == version_id):
        return current
    report: ArticleQualityReport | None = await session.scalar(
        select(ArticleQualityReport)
        .where(
            ArticleQualityReport.article_id == article.id,
            ArticleQualityReport.version_id == version_id,
        )
        .order_by(ArticleQualityReport.id.desc())
        .limit(1)
    )
    return report


async def _step_row(session: AsyncSession, article: Article, step: ArticleStep, version_id: int | None) -> ArticleStepRun | None:  # fmt: skip
    """The step behind the chosen version's report; before any report exists (a validation
    in progress or failed), the latest successful one."""
    report = await _report(session, article, version_id)
    step_id = getattr(report, _REPORT_STEP[step]) if report is not None else None
    if step_id is not None:
        return await session.get(ArticleStepRun, step_id)
    query = select(ArticleStepRun).where(
        ArticleStepRun.article_id == article.id,
        ArticleStepRun.step == step.value,
        ArticleStepRun.status == StepStatus.SUCCEEDED.value,
    )
    if version_id is not None:
        query = query.where(ArticleStepRun.version_id == version_id)
    row: ArticleStepRun | None = await session.scalar(query.order_by(ArticleStepRun.id.desc()).limit(1))  # fmt: skip
    return row


async def _checked_version(session: AsyncSession, article_id: int, version_id: int | None) -> Article | None:  # fmt: skip
    article = await session.get(Article, article_id)
    if article is None:
        return None
    if version_id is not None:
        version = await session.get(ArticleVersion, version_id)
        if version is None or version.article_id != article_id:
            raise UnknownVersionError(f"Article {article_id} has no version {version_id}")
    return article


def _report_view(report: ArticleQualityReport, version: ArticleVersion) -> QualityReportView:
    return QualityReportView(
        id=report.id,
        article_id=report.article_id,
        version_id=report.version_id,
        version_kind=version.kind,
        version_number=version.number,
        run_id=report.run_id,
        overall_score=report.overall_score,
        breakdown=[ScoreComponent.model_validate(c) for c in report.breakdown],
        gates=[Gate.model_validate(g) for g in report.gates],
        passed=report.passed,
        authored=report.authored,
        issues=[QualityIssue.model_validate(i) for i in report.issues],
        config_fingerprint=report.config_fingerprint,
        fact_check_step_id=report.fact_check_step_id,
        originality_step_id=report.originality_step_id,
        seo_step_id=report.seo_step_id,
        metrics_step_id=report.metrics_step_id,
        judge_step_id=report.judge_step_id,
        created_at=report.created_at,
    )


async def _latest_reports(session: AsyncSession, article_id: int) -> dict[int, ArticleQualityReport]:  # fmt: skip
    rows = await session.scalars(select(ArticleQualityReport).where(ArticleQualityReport.article_id == article_id).order_by(ArticleQualityReport.id))  # fmt: skip
    return {r.version_id: r for r in rows}


async def quality_overview(session: AsyncSession, article_id: int, *, token_budget: int, version_id: int | None = None) -> QualityOverview | None:  # fmt: skip
    """The status, the recommended (or chosen) version's report with its metrics and the
    judge's rubric, and every validated version's score."""
    article = await _checked_version(session, article_id, version_id)
    if article is None:
        return None
    report = await _report(session, article, version_id)
    report_view = metrics = judged = None
    if report is not None:
        version = await session.get_one(ArticleVersion, report.version_id)
        report_view = _report_view(report, version)
        metrics_row = await session.get(ArticleStepRun, report.metrics_step_id) if report.metrics_step_id else None  # fmt: skip
        judge_row = await session.get(ArticleStepRun, report.judge_step_id) if report.judge_step_id else None  # fmt: skip
        metrics = QualityMetrics.model_validate(metrics_row.output) if metrics_row and metrics_row.output else None  # fmt: skip
        judged = JudgeReport.model_validate(judge_row.output) if judge_row and judge_row.output else None  # fmt: skip
    latest = await _latest_reports(session, article_id)
    versions = await session.scalars(select(ArticleVersion).where(ArticleVersion.article_id == article_id, ArticleVersion.kind.in_(_VALIDATED_KINDS)).order_by(ArticleVersion.id))  # fmt: skip
    return QualityOverview(
        article_id=article.id,
        status=article.status,
        current_step=article.current_step,
        recommended_version_id=article.recommended_version_id,
        quality_score=article.quality_score,
        revision_count=article.revision_count,
        validated_at=article.validated_at,
        quality_tokens_used=article.quality_tokens_used,
        token_budget=token_budget,
        report=report_view,
        metrics=metrics,
        judge=judged,
        versions=[
            VersionScore(
                version_id=v.id, kind=v.kind, number=v.number, parent_version_id=v.parent_version_id,
                report_id=latest[v.id].id, score=latest[v.id].overall_score, passed=latest[v.id].passed,
                recommended=v.id == article.recommended_version_id,
            )
            for v in versions
            if v.id in latest
        ],
    )  # fmt: skip


async def fact_check(session: AsyncSession, article_id: int, *, version_id: int | None = None, verdicts: set[ClaimVerdict] | None = None) -> FactCheckView | None:  # fmt: skip
    """Every claim check of a version's fact-check (each cited claim and source pair, then the
    uncited factual claims), with the verdict, explanation, evidence and provenance."""
    article = await _checked_version(session, article_id, version_id)
    if article is None:
        return None
    row = await _step_row(session, article, ArticleStep.FACT_CHECK, version_id)
    if row is None or row.output is None:
        return None
    report = FactCheckReport.model_validate(row.output)
    query = (
        select(ArticleClaimCheck, ArticleSource.label, ArticleSource.url)
        .outerjoin(ArticleSource, ArticleSource.id == ArticleClaimCheck.source_id)
        .where(ArticleClaimCheck.step_id == row.id)
        .order_by(
            ArticleClaimCheck.kind,
            ArticleClaimCheck.section_index,
            ArticleClaimCheck.block_index,
            ArticleClaimCheck.id,
        )
    )
    if verdicts:
        query = query.where(ArticleClaimCheck.verdict.in_([v.value for v in verdicts]))
    checks = [
        ClaimCheckView(
            id=c.id, kind=ClaimKind(c.kind), section=c.section_index, block=c.block_index, item=c.item_index,
            claim=c.claim, source_id=c.source_id, source_label=label, source_url=url, verdict=ClaimVerdict(c.verdict),
            explanation=c.explanation, evidence=c.evidence, evidence_verified=c.evidence_verified,
            confidence=c.confidence, reread=c.reread, claim_type=c.claim_type, reused=c.reused_from_id is not None,
            model=c.model, prompt_version=c.prompt_version, created_at=c.created_at,
        )
        for c, label, url in await session.execute(query)
    ]  # fmt: skip
    return FactCheckView(article_id=article_id, version_id=row.version_id, step_id=row.id, metrics=report.metrics, checks=checks, notes=report.notes)  # fmt: skip


async def originality(session: AsyncSession, article_id: int, *, version_id: int | None = None) -> OriginalityView | None:  # fmt: skip
    article = await _checked_version(session, article_id, version_id)
    if article is None:
        return None
    row = await _step_row(session, article, ArticleStep.ORIGINALITY, version_id)
    if row is None or row.output is None:
        return None
    return OriginalityView(article_id=article_id, version_id=row.version_id, step_id=row.id, report=OriginalityReport.model_validate(row.output))  # fmt: skip


async def seo(session: AsyncSession, article_id: int, *, version_id: int | None = None) -> SEOView | None:  # fmt: skip
    article = await _checked_version(session, article_id, version_id)
    if article is None:
        return None
    row = await _step_row(session, article, ArticleStep.SEO, version_id)
    if row is None or row.output is None:
        return None
    return SEOView(article_id=article_id, version_id=row.version_id, step_id=row.id, report=SEOReport.model_validate(row.output))  # fmt: skip


async def revisions(session: AsyncSession, article_id: int) -> list[RevisionView] | None:
    """The edited version and every revision, oldest first, each with its latest score."""
    article = await session.get(Article, article_id)
    if article is None:
        return None
    latest = await _latest_reports(session, article_id)
    rows = await session.scalars(select(ArticleVersion).where(ArticleVersion.article_id == article_id, ArticleVersion.kind.in_(_VALIDATED_KINDS)).order_by(ArticleVersion.id))  # fmt: skip
    return [
        RevisionView(
            version_id=v.id, kind=v.kind, number=v.number, parent_version_id=v.parent_version_id, reason=v.reason,
            issues_addressed=list(v.issues_addressed or []), changes=list(v.changes), tokens=v.tokens,
            word_count=v.word_count, prompt_version=v.prompt_version, model=v.model, created_at=v.created_at,
            score=latest[v.id].overall_score if v.id in latest else None,
            passed=latest[v.id].passed if v.id in latest else None,
            recommended=v.id == article.recommended_version_id,
        )
        for v in rows
    ]  # fmt: skip


__all__ = ["UnknownVersionError", "fact_check", "originality", "quality_overview", "revisions", "seo"]  # fmt: skip
