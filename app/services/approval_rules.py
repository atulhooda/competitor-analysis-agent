"""The rules that tie an approval to one exact article version and quality report (Phase 7).

Shared by Phase 5 (a new edit), Phase 6 (a new recommended version or report), approvals and
publishing. An approval authorizes publication only while all of these hold:

- it is the article's live decision (not invalidated) and says ``approved``;
- the article is ``ready`` (so every Phase 6 gate passes) and not cancelled;
- its version is the article's recommended version, and its quality report is the article's
  current report for that version.

Otherwise it's stale: ``invalidate_stale`` records that once, with the reason, and never
deletes or rewrites the decision itself.

**Authored reports.** An article a person wrote and imported (``articles import``) carries a
report marked ``authored``: its deterministic gates ran and the ones that need Gemini are
recorded as *not run*. Such a report is publishable, and only on an imported article —
``authored_problems`` refuses it on a generated one, and refuses a not-run gate on a report
that isn't authored. A generated article can never skip its gates this way.
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Article, ArticleApproval, ArticleQualityReport
from app.domain.articles import ArticleOrigin, ArticleStatus
from app.domain.publishing import ApprovalDecision
from app.domain.quality import GateStatus


async def live_decision(session: AsyncSession, article_id: int) -> ArticleApproval | None:
    query = select(ArticleApproval).where(ArticleApproval.article_id == article_id, ArticleApproval.invalidated_at.is_(None))  # fmt: skip
    row: ArticleApproval | None = await session.scalar(query)
    return row


def readiness_problems(article: Article, report: ArticleQualityReport | None) -> list[str]:
    """Why the article can't be approved (or published) now; empty if it can."""
    status = ArticleStatus(article.status)
    if status is ArticleStatus.CANCELLED:
        return ["the article was cancelled"]
    problems = []
    if status is ArticleStatus.NEEDS_REVIEW:
        problems.append("the article is needs_review: resolve Phase 6 first (revise it, or fix what the gates flag, then validate again)")  # fmt: skip
    elif status is not ArticleStatus.READY:
        problems.append(f"the article is {status.value}: only ready articles can be approved and published")  # fmt: skip
    if article.recommended_version_id is None or report is None:
        problems.append("it has no validated recommended version")
    else:
        if report.version_id != article.recommended_version_id:
            problems.append(f"its quality report #{report.id} is for version {report.version_id}, not the recommended version {article.recommended_version_id}")  # fmt: skip
        if not report.passed:
            failed = [g["name"] for g in report.gates if not g.get("passed")]
            problems.append(f"its quality report #{report.id} fails gate(s): {', '.join(failed) or 'unknown'}")  # fmt: skip
        problems += authored_problems(article, report)
    return problems


def authored_problems(article: Article, report: ArticleQualityReport) -> list[str]:
    """An authored report belongs to an imported article and to no other. Checked on every
    approval and again immediately before the CMS is called, so a report that claims gates
    were skipped can never carry an article the agent wrote."""
    not_run = [g["name"] for g in report.gates if g.get("status") == GateStatus.NOT_RUN.value]
    if not report.authored:
        if not_run:
            return [f"quality report #{report.id} records gate(s) as not run ({', '.join(not_run)}) but isn't an authored report: only an article written by a person may skip a gate"]  # fmt: skip
        return []
    if article.origin != ArticleOrigin.IMPORTED.value:
        return [f"quality report #{report.id} is an authored report (a person's own checks, no Gemini gates) but article {article.id} was written by the agent: it must pass its own gates"]  # fmt: skip
    return []


def stale_reason(article: Article, approval: ArticleApproval) -> str | None:
    """Why a (live) decision no longer applies to the article, or None."""
    if article.status == ArticleStatus.CANCELLED.value:
        return "the article was cancelled"
    if article.recommended_version_id != approval.version_id:
        now = f"version {article.recommended_version_id}" if article.recommended_version_id else "no validated version"  # fmt: skip
        return f"the recommended version changed (from {approval.version_id} to {now})"
    if article.quality_report_id != approval.quality_report_id:
        now = f"report #{article.quality_report_id}" if article.quality_report_id else "no report"
        return f"a new validation replaced quality report #{approval.quality_report_id} ({now})"
    return None


def authorization_problem(article: Article, report: ArticleQualityReport | None, approval: ArticleApproval | None) -> str | None:  # fmt: skip
    """Why ``approval`` doesn't authorize publishing the article now, or None if it does."""
    if approval is None:
        return "no approval for the recommended version: approve it first"
    if approval.invalidated_at is not None:
        return f"approval #{approval.id} was invalidated: {approval.invalidated_reason}"
    if approval.decision != ApprovalDecision.APPROVED.value:
        return f"the recommended version was rejected (#{approval.id}): {approval.note or 'no reason given'}"  # fmt: skip
    reason = stale_reason(article, approval)
    if reason:
        return f"approval #{approval.id} no longer applies: {reason}"
    problems = readiness_problems(article, report)
    return "; ".join(problems) if problems else None


async def invalidate_stale(session: AsyncSession, article: Article, now: datetime) -> int:
    """Record that the live decision no longer applies (a new recommended version, a new
    report, cancellation). Returns how many decisions were invalidated."""
    live = await live_decision(session, article.id)
    reason = stale_reason(article, live) if live is not None else None
    if live is None or reason is None:
        return 0
    live.invalidated_at, live.invalidated_reason = now, reason
    return 1


async def invalidate_all(session: AsyncSession, article_id: int, now: datetime, reason: str) -> int:  # fmt: skip
    live = await live_decision(session, article_id)
    if live is None:
        return 0
    live.invalidated_at, live.invalidated_reason = now, reason
    return 1


__all__ = ["authored_problems", "authorization_problem", "invalidate_all", "invalidate_stale", "live_decision", "readiness_problems", "stale_reason"]  # fmt: skip
