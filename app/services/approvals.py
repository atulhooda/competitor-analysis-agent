"""Article approval (Phase 7): a person's (or, only if explicitly enabled, the auto-approval
policy's) decision on the exact version and quality report that made an article ready.

- Only ``ready`` articles can be approved or rejected; ``needs_review`` must be resolved in
  Phase 6 first. There is no manual override.
- A decision is recorded with the version, the quality report, the approver, the method
  (manual or auto), the channel (API, CLI, policy), a note and the time. Rejections need a
  reason.
- Approving an already-approved version again changes nothing; any other new decision
  supersedes the live one, which is invalidated (kept, with the reason). A new recommended
  version or quality report invalidates it too.
- Auto-approval (``PUBLISH_AUTO_APPROVE``, off by default) applies the same checks and never
  overrides a rejection of the same version and report.
"""

from collections.abc import Callable
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.timeutils import utcnow
from app.db.models import Article, ArticleApproval, ArticleQualityReport, ArticleVersion
from app.db.session import SessionFactory
from app.domain.publishing import (
    ApprovalChannel,
    ApprovalDecision,
    ApprovalMethod,
    ApprovalRecord,
    ApprovalState,
    ApprovalView,
)
from app.domain.quality import Gate
from app.services.approval_rules import (
    authorization_problem,
    invalidate_stale,
    live_decision,
    readiness_problems,
    stale_reason,
)
from app.services.articles import ArticleConflictError, ArticleNotFoundError

_DEFAULT_APPROVER = {ApprovalChannel.API: "api", ApprovalChannel.CLI: "cli", ApprovalChannel.POLICY: "auto-approval policy"}  # fmt: skip


class ApprovalRefusedError(ArticleConflictError):
    """The article can't be approved or rejected now (not ready, needs_review, ...)."""


def record(row: ArticleApproval) -> ApprovalRecord:
    return ApprovalRecord(
        id=row.id,
        article_id=row.article_id,
        version_id=row.version_id,
        quality_report_id=row.quality_report_id,
        decision=ApprovalDecision(row.decision),
        method=ApprovalMethod(row.method),
        channel=ApprovalChannel(row.channel),
        approver=row.approver,
        note=row.note,
        created_at=row.created_at,
        invalidated_at=row.invalidated_at,
        invalidated_reason=row.invalidated_reason,
        live=row.invalidated_at is None,
    )


class ApprovalService:
    def __init__(self, sessions: SessionFactory, settings: Settings, *, now: Callable[[], datetime] = utcnow) -> None:  # fmt: skip
        self._sessions = sessions
        self._settings = settings
        self._now = now

    async def approve(self, article_id: int, *, channel: ApprovalChannel, approver: str | None = None, note: str | None = None) -> tuple[ApprovalRecord, bool]:  # fmt: skip
        """(the approval, whether it's new). Approving the approved version again is a no-op."""
        return await self._decide(article_id, ApprovalDecision.APPROVED, channel=channel, approver=approver, note=note)  # fmt: skip

    async def reject(self, article_id: int, *, channel: ApprovalChannel, note: str, approver: str | None = None) -> tuple[ApprovalRecord, bool]:  # fmt: skip
        if not note or not note.strip():
            raise ApprovalRefusedError('a rejection needs a reason (--note / "note")')
        return await self._decide(article_id, ApprovalDecision.REJECTED, channel=channel, approver=approver, note=note)  # fmt: skip

    async def _decide(self, article_id: int, decision: ApprovalDecision, *, channel: ApprovalChannel, approver: str | None, note: str | None) -> tuple[ApprovalRecord, bool]:  # fmt: skip
        now = self._now()
        async with self._sessions() as session, session.begin():
            article = await session.get(Article, article_id, with_for_update=True)
            if article is None:
                raise ArticleNotFoundError(f"Unknown article {article_id}")
            report = await session.get(ArticleQualityReport, article.quality_report_id) if article.quality_report_id else None  # fmt: skip
            problems = readiness_problems(article, report)
            if problems:
                verb = "approved" if decision is ApprovalDecision.APPROVED else "rejected"
                raise ApprovalRefusedError(f"Article {article_id} can't be {verb}: " + "; ".join(problems))  # fmt: skip
            row, created = await self.decide_in(session, article, decision, method=ApprovalMethod.MANUAL, channel=channel, approver=approver, note=note, now=now)  # fmt: skip
            return record(row), created

    @staticmethod
    async def decide_in(session: AsyncSession, article: Article, decision: ApprovalDecision, *, method: ApprovalMethod, channel: ApprovalChannel, approver: str | None, note: str | None, now: datetime) -> tuple[ArticleApproval, bool]:  # fmt: skip
        """Record a decision inside the caller's transaction (the article row locked, its
        readiness checked). The live decision it replaces is invalidated, not deleted."""
        if article.recommended_version_id is None or article.quality_report_id is None:
            raise ApprovalRefusedError(f"Article {article.id} has no validated recommended version")  # fmt: skip
        await invalidate_stale(session, article, now)
        live = await live_decision(session, article.id)
        if live is not None and live.decision == decision.value and (live.method == method.value or method is ApprovalMethod.AUTO):  # fmt: skip
            return live, False  # the same decision on the same version and report: no-op
        if live is not None:
            live.invalidated_at = now
            await session.flush()  # free the one-live-decision slot first
        row = ArticleApproval(
            article_id=article.id,
            version_id=article.recommended_version_id,
            quality_report_id=article.quality_report_id,
            decision=decision.value,
            method=method.value,
            channel=channel.value,
            approver=(approver or "").strip()[:200] or _DEFAULT_APPROVER[channel],
            note=note.strip()[:2_000] if note and note.strip() else None,
            created_at=now,
        )
        session.add(row)
        await session.flush()
        if live is not None:
            live.invalidated_reason = f"superseded by {'approval' if decision is ApprovalDecision.APPROVED else 'rejection'} #{row.id}"  # fmt: skip
        return row, True

    async def auto_approve_in(self, session: AsyncSession, article: Article, now: datetime) -> ArticleApproval | None:  # fmt: skip
        """The auto-approval policy, only when PUBLISH_AUTO_APPROVE is on. It approves only a
        ready article (every gate passed) and never overrides a person's rejection of the
        same version and report. Returns the approval, or None."""
        if not self._settings.publish_auto_approve:
            return None
        report = await session.get(ArticleQualityReport, article.quality_report_id) if article.quality_report_id else None  # fmt: skip
        if readiness_problems(article, report):
            return None
        await invalidate_stale(session, article, now)
        live = await live_decision(session, article.id)
        if live is not None and live.decision == ApprovalDecision.REJECTED.value:
            return None
        if live is not None:  # already approved (manually or automatically)
            return live
        row, _ = await self.decide_in(session, article, ApprovalDecision.APPROVED, method=ApprovalMethod.AUTO, channel=ApprovalChannel.POLICY, approver=None, note="PUBLISH_AUTO_APPROVE: ready, every quality gate passed", now=now)  # fmt: skip
        return row

    async def view(self, article_id: int) -> ApprovalView:
        """The article's approval as it stands (read-only)."""
        async with self._sessions() as session:
            article = await session.get(Article, article_id)
            if article is None:
                raise ArticleNotFoundError(f"Unknown article {article_id}")
            report = await session.get(ArticleQualityReport, article.quality_report_id) if article.quality_report_id else None  # fmt: skip
            version = await session.get(ArticleVersion, article.recommended_version_id) if article.recommended_version_id else None  # fmt: skip
            live = await live_decision(session, article_id)
            last = await session.scalar(select(ArticleApproval).where(ArticleApproval.article_id == article_id).order_by(ArticleApproval.id.desc()).limit(1))  # fmt: skip
        problems = readiness_problems(article, report)
        stale = stale_reason(article, live) if live is not None else None
        if problems:
            state = ApprovalState.NOT_READY
        elif live is not None and stale is None:
            state = ApprovalState.APPROVED if live.decision == ApprovalDecision.APPROVED.value else ApprovalState.REJECTED  # fmt: skip
        elif last is not None and (stale is not None or last.invalidated_at is not None):
            state = ApprovalState.INVALIDATED
        else:
            state = ApprovalState.PENDING
        problem = authorization_problem(article, report, live)
        blocking = problems or ([problem] if problem else [])
        live_record = record(live) if live is not None else None
        if live_record is not None and stale is not None:  # stale but not yet recorded as such
            live_record = live_record.model_copy(update={"live": False, "invalidated_reason": stale})  # fmt: skip
        return ApprovalView(
            article_id=article.id,
            article_status=article.status,
            state=state,
            can_publish=problem is None,
            blocking=blocking,
            recommended_version_id=article.recommended_version_id,
            version_kind=version.kind if version else None,
            version_number=version.number if version else None,
            quality_report_id=article.quality_report_id,
            quality_score=report.overall_score if report else None,
            gates_passed=report.passed if report else None,
            gates=[Gate.model_validate(g) for g in report.gates] if report else [],
            decision=live_record if live_record is not None and live_record.live else None,
            last_decision=record(last)
            if last is not None and (live is None or last.id != live.id)
            else live_record,
            auto_approve=self._settings.publish_auto_approve,
        )

    async def history(self, article_id: int) -> list[ApprovalRecord]:
        async with self._sessions() as session:
            if await session.get(Article, article_id) is None:
                raise ArticleNotFoundError(f"Unknown article {article_id}")
            rows = await session.scalars(select(ArticleApproval).where(ArticleApproval.article_id == article_id).order_by(ArticleApproval.id))  # fmt: skip
            return [record(r) for r in rows]


__all__ = ["ApprovalRefusedError", "ApprovalService", "record"]
