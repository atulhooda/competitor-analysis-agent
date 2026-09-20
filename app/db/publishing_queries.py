"""Read queries for publications (Phase 7), shared by the API and the CLI."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Article, Publication, PublicationAttempt
from app.domain.publishing import (
    AttemptAction,
    AttemptOutcome,
    PublicationAttemptView,
    PublicationStatus,
    PublicationView,
    TargetStatus,
)


def _view(p: Publication, attempts: list[PublicationAttempt]) -> PublicationView:
    return PublicationView(
        id=p.id,
        article_id=p.article_id,
        version_id=p.version_id,
        approval_id=p.approval_id,
        cms=p.cms,
        site=p.site,
        status=PublicationStatus(p.status),
        target_status=TargetStatus(p.target_status),
        external_id=p.external_id,
        external_status=p.external_status,
        url=p.url,
        edit_url=p.edit_url,
        idempotency_key=p.idempotency_key,
        attempt_count=p.attempt_count,
        last_error=p.last_error,
        content_hash=p.content_hash,
        details=dict(p.details),
        preflight=p.preflight,
        superseded_by_id=p.superseded_by_id,
        run_id=p.run_id,
        created_at=p.created_at,
        updated_at=p.updated_at,
        published_at=p.published_at,
        attempts=[
            PublicationAttemptView(
                id=a.id, run_id=a.run_id, action=AttemptAction(a.action), outcome=AttemptOutcome(a.outcome),
                http_status=a.http_status, external_id=a.external_id, error=a.error, started_at=a.started_at,
                finished_at=a.finished_at,
            )
            for a in attempts
        ],
    )  # fmt: skip


async def _attempts(session: AsyncSession, publication_ids: list[int]) -> dict[int, list[PublicationAttempt]]:  # fmt: skip
    grouped: dict[int, list[PublicationAttempt]] = {i: [] for i in publication_ids}
    if publication_ids:
        rows = await session.scalars(select(PublicationAttempt).where(PublicationAttempt.publication_id.in_(publication_ids)).order_by(PublicationAttempt.id))  # fmt: skip
        for row in rows:
            grouped[row.publication_id].append(row)
    return grouped


async def list_publications(session: AsyncSession, article_id: int) -> list[PublicationView] | None:
    """Every publication of the article, newest first, with its attempts."""
    if await session.get(Article, article_id) is None:
        return None
    rows = list(await session.scalars(select(Publication).where(Publication.article_id == article_id).order_by(Publication.id.desc())))  # fmt: skip
    attempts = await _attempts(session, [p.id for p in rows])
    return [_view(p, attempts[p.id]) for p in rows]


async def current_publication(session: AsyncSession, article_id: int) -> PublicationView | None:
    """The article's latest publication (None if the article has none)."""
    row = await session.scalar(select(Publication).where(Publication.article_id == article_id).order_by(Publication.id.desc()).limit(1))  # fmt: skip
    if row is None:
        return None
    return _view(row, (await _attempts(session, [row.id]))[row.id])


__all__ = ["current_publication", "list_publications"]
