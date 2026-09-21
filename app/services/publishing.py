"""Publishing (Phase 7): an approved, ready article version → a CMS post. Drafts by default;
nothing is scheduled. Phase 7 publishes only approved, ready article versions.

    request   the article is ready; its recommended version and current quality report
              carry a live approval (or PUBLISH_AUTO_APPROVE records one); a publication row
              (idempotency key: article, version, CMS, site) and a run are queued
    execute   under the article lock: preflight (article, version, quality, approval,
              content, citations, links, SEO fields, target status, CMS config, the CMS
              itself, the post, the slug, categories and tags, idempotency)
              → terms → re-check the approval → create or update the post → verify
              → record (and, once public, mark the opportunity used)

It can't publish anything else: the version is always the article's recommended version,
the content is rendered from that stored version only (never from a request), and the
approval must match that version and report at the moment the CMS is called.

**Idempotency.** One publication per (article, version, CMS, site); the CMS post id is
stored as soon as it's known, and later versions update the same post. Every post carries
an opaque marker. A creation whose answer was lost (a timeout after the CMS saved it) is
never retried blindly: the post is looked up by slug and marker first and adopted if found.
A post that isn't ours is never updated, even if its slug matches.

**Draft first.** Publishing leaves a draft unless the target is ``publish``, which needs
``PUBLISH_ALLOW_DIRECT_PUBLISH``; with ``PUBLISH_DRAFT_FIRST`` (default) the draft is
written and verified before it goes public. A public post is never taken back to draft.
"""

import asyncio
import dataclasses
import hashlib
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.cms import LazyCMS
from app.cms.base import CMSPost, CMSPublisher, TermResolution
from app.cms.errors import CMSConfigurationError, CMSError, CMSTransientError, CMSValidationError
from app.config import Settings
from app.core.errors import AppError
from app.core.timeutils import utcnow
from app.db.locks import article_lock
from app.db.models import (
    Article,
    ArticleApproval,
    ArticleCitation,
    ArticleQualityReport,
    ArticleSource,
    ArticleStepRun,
    ArticleVersion,
    CompanyProfileVersion,
    Competitor,
    ContentItem,
    Opportunity,
    OpportunityEvent,
    Publication,
    PublicationAttempt,
    Run,
)
from app.db.session import SessionFactory
from app.domain.articles import ArticleContent, ArticleStatus, VersionKind
from app.domain.history import ACTIVE_RUN_STATUSES, RunStatus, RunTrigger
from app.domain.opportunities import OpportunityEventKind, OpportunityStatus
from app.domain.publishing import (
    IN_FLIGHT_PUBLICATION,
    AttemptAction,
    AttemptOutcome,
    CMSPostStatus,
    DryRunReport,
    PreflightCheck,
    PreflightReport,
    PublicationStatus,
    RenderedDocument,
    TargetStatus,
)
from app.domain.quality import SEOReport
from app.services.approval_rules import authorization_problem, live_decision, readiness_problems
from app.services.approvals import ApprovalService
from app.services.article_brief import domain_of
from app.services.article_content import structural_problems
from app.services.article_render import render_article
from app.services.articles import ArticleConflictError, ArticleNotFoundError, ArticleRunActiveError
from app.services.checkpoints import new_run
from app.services.covers import CoverService
from app.services.daily_limits import reserve_publication_slot
from app.services.runs import fail_abandoned_runs, finish_run, run_slot_free

log = structlog.get_logger(__name__)

RUN_KIND = "article_publish"
_FINAL = {TargetStatus.DRAFT: PublicationStatus.DRAFT_CREATED, TargetStatus.PENDING: PublicationStatus.DRAFT_CREATED, TargetStatus.PUBLISH: PublicationStatus.PUBLISHED}  # fmt: skip
_EXPECTED = {TargetStatus.DRAFT: CMSPostStatus.DRAFT, TargetStatus.PENDING: CMSPostStatus.PENDING, TargetStatus.PUBLISH: CMSPostStatus.PUBLISHED}  # fmt: skip


class PublishingConflictError(ArticleConflictError):
    """Publishing is refused before anything is queued (not ready, not approved, ...)."""


class ApprovalRequiredError(PublishingConflictError):
    pass


class _Blocked(AppError):
    pass


class _LimitReached(AppError):
    pass


@dataclass(frozen=True)
class PublishRequestResult:
    article_id: int
    publication_id: int | None
    run_id: int | None
    created: bool  # a new publication (else the existing one for this version)
    status: PublicationStatus | None
    message: str | None = None
    queued: bool = True  # this request queued a run (False: one was already in progress)


@dataclass(frozen=True)
class PublishOutcome:
    article_id: int
    publication_id: int
    run_id: int
    run_status: RunStatus
    status: PublicationStatus
    action: str | None
    external_id: str | None
    url: str | None
    error: str | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Snapshot:
    article: Article
    version: ArticleVersion | None
    content: ArticleContent | None
    report: ArticleQualityReport | None
    seo: SEOReport | None
    sources: dict[str, tuple[str | None, str]]
    allowed_internal: set[str]
    allowed_external: set[str]
    live: ArticleApproval | None
    publication: Publication | None  # this version's publication on this site
    lineage: Publication | None  # the article's latest publication that reached the CMS
    marker: str
    key: str | None
    opportunity_title: str | None = None


@dataclass
class _Plan:
    report: PreflightReport
    terms: TermResolution | None = None
    post: CMSPost | None = None
    slug: str | None = None
    action: str = "create"
    transient: bool = False  # the CMS couldn't be read (retry later)
    warnings: list[str] = field(default_factory=list)


def idempotency_key(article_id: int, version_id: int, cms: str, site: str) -> str:
    return hashlib.sha256(f"{article_id}:{version_id}:{cms}:{site}".encode()).hexdigest()


class PublishingService:
    def __init__(
        self,
        engine: AsyncEngine,
        sessions: SessionFactory,
        settings: Settings,
        cms: LazyCMS,
        *,
        now: Callable[[], datetime] = utcnow,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        covers: CoverService | None = None,
    ) -> None:
        """``covers`` (PUBLISH_COVER_IMAGES) gives each published version one cover picture,
        from the configured source (COVER_IMAGE_SOURCE). It is consulted only once a run is
        about to change the target, so blocked publications, dry runs and no-op republishes
        never generate an image or search for a photo."""
        self._engine = engine
        self._sessions = sessions
        self._settings = settings
        self._cms = cms
        self._now = now
        self._sleep = sleep
        self._covers = covers
        self._approvals = ApprovalService(sessions, settings, now=now)

    def _target(self, target: TargetStatus | str | None) -> TargetStatus:
        return TargetStatus(target or self._settings.publish_default_status)

    # ── read-only ────────────────────────────────────────────────────────────

    async def preflight(self, article_id: int, *, target: TargetStatus | str | None = None) -> PreflightReport:  # fmt: skip
        return (await self.dry_run(article_id, target=target)).preflight

    async def dry_run(self, article_id: int, *, target: TargetStatus | str | None = None) -> DryRunReport:  # fmt: skip
        """Every check, the rendered article and the CMS request body, with no change made
        anywhere: the CMS is only read, through a client that refuses any change."""
        wanted = self._target(target)
        async with self._sessions() as session:
            snap = await self._snapshot(session, article_id)
        document = self._render(snap)
        publisher = self._cms.read_only() if self._cms.configured else None
        try:
            plan = await self._preflight(snap, document, wanted, publisher)
        finally:
            if publisher is not None:
                await publisher.aclose()
        payload = None
        if document is not None:
            doc = document.model_copy(update={"slug": plan.slug or document.slug})
            terms = plan.terms or TermResolution(None, ())
            if publisher is not None:  # the exact payload a run would send (no request is made)
                try:
                    payload = publisher.build_payload(doc, status=wanted, terms=terms, marker=snap.marker)  # fmt: skip
                except CMSValidationError as exc:
                    raise PublishingConflictError(str(exc)) from exc
            else:
                payload = self._cms.preview_payload(doc, status=wanted, terms=terms, marker=snap.marker)  # fmt: skip
        return DryRunReport(preflight=plan.report, document=document, payload=payload)

    # ── requests ─────────────────────────────────────────────────────────────

    async def request(self, article_id: int, *, trigger: RunTrigger, target: TargetStatus | str | None = None, daily_limit: bool = False) -> PublishRequestResult:  # fmt: skip
        """Queue the publication of the article's recommended version. Refused unless it is
        ready and approved (checked again, with the CMS, when the run executes).

        ``daily_limit`` (the autonomous pipeline, Phase 8): making the post public needs one
        of today's MAX_ARTICLES_PER_DAY slots, reserved atomically right before the CMS
        call. Publications made by hand keep Phase 7's behavior (they still count)."""
        wanted = self._target(target)
        if not self._cms.configured:
            raise CMSConfigurationError(self._cms.configuration_hint)
        now = self._now()
        async with self._sessions() as session, session.begin():
            article = await session.get(Article, article_id, with_for_update=True)
            if article is None:
                raise ArticleNotFoundError(f"Unknown article {article_id}")
            active = await self._active_publication(session, article_id)
            free = await run_slot_free(session, kind=RUN_KIND, competitor_id=None, lock=article_lock(self._engine, article_id), now=now, article_id=article_id)  # fmt: skip
            if not free:
                if active is not None:
                    return PublishRequestResult(article_id, active.id, active.run_id, False, PublicationStatus(active.status), f"publication #{active.id} is already in progress (run {active.run_id})", queued=False)  # fmt: skip
                raise ArticleRunActiveError(f"Article {article_id} is being processed right now")
            snap = await self._snapshot(session, article_id)
            problems = readiness_problems(article, snap.report)
            if problems:
                raise PublishingConflictError(f"Article {article_id} can't be published: " + "; ".join(problems))  # fmt: skip
            approval = snap.live if authorization_problem(article, snap.report, snap.live) is None else None  # fmt: skip
            if approval is None:
                approval = await self._approvals.auto_approve_in(session, article, now)
            problem = authorization_problem(article, snap.report, approval or snap.live)
            if problem or approval is None:
                raise ApprovalRequiredError(f"Article {article_id} can't be published: {problem}")
            if wanted is TargetStatus.PUBLISH and not self._settings.publish_allow_direct_publish:
                raise PublishingConflictError("Making a post public needs PUBLISH_ALLOW_DIRECT_PUBLISH=true (drafts don't)")  # fmt: skip
            if snap.version is None or snap.key is None:
                raise PublishingConflictError(f"Article {article_id} has no publishable version")
            publication = snap.publication
            created = publication is None
            if publication is None:
                publication = Publication(article_id=article_id, version_id=snap.version.id, approval_id=approval.id, cms=self._cms.name, site=self._cms.site or "", marker=snap.marker, idempotency_key=snap.key, status=PublicationStatus.QUEUED.value, target_status=wanted.value, details={}, created_at=now, updated_at=now)  # fmt: skip
                session.add(publication)
                await session.flush()
            else:
                if publication.status in IN_FLIGHT_PUBLICATION:  # its run died (the slot was free)
                    publication.last_error = (
                        "interrupted before it finished: reconciled on this run"
                    )
                publication.status, publication.target_status, publication.approval_id = PublicationStatus.QUEUED.value, wanted.value, approval.id  # fmt: skip
                publication.updated_at = now
            run = new_run(RUN_KIND, article_id, trigger, now, {"publication_id": publication.id, "target_status": wanted.value, "daily_limit": self._settings.max_articles_per_day if daily_limit else None})  # fmt: skip
            session.add(run)
            await session.flush()
            publication.run_id = run.id
            return PublishRequestResult(article_id, publication.id, run.id, created, PublicationStatus.QUEUED, None if created else f"publication #{publication.id} of this version exists: it is checked against the CMS and updated only if needed")  # fmt: skip

    async def publish_now(self, article_id: int, *, trigger: RunTrigger, target: TargetStatus | str | None = None, daily_limit: bool = False) -> tuple[PublishRequestResult, PublishOutcome | None]:  # fmt: skip
        result = await self.request(article_id, trigger=trigger, target=target, daily_limit=daily_limit)  # fmt: skip
        if not result.queued or result.run_id is None:  # another request's run is in progress
            return result, None
        return result, await self.execute(result.run_id)

    # ── execution ────────────────────────────────────────────────────────────

    async def execute(self, run_id: int) -> PublishOutcome:
        async with self._sessions() as session:
            run = await session.get_one(Run, run_id)
            article_id, params = run.article_id, dict(run.params)
        if article_id is None:
            raise ValueError(f"Run {run_id} is not an article run")
        publication_id = int(params["publication_id"])
        target = TargetStatus(params["target_status"])
        limit = params.get("daily_limit")
        async with article_lock(self._engine, article_id) as acquired:
            if not acquired:
                return await self._finish(run_id, publication_id, RunStatus.FAILED, None, "another run is processing this article", mark_failed=False)  # fmt: skip
            try:
                async with self._sessions() as session, session.begin():
                    now = self._now()
                    await fail_abandoned_runs(session, kind=RUN_KIND, competitor_id=None, now=now, keep=run_id, article_id=article_id)  # fmt: skip
                    run = await session.get_one(Run, run_id)
                    run.status, run.started_at = RunStatus.RUNNING.value, now
                    publication = await session.get_one(Publication, publication_id)
                    publication.status, publication.run_id, publication.updated_at = PublicationStatus.PREFLIGHT.value, run_id, now  # fmt: skip
                return await self._execute(run_id, article_id, publication_id, target, int(limit) if limit is not None else None)  # fmt: skip
            except asyncio.CancelledError:
                await self._fail(publication_id, "interrupted: the server stopped during publishing (the next run reconciles with the CMS first)")  # fmt: skip
                await finish_run(self._sessions, run_id, status=RunStatus.FAILED, now=self._now(), error="cancelled")  # fmt: skip
                raise
            except Exception as exc:  # the publication and run must never be left in progress
                log.exception("publishing.crashed", article_id=article_id, run_id=run_id)
                return await self._finish(run_id, publication_id, RunStatus.FAILED, None, f"{type(exc).__name__}: {exc}")  # fmt: skip

    async def _execute(self, run_id: int, article_id: int, publication_id: int, target: TargetStatus, limit: int | None = None) -> PublishOutcome:  # fmt: skip
        async with self._sessions() as session:
            snap = await self._snapshot(session, article_id)
        if snap.publication is None or snap.publication.id != publication_id:
            return await self._blocked(run_id, publication_id, None, "the article's recommended version changed since this publication was requested: approve and publish the new version")  # fmt: skip
        document = self._render(snap)
        publisher = self._cms.get()
        plan = await self._preflight(snap, document, target, publisher, run_id=run_id)
        async with self._sessions() as session, session.begin():
            publication = await session.get_one(Publication, publication_id)
            publication.preflight = plan.report.model_dump(mode="json")
        if not plan.report.ready or document is None:
            reasons = (
                "; ".join(f"{c.name}: {c.detail}" for c in plan.report.blocking) or "no content"
            )
            if plan.transient:  # the CMS is unavailable: a later run may succeed
                return await self._finish(run_id, publication_id, RunStatus.FAILED, plan, reasons)
            return await self._blocked(run_id, publication_id, plan, reasons)
        if plan.report.action == "none":
            async with self._sessions() as session, session.begin():
                publication = await session.get_one(Publication, publication_id)
                publication.status, publication.last_error, publication.updated_at = _FINAL[target].value, None, self._now()  # fmt: skip
            return await self._finish(run_id, publication_id, RunStatus.SUCCEEDED, plan, None, action="none", warnings=plan.warnings)  # fmt: skip
        doc = await self._with_cover(document.model_copy(update={"slug": plan.slug or document.slug}), run_id, plan.warnings)  # fmt: skip
        try:  # the last checks (and the daily slot) come before any change to the CMS
            await self._begin_submit(publication_id, article_id, target, limit)
        except _LimitReached as exc:
            return await self._deferred(run_id, publication_id, plan, str(exc))
        except _Blocked as exc:
            return await self._blocked(run_id, publication_id, plan, str(exc))
        terms = plan.terms or TermResolution(None, ())
        if self._settings.wordpress_create_missing_terms and (terms.missing_category or terms.missing_tags):  # fmt: skip
            terms = await self._create_terms(publisher, publication_id, run_id, doc)
        payload = publisher.build_payload(doc, status=target, terms=terms, marker=snap.marker)
        try:
            post = await self._write(publisher, publication_id, run_id, snap.marker, plan.post, doc, terms, target)  # fmt: skip
        except CMSError as exc:
            return await self._finish(run_id, publication_id, RunStatus.FAILED, plan, str(exc))
        problems, warnings = publisher.verify(post, payload, status=target, marker=snap.marker)
        if problems:
            await self._store_post(publication_id, post)
            return await self._finish(run_id, publication_id, RunStatus.FAILED, plan, "verification failed: " + "; ".join(problems))  # fmt: skip
        await self._record_success(run_id, publication_id, article_id, post, doc, terms, target, plan.warnings + warnings)  # fmt: skip
        return await self._finish(run_id, publication_id, RunStatus.SUCCEEDED, plan, None, status=_FINAL[target], action=plan.report.action, warnings=plan.warnings + warnings)  # fmt: skip

    async def _with_cover(self, doc: RenderedDocument, run_id: int, warnings: list[str]) -> RenderedDocument:  # fmt: skip
        """The document plus the metadata of this version's cover picture, obtained once
        and stored. Without a cover service, or when a cover can't be had, the document is
        returned untouched, a warning is recorded and the post is published without one."""
        if self._covers is None or not self._covers.enabled:
            return doc
        cover = await self._covers.cover(doc, run_id=run_id)
        if cover is None:
            warnings.append("no cover image was available: the post is published without one")
            return doc
        return doc.model_copy(update={"cover": cover})

    async def _write(self, publisher: CMSPublisher, publication_id: int, run_id: int, marker: str, post: CMSPost | None, doc: RenderedDocument, terms: TermResolution, target: TargetStatus) -> CMSPost:  # fmt: skip
        """Create or update the post; a public target goes through a verified draft first
        (PUBLISH_DRAFT_FIRST) unless the post is public already."""
        payload = publisher.build_payload(doc, status=target, terms=terms, marker=marker)
        if target is TargetStatus.PUBLISH and self._settings.publish_draft_first and (post is None or post.status is not CMSPostStatus.PUBLISHED):  # fmt: skip
            draft = publisher.build_payload(doc, status=TargetStatus.DRAFT, terms=terms, marker=marker)  # fmt: skip
            post = await self._put(publisher, publication_id, run_id, marker, post, draft, AttemptAction.CREATE if post is None else AttemptAction.UPDATE)  # fmt: skip
            problems, _ = publisher.verify(post, draft, status=TargetStatus.DRAFT, marker=marker)
            if problems:
                raise _VerificationError("the draft didn't verify: " + "; ".join(problems))
            return await self._put(publisher, publication_id, run_id, marker, post, payload, AttemptAction.PUBLISH)  # fmt: skip
        if post is None:
            action = AttemptAction.CREATE
        elif target is TargetStatus.PUBLISH and post.status is not CMSPostStatus.PUBLISHED:
            action = AttemptAction.PUBLISH
        else:
            action = AttemptAction.UPDATE
        return await self._put(publisher, publication_id, run_id, marker, post, payload, action)

    async def _put(self, publisher: CMSPublisher, publication_id: int, run_id: int, marker: str, post: CMSPost | None, payload: dict[str, Any], action: AttemptAction) -> CMSPost:  # fmt: skip
        payload_hash = hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()
        if post is not None:
            attempt = await self._attempt(publication_id, run_id, action, payload_hash)
            try:
                updated = await publisher.update_post(post.external_id, payload)
            except CMSError as exc:
                await self._attempt_done(attempt, AttemptOutcome.UNKNOWN if exc.outcome_unknown else AttemptOutcome.FAILED, exc.status, post.external_id, str(exc))  # fmt: skip
                raise
            await self._attempt_done(attempt, AttemptOutcome.SUCCEEDED, 200, updated.external_id, None)  # fmt: skip
            await self._store_post(publication_id, updated)
            return updated
        # A creation is never retried blindly: after an unknown outcome the post is looked up
        # (by slug and marker) and adopted if the lost request did create it.
        tries = 1 + self._settings.cms_max_retries
        for n in range(tries):
            attempt = await self._attempt(publication_id, run_id, AttemptAction.CREATE, payload_hash)  # fmt: skip
            try:
                created = await publisher.create_post(payload)
            except CMSError as exc:
                unknown = exc.outcome_unknown
                await self._attempt_done(attempt, AttemptOutcome.UNKNOWN if unknown else AttemptOutcome.FAILED, exc.status, None, str(exc))  # fmt: skip
                if not (unknown or isinstance(exc, CMSTransientError)):
                    raise
                found = await self._reconcile(publisher, publication_id, run_id, marker, str(payload["slug"]))  # fmt: skip
                if found is not None:
                    return found
                if n == tries - 1:
                    raise
                await self._sleep(min(2.0**n, 30.0))
                continue
            await self._attempt_done(attempt, AttemptOutcome.SUCCEEDED, 201, created.external_id, None)  # fmt: skip
            await self._store_post(publication_id, created)
            return created
        raise AssertionError("unreachable")  # pragma: no cover

    async def _reconcile(self, publisher: CMSPublisher, publication_id: int, run_id: int, marker: str, slug: str) -> CMSPost | None:  # fmt: skip
        attempt = await self._attempt(publication_id, run_id, AttemptAction.RECONCILE, None)
        try:
            candidates = await publisher.find_posts(slug=slug)
            if not any(publisher.owns(p, marker) for p in candidates):
                candidates += await publisher.find_posts(marker=marker)
        except CMSError as exc:
            await self._attempt_done(attempt, AttemptOutcome.FAILED, exc.status, None, f"lookup failed, so the creation isn't retried: {exc}")  # fmt: skip
            raise
        found = next((p for p in candidates if publisher.owns(p, marker)), None)
        await self._attempt_done(attempt, AttemptOutcome.SUCCEEDED, 200, found.external_id if found else None, None if found else "no post with this publication's marker: safe to create")  # fmt: skip
        if found is not None:
            await self._store_post(publication_id, found)
            log.info("publishing.reconciled", publication_id=publication_id, external_id=found.external_id)  # fmt: skip
        return found

    async def _create_terms(self, publisher: CMSPublisher, publication_id: int, run_id: int, doc: RenderedDocument) -> TermResolution:  # fmt: skip
        attempt = await self._attempt(publication_id, run_id, AttemptAction.TERMS, None)
        try:
            terms = await publisher.resolve_terms(doc.category, doc.tags, create=True)
        except CMSError as exc:
            await self._attempt_done(attempt, AttemptOutcome.UNKNOWN if exc.outcome_unknown else AttemptOutcome.FAILED, exc.status, None, str(exc))  # fmt: skip
            raise
        await self._attempt_done(attempt, AttemptOutcome.SUCCEEDED, 200, None, "created: " + ", ".join(terms.created) if terms.created else None)  # fmt: skip
        return terms

    async def _begin_submit(self, publication_id: int, article_id: int, target: TargetStatus, limit: int | None) -> None:  # fmt: skip
        """Right before the CMS is changed: the article is still ready, its recommended
        version and report unchanged, and an approval for them live. An automated public
        publication also takes one of today's slots here, atomically (Phase 8)."""
        async with self._sessions() as session, session.begin():
            article = await session.get_one(Article, article_id, with_for_update=True)
            publication = await session.get_one(Publication, publication_id)
            report = await session.get(ArticleQualityReport, article.quality_report_id) if article.quality_report_id else None  # fmt: skip
            live = await live_decision(session, article_id)
            problem = authorization_problem(article, report, live)
            if problem is None and article.recommended_version_id != publication.version_id:
                problem = "the recommended version changed"
            if problem is not None or live is None:
                raise _Blocked(problem or "no approval")
            if limit is not None and target is TargetStatus.PUBLISH:
                reserved, used, day = await reserve_publication_slot(session, publication, limit=limit, settings=self._settings, now=self._now())  # fmt: skip
                if not reserved:
                    raise _LimitReached(f"the daily publishing limit is reached ({used} of {limit} on {day}, {self._settings.scheduler_timezone}): left for a later run")  # fmt: skip
            publication.status, publication.approval_id, publication.updated_at = PublicationStatus.SUBMITTING.value, live.id, self._now()  # fmt: skip

    async def _record_success(self, run_id: int, publication_id: int, article_id: int, post: CMSPost, doc: RenderedDocument, terms: TermResolution, target: TargetStatus, warnings: list[str]) -> None:  # fmt: skip
        now = self._now()
        public = post.status is CMSPostStatus.PUBLISHED
        async with self._sessions() as session, session.begin():
            publication = await session.get_one(Publication, publication_id, with_for_update=True)
            publication.status = (PublicationStatus.PUBLISHED if public else PublicationStatus.DRAFT_CREATED).value  # fmt: skip
            publication.external_id, publication.external_status = post.external_id, post.status.value  # fmt: skip
            publication.url = post.url if public else None
            publication.edit_url = post.edit_url
            publication.content_hash, publication.last_error, publication.updated_at = doc.content_hash, None, now  # fmt: skip
            if public and publication.published_at is None:
                publication.published_at = now
            category = {"name": terms.category.name, "id": terms.category.id} if terms.category else None  # fmt: skip
            image = doc.image.model_dump(mode="json") if doc.image else None
            publication.details = {
                "render_version": doc.render_version,
                "title": doc.title,
                "slug": post.slug or doc.slug,
                "meta_title": doc.meta_title,
                "meta_description": doc.excerpt,
                "seo_plugin": None,  # none configured: meta title/description kept here; the excerpt carries the description
                "primary_keyword": doc.primary_keyword,
                # Provenance: this post was written by a person, not by the agent, and the
                # agent's Gemini gates never ran on it.
                "authored": doc.authored,
                "category": category,
                "tags": [{"name": t.name, "id": t.id} for t in terms.tags],
                "missing_tags": list(terms.missing_tags),
                "terms_created": list(terms.created),
                "links": [link.model_dump(mode="json") for link in doc.links],
                "sources": [s.model_dump(mode="json") for s in doc.sources],
                "faq": len(doc.faq),
                "image_suggestion": image,  # a suggestion the model made; not what was published
                "featured_image": None,
                "cover_image": doc.cover.model_dump(mode="json") if doc.cover else None,
                "word_count": doc.word_count,
                "warnings": warnings,
                "notes": doc.notes,
            }
            await session.execute(
                update(Publication)
                .where(
                    Publication.article_id == article_id,
                    Publication.site == publication.site,
                    Publication.id != publication_id,
                    Publication.external_id == post.external_id,
                    Publication.superseded_by_id.is_(None),
                )
                .values(superseded_by_id=publication_id)
            )
            if public:
                article = await session.get_one(Article, article_id)
                opportunity = await session.get(Opportunity, article.opportunity_id, with_for_update=True)  # fmt: skip
                if opportunity is not None and opportunity.status == OpportunityStatus.APPROVED.value:  # fmt: skip
                    session.add(OpportunityEvent(opportunity_id=opportunity.id, created_at=now, kind=OpportunityEventKind.STATUS_CHANGED.value, from_status=opportunity.status, to_status=OpportunityStatus.USED.value, note=f"article {article_id} published: {post.url}"[:2_000], actor="publishing", run_id=run_id, assessment_id=opportunity.current_assessment_id))  # fmt: skip
                    opportunity.status, opportunity.status_note, opportunity.status_changed_at = OpportunityStatus.USED.value, f"published: {post.url}"[:2_000], now  # fmt: skip
                elif opportunity is not None and opportunity.status != OpportunityStatus.USED.value:
                    publication.details = {**publication.details, "opportunity": f"left {opportunity.status}: only an approved opportunity becomes used"}  # fmt: skip
        log.info("publishing.done", article_id=article_id, publication_id=publication_id, status="published" if public else "draft", external_id=post.external_id)  # fmt: skip

    # ── bookkeeping ──────────────────────────────────────────────────────────

    async def _store_post(self, publication_id: int, post: CMSPost) -> None:
        """Keep the CMS post id as soon as it's known (so no later run creates another)."""
        async with self._sessions() as session, session.begin():
            publication = await session.get_one(Publication, publication_id)
            publication.external_id, publication.external_status, publication.edit_url = post.external_id, post.status.value, post.edit_url  # fmt: skip
            publication.updated_at = self._now()

    async def _attempt(self, publication_id: int, run_id: int, action: AttemptAction, payload_hash: str | None) -> int:  # fmt: skip
        async with self._sessions() as session, session.begin():
            row = PublicationAttempt(publication_id=publication_id, run_id=run_id, action=action.value, outcome=AttemptOutcome.UNKNOWN.value, payload_hash=payload_hash, started_at=self._now())  # fmt: skip
            session.add(row)
            publication = await session.get_one(Publication, publication_id)
            if action is not AttemptAction.RECONCILE:
                publication.attempt_count += 1
            await session.flush()
            return row.id

    async def _attempt_done(self, attempt_id: int, outcome: AttemptOutcome, http_status: int | None, external_id: str | None, error: str | None) -> None:  # fmt: skip
        async with self._sessions() as session, session.begin():
            row = await session.get_one(PublicationAttempt, attempt_id)
            row.outcome, row.http_status, row.external_id, row.finished_at = outcome.value, http_status, external_id, self._now()  # fmt: skip
            row.error = error[:2_000] if error else None

    async def _blocked(self, run_id: int, publication_id: int, plan: _Plan | None, reason: str) -> PublishOutcome:  # fmt: skip
        async with self._sessions() as session, session.begin():
            publication = await session.get_one(Publication, publication_id)
            publication.status, publication.last_error, publication.updated_at = PublicationStatus.BLOCKED.value, f"blocked: {reason}"[:2_000], self._now()  # fmt: skip
        return await self._finish(run_id, publication_id, RunStatus.FAILED, plan, f"blocked: {reason}", mark_failed=False)  # fmt: skip

    async def _deferred(self, run_id: int, publication_id: int, plan: _Plan | None, reason: str) -> PublishOutcome:  # fmt: skip
        """Nothing was sent: the daily allowance is used up. The publication waits for a
        later run (it isn't a failure)."""
        async with self._sessions() as session, session.begin():
            publication = await session.get_one(Publication, publication_id)
            publication.status, publication.last_error, publication.updated_at = PublicationStatus.CANCELLED.value, reason[:2_000], self._now()  # fmt: skip
        log.info("publishing.deferred_daily_limit", publication_id=publication_id, reason=reason)
        outcome = await self._finish(run_id, publication_id, RunStatus.SUCCEEDED, plan, None, action="deferred_daily_limit", note=reason)  # fmt: skip
        return dataclasses.replace(outcome, error=reason)

    async def _fail(self, publication_id: int, error: str) -> None:
        async with self._sessions() as session, session.begin():
            publication = await session.get_one(Publication, publication_id)
            # The post's own state stays in external_status / url; this records the request.
            publication.status, publication.last_error, publication.updated_at = PublicationStatus.FAILED.value, error[:2_000], self._now()  # fmt: skip

    async def _finish(self, run_id: int, publication_id: int, run_status: RunStatus, plan: _Plan | None, error: str | None, *, status: PublicationStatus | None = None, action: str | None = None, warnings: list[str] | None = None, mark_failed: bool = True, note: str | None = None) -> PublishOutcome:  # fmt: skip
        if error is not None and mark_failed:
            await self._fail(publication_id, error)
        async with self._sessions() as session:
            publication = await session.get_one(Publication, publication_id)
            outcome = PublishOutcome(publication.article_id, publication_id, run_id, run_status, status or PublicationStatus(publication.status), action, publication.external_id, publication.url, error, warnings or [])  # fmt: skip
        summary = {"publication_id": publication_id, "status": outcome.status.value, "action": action, "external_id": outcome.external_id, "url": outcome.url}  # fmt: skip
        if note:
            summary["note"] = note
        await finish_run(self._sessions, run_id, status=run_status, now=self._now(), error=error, summary=summary)  # fmt: skip
        log.info("publishing.run_finished", run_id=run_id, publication_id=publication_id, status=run_status.value, publication_status=outcome.status.value, error=error)  # fmt: skip
        return outcome

    async def _active_publication(self, session: AsyncSession, article_id: int) -> Publication | None:  # fmt: skip
        row: Publication | None = await session.scalar(
            select(Publication)
            .join(Run, Run.id == Publication.run_id)
            .where(
                Publication.article_id == article_id,
                Publication.status.in_([s.value for s in IN_FLIGHT_PUBLICATION]),
                Run.status.in_([s.value for s in ACTIVE_RUN_STATUSES]),
                Run.kind == RUN_KIND,
            )
            .order_by(Publication.id.desc())
            .limit(1)
        )
        return row

    # ── inputs ───────────────────────────────────────────────────────────────

    async def _snapshot(self, session: AsyncSession, article_id: int) -> _Snapshot:
        article = await session.get(Article, article_id)
        if article is None:
            raise ArticleNotFoundError(f"Unknown article {article_id}")
        version = await session.get(ArticleVersion, article.recommended_version_id) if article.recommended_version_id else None  # fmt: skip
        if version is not None and (version.article_id != article_id or version.kind not in (VersionKind.FINAL.value, VersionKind.REVISION.value)):  # fmt: skip
            version = None
        report = await session.get(ArticleQualityReport, article.quality_report_id) if article.quality_report_id else None  # fmt: skip
        seo = None
        if report is not None and report.seo_step_id:
            step = await session.get(ArticleStepRun, report.seo_step_id)
            seo = (
                SEOReport.model_validate(step.output) if step is not None and step.output else None
            )
        sources: dict[str, tuple[str | None, str]] = {}
        if version is not None:
            rows = await session.execute(select(ArticleSource.label, ArticleSource.title, ArticleSource.url).join(ArticleCitation, ArticleCitation.source_id == ArticleSource.id).where(ArticleCitation.version_id == version.id).distinct())  # fmt: skip
            sources = {label: (title, url) for label, title, url in rows}
        external = set(await session.scalars(select(ArticleSource.url).where(ArticleSource.step_id == article.research_step_id))) if article.research_step_id else set()  # fmt: skip
        internal: set[str] = set()
        profile = await session.get(CompanyProfileVersion, article.company_profile_id)
        website = profile.to_profile().website if profile is not None else None
        if website:
            domain = domain_of(str(website))
            urls = await session.scalars(select(ContentItem.url).join(Competitor, Competitor.id == ContentItem.competitor_id).where(ContentItem.status == "active", Competitor.active.is_(True)))  # fmt: skip
            internal = {
                u for u in urls if (h := domain_of(u)) == domain or h.endswith("." + domain)
            }
        site = self._cms.site or ""
        key = idempotency_key(article_id, version.id, self._cms.name, site) if version is not None and site else None  # fmt: skip
        publication = await session.scalar(select(Publication).where(Publication.idempotency_key == key)) if key else None  # fmt: skip
        lineage = await session.scalar(select(Publication).where(Publication.article_id == article_id, Publication.cms == self._cms.name, Publication.site == site, Publication.external_id.is_not(None)).order_by(Publication.id.desc()).limit(1))  # fmt: skip
        existing_marker = publication or lineage or await session.scalar(select(Publication).where(Publication.article_id == article_id, Publication.site == site).order_by(Publication.id.desc()).limit(1))  # fmt: skip
        opportunity = await session.get(Opportunity, article.opportunity_id)
        return _Snapshot(
            article=article,
            version=version,
            content=ArticleContent.model_validate(version.content) if version is not None else None,
            report=report,
            seo=seo,
            sources=sources,
            allowed_internal=internal,
            allowed_external=external,
            live=await live_decision(session, article_id),
            publication=publication,
            lineage=lineage,
            marker=existing_marker.marker if existing_marker is not None else uuid.uuid4().hex,
            key=key,
            opportunity_title=opportunity.title if opportunity is not None else None,
        )

    def _render(self, snap: _Snapshot) -> RenderedDocument | None:
        if snap.content is None:
            return None
        doc = render_article(snap.content, sources=snap.sources, seo=snap.seo.package if snap.seo else None, allowed_internal=snap.allowed_internal, allowed_external=snap.allowed_external)  # fmt: skip
        # Provenance the target may show (a pull request description): no secret, no id
        # the reader can't use.
        # The byline, the closing line, the call to action and the cover come from
        # configuration, not from the article: without them in the hash, changing one leaves
        # every published post showing the old one, and a republish reports nothing to do.
        content_hash = hashlib.sha256(f"{doc.content_hash}:{self._cms.presentation}".encode()).hexdigest()  # fmt: skip
        # An authored report has no combined score (the components that carry it need
        # Gemini): the pull request says who wrote it instead of showing a score of zero.
        authored = bool(snap.report is not None and snap.report.authored)
        score = snap.report.overall_score if snap.report is not None and not authored else None
        return doc.model_copy(update={"article_id": snap.article.id, "version_id": snap.version.id if snap.version else None, "content_type": snap.article.content_type, "authored": authored, "quality_score": score, "opportunity_title": snap.opportunity_title, "content_hash": content_hash})  # fmt: skip

    # ── preflight ────────────────────────────────────────────────────────────

    async def _preflight(self, snap: _Snapshot, doc: RenderedDocument | None, target: TargetStatus, publisher: CMSPublisher | None, *, run_id: int | None = None) -> _Plan:  # fmt: skip
        checks: list[PreflightCheck] = []

        def add(name: str, passed: bool, detail: str, *, blocking: bool = True) -> None:
            checks.append(PreflightCheck(name=name, passed=passed, detail=detail, blocking=blocking))  # fmt: skip

        article, version, report, s = snap.article, snap.version, snap.report, self._settings
        add("article", article.status == ArticleStatus.READY.value, f"article {article.id} is {article.status}" + ("" if article.status == ArticleStatus.READY.value else ": only ready articles are published"))  # fmt: skip
        add("version", version is not None, f"recommended version {version.id} ({version.kind} v{version.number})" if version else "no validated recommended version")  # fmt: skip
        if report is None or version is None:
            add("quality", False, "no quality report for the recommended version")
        else:
            failed = [g["name"] for g in report.gates if not g.get("passed")]
            ok = report.passed and report.version_id == version.id and not failed and not readiness_problems(article, report)  # fmt: skip
            score = "written by a person: no agent score" if report.authored else f"{report.overall_score:.1f}/100"  # fmt: skip
            add("quality", ok, f"report #{report.id}: {score}, every gate that ran passes" if ok else f"report #{report.id} fails: {', '.join(failed) or '; '.join(readiness_problems(article, report)) or 'it is for another version'}")  # fmt: skip
        if report is not None and report.authored:
            add("provenance", True, "written by a person and imported: the agent checked its length, citations, structure and the site's MDX contract, and did not fact-check, originality-check or score it", blocking=False)  # fmt: skip
        problem = authorization_problem(article, report, snap.live)
        auto = problem is not None and s.publish_auto_approve and not readiness_problems(article, report) and not (snap.live is not None and snap.live.decision == "rejected" and snap.live.invalidated_at is None and snap.live.version_id == article.recommended_version_id and snap.live.quality_report_id == article.quality_report_id)  # fmt: skip
        if auto:
            add("approval", True, "not approved yet: PUBLISH_AUTO_APPROVE records an automatic approval when publishing")  # fmt: skip
        elif problem is None and snap.live is not None:
            add("approval", True, f"approval #{snap.live.id} by {snap.live.approver} ({snap.live.method}) for version {snap.live.version_id} and report #{snap.live.quality_report_id}")  # fmt: skip
        else:
            add("approval", False, problem or "no approval")
        if snap.content is None or doc is None:
            add("content", False, "no content to publish")
        else:
            problems = structural_problems(snap.content, min_words=s.article_min_words, labels=set(snap.sources))  # fmt: skip
            add("content", not problems, f"{doc.word_count} words, {len(doc.body_html):,} characters of HTML" if not problems else "; ".join(problems))  # fmt: skip
            add("citations", not doc.unknown_citations, f"{len(doc.sources)} cited source(s), every one stored" if not doc.unknown_citations else f"citation(s) without a stored source: {', '.join(doc.unknown_citations)}")  # fmt: skip
            rejected = [n for n in doc.notes if "not a validated target" in n]
            placed = sum(1 for link in doc.links if link.placed)
            add("links", not rejected, f"{placed} of {len(doc.links)} validated link(s) placed" + (f"; {len(rejected)} left out" if rejected else ""), blocking=False)  # fmt: skip
        seo = snap.seo.package if snap.seo else None
        missing = [name for name, value in (("primary keyword", seo.primary_keyword if seo else ""), ("meta title", seo.meta_title if seo else ""), ("meta description", seo.meta_description if seo else ""), ("slug", seo.slug if seo else "")) if not value.strip()]  # fmt: skip
        add("seo", not missing, f"keyword '{seo.primary_keyword}', meta title and description, slug '{doc.slug if doc else seo.slug}'" if seo and not missing else f"missing: {', '.join(missing)}")  # fmt: skip
        if target is TargetStatus.PUBLISH and not s.publish_allow_direct_publish:
            add("target_status", False, "making a post public needs PUBLISH_ALLOW_DIRECT_PUBLISH=true")  # fmt: skip
        else:
            add("target_status", True, f"leaves the post as {target.value}" + (" (a verified draft first)" if target is TargetStatus.PUBLISH and s.publish_draft_first else ""))  # fmt: skip
        add("cms_config", self._cms.configured, f"{self._cms.name} at {self._cms.site}" if self._cms.configured else self._cms.configuration_hint)  # fmt: skip
        pub = snap.publication
        plan = _Plan(report=self._report(snap, checks, target, "create"))
        if pub is not None and pub.status in IN_FLIGHT_PUBLICATION and pub.run_id not in (None, run_id):  # fmt: skip
            async with self._sessions() as session:
                other = await session.get(Run, pub.run_id)
            if other is not None and other.status in {r.value for r in ACTIVE_RUN_STATUSES}:
                add(
                    "idempotency", False, f"publication #{pub.id} is in progress (run {pub.run_id})"
                )
        if publisher is None or doc is None:
            if publisher is None:
                add("cms", False, "not checked: the CMS isn't configured")
            plan.report = self._report(snap, checks, target, "create")
            return plan
        try:
            await self._cms_checks(snap, doc, target, publisher, plan, add)
        except CMSError as exc:  # the CMS couldn't be read: nothing is changed
            add("cms", False, f"the CMS couldn't be checked: {exc}")
            plan.transient = isinstance(exc, CMSTransientError)
            plan.report = self._report(snap, checks, target, "create")
            return plan
        plan.report = self._report(snap, checks, target, plan.action, external_id=plan.post.external_id if plan.post else None)  # fmt: skip
        return plan

    async def _cms_checks(self, snap: _Snapshot, doc: RenderedDocument, target: TargetStatus, publisher: CMSPublisher, plan: _Plan, add: Callable[..., None]) -> None:  # fmt: skip
        """The checks that read the CMS: the site and user, the post, the slug, the terms,
        idempotency. They never change anything."""
        s, pub = self._settings, snap.publication
        check = await publisher.check(need_publish=target is TargetStatus.PUBLISH)
        cms_ok = check.reachable and check.authenticated and check.can_create and (check.can_publish or target is not TargetStatus.PUBLISH)  # fmt: skip
        add("cms", cms_ok, f"{check.site_name or publisher.site}: {check.detail}")
        if not cms_ok:
            return
        post: CMSPost | None = None
        known = (pub.external_id if pub and pub.external_id else None) or (snap.lineage.external_id if snap.lineage else None)  # fmt: skip
        slug = doc.slug
        if known:
            post = await publisher.get_post(known)
            if post is None or post.status is CMSPostStatus.TRASH:
                add("post", False, f"{publisher.name} post {known} (this article's earlier publication) no longer exists or is in the trash: restore it there first")  # fmt: skip
                post = None
            elif not publisher.owns(post, snap.marker):
                add("post", False, f"{publisher.name} post {known} no longer carries this system's marker (edited outside?): it won't be overwritten")  # fmt: skip
                post = None
            elif post.status is CMSPostStatus.PUBLISHED and target is not TargetStatus.PUBLISH:
                add("post", False, f"post {known} is public: updating a public post needs the publish target (and PUBLISH_ALLOW_DIRECT_PUBLISH); it is never taken back to {target.value}")  # fmt: skip
            else:
                if post.status is CMSPostStatus.PUBLISHED and post.slug and post.slug != doc.slug:
                    slug = post.slug
                    plan.warnings.append(f"post {post.external_id} is public at '{post.slug}': it keeps that slug (not '{doc.slug}')")  # fmt: skip
                add("post", True, f"updates this article's post {post.external_id} ({post.status.value})")  # fmt: skip
        found = await publisher.find_posts(slug=slug)
        others = [p for p in found if post is None or p.external_id != post.external_id]
        ours = [p for p in others if publisher.owns(p, snap.marker)]
        foreign = [p for p in others if not publisher.owns(p, snap.marker)]
        if post is None and not known and not ours:
            ours = [p for p in await publisher.find_posts(marker=snap.marker) if publisher.owns(p, snap.marker)]  # fmt: skip
        if post is None and not known and ours:
            post = ours[0]
            add("post", True, f"found this article's post {post.external_id} (created before an answer was lost): it is reused, not duplicated")  # fmt: skip
        if foreign:
            add("slug", False, f"slug '{slug}' is used by {publisher.name} post(s) {', '.join(p.external_id for p in foreign)} not created by this system: rename one of them (it is never overwritten)")  # fmt: skip
        else:
            add("slug", True, f"slug '{slug}' is free" if post is None else f"slug '{slug}' is this article's")  # fmt: skip
        terms = await publisher.resolve_terms(doc.category, doc.tags, create=False, content_type=doc.content_type)  # fmt: skip
        create = s.wordpress_create_missing_terms
        if terms.missing_category:
            add("category", create, f"category '{terms.missing_category}' isn't in {publisher.name}" + (": it will be created" if create else ": create it there, or set WORDPRESS_CREATE_MISSING_TERMS=true"))  # fmt: skip
        else:
            add("category", True, f"category '{terms.category.name}' (id {terms.category.id})" if terms.category else f"no category: {publisher.name}'s default")  # fmt: skip
        if terms.missing_tags:
            add("tags", create, f"tag(s) not in {publisher.name}: {', '.join(terms.missing_tags)}" + (" (they will be created)" if create else " (left out; WORDPRESS_CREATE_MISSING_TERMS=true creates them)"), blocking=False)  # fmt: skip
        else:
            add("tags", True, f"{len(terms.tags)} tag(s) found")
        plan.warnings += list(terms.notes)
        if post is None:
            action = "create"
        elif target is TargetStatus.PUBLISH and post.status is not CMSPostStatus.PUBLISHED:
            action = "publish"
        else:
            action = "update"
        if pub is not None and post is not None and pub.content_hash is not None and pub.external_id == post.external_id and pub.content_hash == doc.content_hash and post.status is _EXPECTED[target]:  # fmt: skip
            action = "none"  # the post already holds this version, as asked
        add("idempotency", True, {"none": f"publication #{pub.id if pub else '?'} already shows this version as {target.value}: nothing to change", "create": "no post yet: one will be created", "update": f"post {post.external_id if post else '?'} will be updated (never duplicated)", "publish": f"post {post.external_id if post else '?'} will be updated and made public"}[action])  # fmt: skip
        plan.terms, plan.post, plan.slug, plan.action = terms, post, slug, action

    def _report(self, snap: _Snapshot, checks: list[PreflightCheck], target: TargetStatus, action: str, *, external_id: str | None = None) -> PreflightReport:  # fmt: skip
        return PreflightReport(
            article_id=snap.article.id,
            ready=all(c.passed for c in checks if c.blocking),
            action=action,
            target_status=target,
            cms=self._cms.name,
            site=self._cms.site,
            version_id=snap.version.id if snap.version else None,
            quality_report_id=snap.report.id if snap.report else None,
            approval_id=snap.live.id if snap.live else None,
            publication_id=snap.publication.id if snap.publication else None,
            external_id=external_id,
            checks=list(checks),
        )


class _VerificationError(CMSError):
    pass


__all__ = [
    "RUN_KIND",
    "ApprovalRequiredError",
    "PublishOutcome",
    "PublishRequestResult",
    "PublishingConflictError",
    "PublishingService",
    "idempotency_key",
]
