"""Cover images for published posts: one picture per article version, generated once.

    the rendered document (title, keyword) + the article's audience + the company profile
      → app.prompts.cover_image.render (our rules; the article only as data)
      → BudgetedLLM.image (GEMINI_IMAGE_MODEL, recorded as LLMPurpose.COVER_IMAGE)
      → one article_covers row, unique per (article, version)
      → RenderedCover on the document (metadata) and the bytes for the adapter

**Generated once.** The row is keyed by (article, version), so a retry after a lost answer,
a second publication of the same version, or a follow-up pull request all reuse the stored
picture: the image model is called exactly once per version. The GitHub adapter asks for the
bytes only when the file is missing on the post's branch, so nothing is even read twice.

**A cover never stops a publication.** Every failure here — no API key, a budget stop, a
refusal, an unusable answer, a database error — is logged as a warning and answered with
None. The post is then published exactly as it would have been with the feature off.

Off unless ``PUBLISH_COVER_IMAGES`` is set.
"""

import hashlib
from collections.abc import Callable
from datetime import datetime

import structlog
from sqlalchemy import Select, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.config import Settings
from app.core.timeutils import utcnow
from app.db.models import Article, ArticleCover, CompanyProfileVersion
from app.db.session import SessionFactory
from app.domain.analysis import LLMPurpose
from app.domain.company import CompanyProfile
from app.domain.publishing import RenderedCover, RenderedDocument
from app.llm import ImageRequest, LazyLLM, LLMError
from app.prompts import cover_image as prompt
from app.services.company import latest_company_profile
from app.services.llm_usage import BudgetedLLM

log = structlog.get_logger(__name__)

# What the site's repository may hold. A model that answers with anything else is refused.
EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
MAX_COVER_BYTES = 8_000_000  # a cover is committed to a git repository: keep it sane
MIN_COVER_BYTES = 100


def extension(mime: str) -> str | None:
    return EXTENSIONS.get(mime.split(";")[0].strip().lower())


class CoverService:
    def __init__(
        self,
        sessions: SessionFactory,
        settings: Settings,
        llm: LazyLLM,
        *,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        self._sessions = sessions
        self._settings = settings
        self._llm = llm
        self._now = now

    @property
    def enabled(self) -> bool:
        return self._settings.publish_cover_images

    # ── what publishing asks for ─────────────────────────────────────────────

    async def cover(self, document: RenderedDocument, *, run_id: int | None = None) -> RenderedCover | None:  # fmt: skip
        """The cover of this article version: the stored one, or one generated and stored
        now. None (with a warning) whenever a cover can't be had; publishing carries on."""
        if not self.enabled:
            return None
        article_id, version_id = document.article_id, document.version_id
        if article_id is None or version_id is None:
            return None
        try:
            stored = await self._stored(article_id, version_id)
            if stored is not None:
                return stored
            return await self._generate(document, article_id, version_id, run_id)
        except (LLMError, SQLAlchemyError, ValueError) as exc:
            log.warning("cover.unavailable", article_id=article_id, version_id=version_id, error=f"{type(exc).__name__}: {exc}")  # fmt: skip
            return None

    async def image(self, article_id: int, version_id: int) -> tuple[RenderedCover, bytes] | None:
        """The stored picture itself, for the publishing adapter (``CoverSource``). Never
        generates: a cover that doesn't exist yet simply isn't committed."""
        async with self._sessions() as session:
            row = await session.scalar(self._query(article_id, version_id))
        if row is None:
            log.warning("cover.not_stored", article_id=article_id, version_id=version_id)
            return None
        return _rendered(row), row.data

    # ── generation ───────────────────────────────────────────────────────────

    async def _generate(self, document: RenderedDocument, article_id: int, version_id: int, run_id: int | None) -> RenderedCover | None:  # fmt: skip
        if not self._llm.configured:
            log.warning("cover.no_llm", article_id=article_id)
            return None
        audience, profile = await self._context(article_id)
        text = prompt.render(
            title=document.title,
            primary_keyword=document.primary_keyword,
            audience=audience,
            positioning=profile.positioning if profile else None,
            tone=profile.tone if profile else None,
        )
        budgeted = BudgetedLLM(self._llm.get(), self._sessions, self._settings, run_id=run_id, now=self._now)  # fmt: skip
        request = ImageRequest(prompt=text, model=self._settings.gemini_image_model, aspect_ratio=prompt.ASPECT_RATIO)  # fmt: skip
        response = await budgeted.image(request, purpose=LLMPurpose.COVER_IMAGE, prompt_version=prompt.VERSION)  # fmt: skip
        suffix = extension(response.mime_type)
        if suffix is None:
            raise ValueError(f"the image model answered with {response.mime_type!r}, which the site can't serve")  # fmt: skip
        if not MIN_COVER_BYTES <= len(response.data) <= MAX_COVER_BYTES:
            raise ValueError(f"the generated cover is {len(response.data):,} bytes: outside {MIN_COVER_BYTES}-{MAX_COVER_BYTES:,}")  # fmt: skip
        row = ArticleCover(
            article_id=article_id,
            version_id=version_id,
            filename=f"{document.slug}{suffix}",
            mime=response.mime_type,
            width=response.width,
            height=response.height,
            data=response.data,
            byte_size=len(response.data),
            sha256=hashlib.sha256(response.data).hexdigest(),
            alt=prompt.alt_text(document.title, document.primary_keyword),
            prompt=text,
            prompt_version=prompt.VERSION,
            model=response.model[:100],
            created_at=self._now(),
        )
        try:
            async with self._sessions() as session, session.begin():
                session.add(row)
        except IntegrityError:  # another run stored one first: that one is the cover
            stored = await self._stored(article_id, version_id)
            if stored is None:
                raise
            return stored
        log.info("cover.generated", article_id=article_id, version_id=version_id, model=row.model, mime=row.mime, width=row.width, height=row.height, bytes=row.byte_size)  # fmt: skip
        return _rendered(row)

    async def _context(self, article_id: int) -> tuple[str | None, CompanyProfile | None]:
        async with self._sessions() as session:
            article = await session.get(Article, article_id)
            version = await latest_company_profile(session)
            if article is not None and article.company_profile_id:
                version = await session.get(CompanyProfileVersion, article.company_profile_id) or version  # fmt: skip
            return (
                article.target_audience if article is not None else None,
                version.to_profile() if version is not None else None,
            )

    async def _stored(self, article_id: int, version_id: int) -> RenderedCover | None:
        async with self._sessions() as session:
            row = await session.scalar(self._query(article_id, version_id))
        return _rendered(row) if row is not None else None

    @staticmethod
    def _query(article_id: int, version_id: int) -> Select[tuple[ArticleCover]]:
        return select(ArticleCover).where(
            ArticleCover.article_id == article_id, ArticleCover.version_id == version_id
        )


def _rendered(row: ArticleCover) -> RenderedCover:
    return RenderedCover(filename=row.filename, mime=row.mime, alt=row.alt, width=row.width, height=row.height, sha256=row.sha256)  # fmt: skip


__all__ = ["EXTENSIONS", "MAX_COVER_BYTES", "CoverService", "extension"]
