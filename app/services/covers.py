"""Cover images for published posts: one picture per article version, obtained once.

Two sources, chosen by ``COVER_IMAGE_SOURCE``; everything after them is identical.

    gemini (the default)
      the rendered document (title, keyword) + the article's audience + the company profile
        → app.prompts.cover_image.render (our rules; the article only as data)
        → BudgetedLLM.image (GEMINI_IMAGE_MODEL, recorded as LLMPurpose.COVER_IMAGE)

    pexels
      the same subject → app.images.pexels.search_queries (fixed rules, no model call)
        → api.pexels.com/v1/search, query by query, landscape first
        → the photo closest to 16:9 that no earlier cover used (article_covers)
        → one sized variant from images.pexels.com (never the original)

    → one article_covers row, unique per (article, version), with its provenance
    → RenderedCover on the document (metadata and credit) and the bytes for the adapter

**Obtained once.** The row is keyed by (article, version), so a retry after a lost answer,
a second publication of the same version, or a follow-up pull request all reuse the stored
picture: the image model is called, or Pexels searched, exactly once per version. The GitHub
adapter asks for the bytes only when the file is missing on the post's branch, so nothing is
even read twice.

**Never the same photo twice.** A stock photo's id is recorded, and a photo any earlier
cover used is skipped when the next post is published.

**A cover never stops a publication.** Every failure here — no API key, a budget stop, a
refusal, an unusable answer, no photo for any query, a database error — is logged as a
warning and answered with None. The post is then published exactly as it would have been
with the feature off.

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
from app.domain.publishing import CoverImageSource, RenderedCover, RenderedDocument
from app.images import dimensions, pexels
from app.llm import ImageRequest, LazyLLM, LLMError
from app.prompts import cover_image as prompt
from app.services.company import latest_company_profile
from app.services.llm_usage import BudgetedLLM

log = structlog.get_logger(__name__)

# What the site's repository may hold. A source that answers with anything else is refused.
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

    @property
    def source(self) -> CoverImageSource:
        return CoverImageSource(self._settings.cover_image_source)

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
            if self.source is CoverImageSource.PEXELS:
                return await self._photograph(document, article_id, version_id)
            return await self._generate(document, article_id, version_id, run_id)
        except (LLMError, pexels.PexelsError, SQLAlchemyError, ValueError) as exc:
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

    # ── gemini: one generated illustration ───────────────────────────────────

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
        suffix = self._usable(response.mime_type, response.data, "the image model answered with")
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
            source=CoverImageSource.GEMINI.value,
            created_at=self._now(),
        )
        stored = await self._store(row, article_id, version_id)
        log.info("cover.generated", article_id=article_id, version_id=version_id, model=row.model, mime=row.mime, width=row.width, height=row.height, bytes=row.byte_size)  # fmt: skip
        return stored

    # ── pexels: one stock photo, chosen by fixed rules ───────────────────────

    async def _photograph(self, document: RenderedDocument, article_id: int, version_id: int) -> RenderedCover | None:  # fmt: skip
        """A Pexels photo for this version: the first query that offers a wide, unused
        picture wins. No model is called and the article's words travel only as search
        words. Nothing found, or no key configured, means no cover — never a failure."""
        key = self._settings.pexels_api_key
        if key is None:
            log.warning("cover.no_pexels_key", article_id=article_id)
            return None
        audience, _ = await self._context(article_id)
        queries = pexels.search_queries(primary_keyword=document.primary_keyword, audience=audience, content_type=document.content_type)  # fmt: skip
        used = await self._used_photos()
        client = pexels.PexelsClient(key, timeout=self._settings.cms_request_timeout, max_retries=self._settings.cms_max_retries, user_agent=self._settings.crawler_user_agent, max_bytes=MAX_COVER_BYTES)  # fmt: skip
        try:
            found = await self._search(client, queries, used)
            if found is None:
                log.warning("cover.no_photo", article_id=article_id, queries=len(queries), used=len(used))  # fmt: skip
                return None
            query, photo = found
            url = photo.variant_url
            if url is None:  # choose() only returns photos that have one
                raise pexels.PexelsResponseError(f"photo {photo.id} offers no usable size")
            data, mime = await client.download(url)
        finally:
            await client.aclose()
        suffix = self._usable(mime, data, "Pexels served")
        width, height = dimensions(data)
        row = ArticleCover(
            article_id=article_id,
            version_id=version_id,
            filename=f"{document.slug}{suffix}",
            mime=mime,
            width=width,
            height=height,
            data=data,
            byte_size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            alt=pexels.photo_alt_text(document.title, document.primary_keyword),
            prompt=query,  # what was searched for, not an instruction to anything
            prompt_version=pexels.QUERY_VERSION,
            model=None,  # a photographer took it
            source=CoverImageSource.PEXELS.value,
            source_id=str(photo.id),
            source_url=photo.page_url or None,
            photographer=photo.photographer or None,
            photographer_url=photo.photographer_url or None,
            created_at=self._now(),
        )
        stored = await self._store(row, article_id, version_id)
        log.info("cover.photo", article_id=article_id, version_id=version_id, photo_id=photo.id, query=query, mime=row.mime, width=row.width, height=row.height, bytes=row.byte_size)  # fmt: skip
        return stored

    @staticmethod
    async def _search(client: pexels.PexelsClient, queries: list[str], used: set[int]) -> tuple[str, pexels.PexelsPhoto] | None:  # fmt: skip
        """The queries in order until one offers a photo; None when none of them does."""
        for query in queries:
            photos = await client.search(query)
            photo = pexels.choose(photos, used=used)
            if photo is not None:
                return query, photo
            log.debug("cover.query_empty", query=query, offered=len(photos))
        return None

    async def _used_photos(self) -> set[int]:
        """The Pexels photos earlier covers already used: never the same picture twice.

        Read once, just before searching. Two publications running at the same instant
        could still land on the same photo — deliberately not a unique constraint, which
        would leave one of them with no cover at all rather than a repeated picture."""
        async with self._sessions() as session:
            rows = await session.scalars(
                select(ArticleCover.source_id).where(
                    ArticleCover.source == CoverImageSource.PEXELS.value,
                    ArticleCover.source_id.is_not(None),
                )
            )
        return {int(value) for value in rows if value and value.isdigit()}

    # ── storing ──────────────────────────────────────────────────────────────

    @staticmethod
    def _usable(mime: str, data: bytes, who: str) -> str:
        """The file extension for a picture the site can serve, or a refusal."""
        suffix = extension(mime)
        if suffix is None:
            raise ValueError(f"{who} {mime!r}, which the site can't serve")
        if not MIN_COVER_BYTES <= len(data) <= MAX_COVER_BYTES:
            raise ValueError(f"the cover is {len(data):,} bytes: outside {MIN_COVER_BYTES}-{MAX_COVER_BYTES:,}")  # fmt: skip
        return suffix

    async def _store(self, row: ArticleCover, article_id: int, version_id: int) -> RenderedCover:
        try:
            async with self._sessions() as session, session.begin():
                session.add(row)
        except IntegrityError:  # another run stored one first: that one is the cover
            stored = await self._stored(article_id, version_id)
            if stored is None:
                raise
            return stored
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
    return RenderedCover(filename=row.filename, mime=row.mime, alt=row.alt, width=row.width, height=row.height, sha256=row.sha256, source=row.source, credit=row.photographer, credit_url=row.photographer_url, source_url=row.source_url)  # fmt: skip


__all__ = ["EXTENSIONS", "MAX_COVER_BYTES", "CoverService", "extension"]
