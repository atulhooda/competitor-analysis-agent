"""Your company profile: versioned storage. The newest version is the one in use."""

from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CompanyProfileVersion
from app.domain.company import CompanyProfile, CompanyProfileView

_LOCK_KEY = (72_011 << 32) | 1


async def latest_company_profile(session: AsyncSession) -> CompanyProfileVersion | None:
    row: CompanyProfileVersion | None = await session.scalar(
        select(CompanyProfileVersion).order_by(CompanyProfileVersion.version.desc()).limit(1)
    )
    return row


async def list_company_profiles(session: AsyncSession) -> list[CompanyProfileVersion]:
    rows = await session.scalars(
        select(CompanyProfileVersion).order_by(CompanyProfileVersion.version.desc())
    )
    return list(rows)


async def save_company_profile(
    session: AsyncSession, profile: CompanyProfile, *, source: str, now: datetime
) -> tuple[CompanyProfileVersion, bool]:
    """Store ``profile`` as a new version unless it equals the current one.
    Returns (the current version, whether it was created). Call inside a transaction."""
    await session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _LOCK_KEY})
    latest = await latest_company_profile(session)
    if latest is not None and latest.fingerprint == profile.fingerprint:
        return latest, False
    row = CompanyProfileVersion(
        version=(latest.version + 1) if latest else 1,
        fingerprint=profile.fingerprint,
        scoring_fingerprint=profile.scoring_fingerprint,
        profile=profile.model_dump(mode="json"),
        source=source,
        created_at=now,
    )
    session.add(row)
    await session.flush()
    return row, True


def company_view(row: CompanyProfileVersion) -> CompanyProfileView:
    return CompanyProfileView(
        id=row.id,
        version=row.version,
        fingerprint=row.fingerprint,
        source=row.source,
        created_at=row.created_at,
        profile=row.to_profile(),
    )
