"""Raw layer: HTML exactly as fetched, kept for every captured content version.

Stored gzip-compressed and only when the extracted content changed, so storage grows
with real changes rather than with scan frequency. Lets later phases re-extract pages
with a better extractor without re-crawling.
"""

from datetime import datetime

from sqlalchemy import BigInteger, ForeignKey, Identity, Index, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class RawDocument(Base):
    __tablename__ = "raw_documents"
    __table_args__ = (Index("ix_raw_documents_competitor_id_url", "competitor_id", "url"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    competitor_id: Mapped[int] = mapped_column(ForeignKey("competitors.id"))
    url: Mapped[str] = mapped_column(Text)
    final_url: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime]
    http_status: Mapped[int]
    content_type: Mapped[str | None] = mapped_column(Text)
    etag: Mapped[str | None] = mapped_column(Text)
    last_modified: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))  # of the uncompressed body
    size_bytes: Mapped[int]  # uncompressed
    compression: Mapped[str] = mapped_column(String(16), default="gzip")
    body: Mapped[bytes] = mapped_column(LargeBinary)
