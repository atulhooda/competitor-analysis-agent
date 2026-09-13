"""Operations layer: runs (scans, analyses, reports) with their events, and LLM calls."""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, ForeignKey, Identity, Index, String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.timeutils import utcnow
from app.db.base import Base, one_of
from app.domain.analysis import LLMCallStatus, LLMPurpose
from app.domain.history import RunStatus, RunTrigger


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        one_of("status", RunStatus),
        one_of("trigger", RunTrigger),
        Index("ix_runs_competitor_id_created_at", "competitor_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))
    trigger: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16))
    competitor_id: Mapped[int | None] = mapped_column(ForeignKey("competitors.id"))
    params: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=text("'{}'::jsonb"))
    stats: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=text("'{}'::jsonb"))
    summary: Mapped[dict[str, Any]] = mapped_column(
        default=dict, server_default=text("'{}'::jsonb")
    )
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]


class RunEvent(Base):
    """Audit trail for a run: skipped URLs, errors, and (later) agent and tool calls."""

    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())
    level: Mapped[str] = mapped_column(String(16))
    event: Mapped[str] = mapped_column(String(64))
    url: Mapped[str | None] = mapped_column(Text)
    detail: Mapped[str | None] = mapped_column(Text)
    data: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=text("'{}'::jsonb"))


class LLMCall(Base):
    """One LLM request: what it was for, which model, what it cost, how it ended.

    The ledger behind the per-run and daily token budgets. Prompts and responses are not
    stored here; their validated results live in the analysis layer.
    """

    __tablename__ = "llm_calls"
    __table_args__ = (one_of("purpose", LLMPurpose), one_of("status", LLMCallStatus))

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"), index=True)  # fmt: skip
    created_at: Mapped[datetime] = mapped_column(
        default=utcnow, server_default=func.now(), index=True
    )
    purpose: Mapped[str] = mapped_column(String(32))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(100))
    prompt_version: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    error: Mapped[str | None] = mapped_column(Text)
    items: Mapped[int] = mapped_column(default=1, server_default=text("1"))
    request_chars: Mapped[int]
    input_tokens: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    output_tokens: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    reasoning_tokens: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    cached_input_tokens: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    total_tokens: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    latency_ms: Mapped[int]
    response_id: Mapped[str | None] = mapped_column(Text)
