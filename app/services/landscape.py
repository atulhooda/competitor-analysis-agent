"""Cross-competitor landscape reports: deterministic snapshot → Gemini briefing → grounding.

The model is given only precomputed statistics (never raw pages). Each finding must cite
topics and competitors present in the data; findings citing anything else are dropped.
The report is stored with the exact metrics snapshot it was written from. One report
run at a time (advisory lock); an unchanged snapshot isn't re-sent unless forced.
"""

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import Settings
from app.core.errors import AppError, TransientError
from app.core.timeutils import utcnow
from app.db import analysis_queries
from app.db.locks import landscape_lock
from app.db.models import LandscapeReport, Run
from app.db.session import SessionFactory
from app.domain.analysis import LLMPurpose, Share
from app.domain.history import RunStatus, RunTrigger
from app.domain.intelligence import (
    CompetitorPositioning,
    Landscape,
    LandscapeFinding,
    LandscapeNarrative,
)
from app.llm import LazyLLM, LLMConfigurationError, LLMError, LLMRequest
from app.prompts import landscape as prompt
from app.services.digest import neutralize
from app.services.intelligence import IntelligenceService
from app.services.llm_usage import BudgetedLLM, RunUsage
from app.services.runs import fail_abandoned_runs, finish_run, run_slot_free

log = structlog.get_logger(__name__)

RUN_KIND = "landscape"
_MAX_TOPIC_ROWS = 40


class LandscapeError(AppError):
    pass


class LandscapeAlreadyRunningError(LandscapeError, TransientError):
    pass


@dataclass(frozen=True)
class LandscapeOutcome:
    run_id: int
    status: RunStatus
    report_id: int | None = None
    unchanged: bool = False
    usage: RunUsage | None = None
    error: str | None = None


def _pct(share: float) -> str:
    return f"{share * 100:.0f}%"


def _mix(shares: list[Share]) -> str:
    return ", ".join(f"{s.value} {_pct(s.share)}" for s in shares) or "none"


def render_data(landscape: Landscape) -> str:
    """The snapshot as compact text: the only thing the model sees."""
    b = landscape.basis
    lines = [
        f"Window: the {b.window_days} days since {b.window_start:%Y-%m-%d}, compared with the "
        f"{b.window_days} days before. Growth comparisons use: "
        f"{', '.join(b.compared_competitors) or 'none'}; insufficient history: "
        f"{', '.join(b.insufficient_history) or 'none'}.",
        "",
        "Competitors:",
    ]
    for c in landscape.competitors:
        lines.append(
            f"- {c.competitor} ({c.name}): {c.analyzed_items} analyzed pages; published "
            f"{c.cadence.recent} in the window and {c.cadence.previous} in the previous one "
            f"({c.cadence.per_week}/week); {c.cadence.undated} without a reliable date"
        )
        if c.positioning_statement:
            lines.append(f"  positioning: {c.positioning_statement}")
        if c.top_topics:
            lines.append("  top topics: " + ", ".join(f"{t.topic.slug} {_pct(t.share)} ({t.trend.value})" for t in c.top_topics))  # fmt: skip
        lines.append(f"  formats: {_mix(c.formats)}")
        lines.append(f"  audiences: {_mix(c.audiences)}")
    lines += ["", "Topic coverage (slug | name | items | competitors | recent | previous | trend | items by competitor):"]  # fmt: skip
    for t in landscape.topics[:_MAX_TOPIC_ROWS]:
        per = ", ".join(f"{k} {v}" for k, v in t.by_competitor.items())
        lines.append(f"- {t.topic.slug} | {t.topic.name} | {t.items} | {t.competitors} | {t.recent} | {t.previous} | {t.trend.value} | {per}")  # fmt: skip
    lines += ["", "Rising topics (previous → recent):"]
    lines += [f"- {t.topic.slug}: {t.previous} → {t.recent} ({t.trend.value}; covered by {t.competitors})" for t in landscape.rising] or ["- none"]  # fmt: skip
    lines += ["", "Neglected topics:"]
    lines += [f"- {n.topic.slug} ({n.topic.name}) | {n.reason} | {n.detail}" for n in landscape.neglected] or ["- none"]  # fmt: skip
    lines += ["", f"Formats overall: {_mix(landscape.formats)}", f"Audiences overall: {_mix(landscape.audiences)}", f"Intents overall: {_mix(landscape.intents)}"]  # fmt: skip
    lines += ["", "Mix shifts (previous → recent share):"]
    lines += [f"- {s.dimension} {s.value}: {_pct(s.previous_share)} → {_pct(s.recent_share)} ({s.change:+.1f} pp)" for s in landscape.strategy_shifts] or ["- none"]  # fmt: skip
    lines += ["", "Significant recent changes:"]
    lines += [
        f"- {ch.competitor} | {ch.content_type.value} | {ch.url} | {ch.summary.significance.value} | {ch.summary.summary}"
        for ch in landscape.recent_changes
        if ch.summary is not None
    ] or ["- none"]
    return neutralize("\n".join(lines)).replace("</data", "<\\/data")


def ground(out: prompt.LandscapeOut, landscape: Landscape) -> LandscapeNarrative:
    """Keep only findings whose topic and competitor references exist in the snapshot."""
    topics = {t.topic.slug: t.topic.name for t in landscape.topics}
    for t in [*landscape.rising, *(t for c in landscape.competitors for t in c.top_topics)]:
        topics[t.topic.slug] = t.topic.name
    for n in landscape.neglected:
        topics[n.topic.slug] = n.topic.name
    by_name = {name.casefold(): slug for slug, name in topics.items()}
    competitors = {c.competitor: c.competitor for c in landscape.competitors}
    competitors |= {c.name.casefold(): c.competitor for c in landscape.competitors}
    dropped = 0

    def topic(value: str) -> str | None:
        value = value.strip()
        return value if value in topics else by_name.get(value.casefold())

    def competitor(value: str) -> str | None:
        value = value.strip()
        return competitors.get(value) or competitors.get(value.casefold())

    def findings(values: list[prompt.FindingOut]) -> list[LandscapeFinding]:
        nonlocal dropped
        kept = []
        for value in values:
            topic_refs = [topic(t) for t in value.topics]
            competitor_refs = [competitor(c) for c in value.competitors]
            if None in topic_refs or None in competitor_refs or not value.text:
                dropped += 1
                continue
            kept.append(
                LandscapeFinding(
                    text=value.text,
                    topics=list(dict.fromkeys(t for t in topic_refs if t)),
                    competitors=list(dict.fromkeys(c for c in competitor_refs if c)),
                )
            )
        return kept

    positioning = []
    for entry in out.positioning:
        slug = competitor(entry.competitor)
        if slug is None:
            dropped += 1
            continue
        focus = [t for t in (topic(f) for f in entry.focus) if t]
        positioning.append(CompetitorPositioning(competitor=slug, positioning=entry.positioning, focus=list(dict.fromkeys(focus))))  # fmt: skip
    return LandscapeNarrative(
        summary=out.summary,
        patterns=findings(out.patterns),
        rising_subjects=findings(out.rising_subjects),
        neglected_subjects=findings(out.neglected_subjects),
        positioning=positioning,
        format_trends=findings(out.format_trends),
        notable_changes=findings(out.notable_changes),
        dropped_findings=dropped,
    )


class LandscapeService:
    def __init__(
        self,
        engine: AsyncEngine,
        sessions: SessionFactory,
        llm: LazyLLM,
        settings: Settings,
        *,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        self._engine = engine
        self._sessions = sessions
        self._llm = llm
        self._settings = settings
        self._now = now

    async def run(
        self, *, trigger: RunTrigger, window_days: int = 30, force: bool = False
    ) -> LandscapeOutcome:
        run_id = await self.create_run(trigger=trigger, window_days=window_days, force=force)
        return await self.execute(run_id)

    async def create_run(self, *, trigger: RunTrigger, window_days: int, force: bool) -> int:
        if not self._llm.configured:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is not set. Landscape reports (Phase 3) require it."
            )
        async with self._sessions() as session, session.begin():
            free = await run_slot_free(
                session,
                kind=RUN_KIND,
                competitor_id=None,
                lock=landscape_lock(self._engine),
                now=self._now(),
            )
            if not free:
                raise LandscapeAlreadyRunningError("A landscape report is already being generated")
            run = Run(
                kind=RUN_KIND,
                trigger=trigger.value,
                status=RunStatus.QUEUED.value,
                competitor_id=None,
                params={"window_days": window_days, "force": force},
                created_at=self._now(),
            )
            session.add(run)
            await session.flush()
            return run.id

    async def execute(self, run_id: int) -> LandscapeOutcome:
        async with landscape_lock(self._engine) as acquired:
            if not acquired:
                return await self._fail(run_id, "another landscape report is being generated")
            llm: BudgetedLLM | None = None
            try:
                async with self._sessions() as session, session.begin():
                    await fail_abandoned_runs(session, kind=RUN_KIND, competitor_id=None, now=self._now(), keep=run_id)  # fmt: skip
                    run = await session.get_one(Run, run_id)
                    run.status = RunStatus.RUNNING.value
                    run.started_at = self._now()
                    window_days = int(run.params.get("window_days", 30))
                    force = bool(run.params.get("force", False))
                llm = BudgetedLLM(self._llm.get(), self._sessions, self._settings, run_id=run_id, now=self._now)  # fmt: skip
                return await self._generate(run_id, llm, window_days=window_days, force=force)
            except asyncio.CancelledError:
                await self._fail(run_id, "cancelled", llm)
                raise
            except LLMError as exc:
                return await self._fail(run_id, f"{type(exc).__name__}: {exc}", llm)
            except Exception as exc:  # the run must never be left "running"
                log.exception("landscape.crashed", run_id=run_id)
                return await self._fail(run_id, f"{type(exc).__name__}: {exc}", llm)

    async def _generate(
        self, run_id: int, llm: BudgetedLLM, *, window_days: int, force: bool
    ) -> LandscapeOutcome:
        landscape = await IntelligenceService(self._sessions, self._settings, now=self._now).landscape(window_days=window_days)  # fmt: skip
        if not any(c.analyzed_items for c in landscape.competitors):
            return await self._fail(run_id, "no analyzed content yet: run `analyze` first", llm)
        data = render_data(landscape)
        model = self._settings.synthesis_model
        input_hash = hashlib.sha256(f"{prompt.VERSION}\n{model}\n{data}".encode()).hexdigest()
        async with self._sessions() as session:
            latest = await analysis_queries.latest_landscape_row(session)
        if latest is not None and latest.input_hash == input_hash and not force:
            await finish_run(self._sessions, run_id, status=RunStatus.SUCCEEDED, now=self._now(), summary={"report_id": latest.id, "unchanged": True})  # fmt: skip
            return LandscapeOutcome(run_id, RunStatus.SUCCEEDED, latest.id, unchanged=True, usage=llm.usage)  # fmt: skip
        response = await llm.structured(
            LLMRequest(
                prompt=prompt.render(data=data),
                system=prompt.SYSTEM,
                model=model,
                max_output_tokens=12_000,
                reasoning_effort=self._settings.synthesis_reasoning_effort,
            ),
            prompt.LandscapeOut,
            purpose=LLMPurpose.LANDSCAPE,
            prompt_version=prompt.VERSION,
            items=len(landscape.competitors),
        )
        narrative = ground(response.data, landscape)
        async with self._sessions() as session, session.begin():
            report = LandscapeReport(
                run_id=run_id,
                window_days=window_days,
                competitor_slugs=[c.competitor for c in landscape.competitors],
                model=response.raw.model,
                prompt_version=prompt.VERSION,
                input_hash=input_hash,
                metrics=landscape.model_dump(mode="json"),
                narrative=narrative.model_dump(mode="json"),
                created_at=self._now(),
            )
            session.add(report)
            await session.flush()
            report_id = report.id
        await finish_run(
            self._sessions, run_id, status=RunStatus.SUCCEEDED, now=self._now(),
            summary={"report_id": report_id, "dropped_findings": narrative.dropped_findings},
            stats=llm.usage.as_dict(),
        )  # fmt: skip
        return LandscapeOutcome(run_id, RunStatus.SUCCEEDED, report_id, usage=llm.usage)

    async def _fail(self, run_id: int, error: str, llm: BudgetedLLM | None = None) -> LandscapeOutcome:  # fmt: skip
        usage = llm.usage if llm else None
        await finish_run(self._sessions, run_id, status=RunStatus.FAILED, now=self._now(), error=error, stats=usage.as_dict() if usage else None)  # fmt: skip
        return LandscapeOutcome(run_id, RunStatus.FAILED, usage=usage, error=error)
