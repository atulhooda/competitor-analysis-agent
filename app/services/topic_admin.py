"""Taxonomy maintenance: seed import, manual merges, and Gemini-assisted consolidation.

Consolidation asks the model which top-level topics are duplicates, validates the
proposal (known ids only, no topic in two groups) and, only when asked to apply, merges
them deterministically. Every merge is recorded as a run (kind ``topics``) for audit.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from app.config import Settings
from app.core.timeutils import utcnow
from app.db import analysis_queries
from app.db.models import Run, Topic
from app.db.session import SessionFactory
from app.domain.analysis import LLMPurpose, TopicRef
from app.domain.history import RunStatus, RunTrigger
from app.domain.topics import TopicSeed
from app.llm import LazyLLM, LLMRequest
from app.prompts import topic_consolidation as prompt
from app.services.llm_usage import BudgetedLLM
from app.services.topics import (
    MergeSummary,
    SeedImportSummary,
    TopicMergeError,
    TopicRegistry,
    lock_taxonomy,
)

RUN_KIND = "topics"
MAX_TOPICS_IN_PROMPT = 400


class MergeProposal(BaseModel):
    target: TopicRef
    sources: list[TopicRef]
    reason: str


class ConsolidationResult(BaseModel):
    run_id: int | None
    applied: bool
    proposals: list[MergeProposal]
    rejected: int = 0  # proposed merges dropped by validation
    merges: list[dict[str, Any]] = []


@dataclass
class _Validated:
    proposals: list[MergeProposal] = field(default_factory=list)
    pairs: list[tuple[int, int]] = field(default_factory=list)  # (source id, target id)
    rejected: int = 0


def validate_proposal(out: prompt.ConsolidationOut, ids: dict[str, tuple[int, TopicRef]]) -> _Validated:  # fmt: skip
    """Known ids only; a topic may appear in one group only; never merge into itself."""
    result = _Validated()
    used: set[str] = set()
    for group in out.merges:
        target = group.target_id.strip().upper()
        if target not in ids or target in used:
            result.rejected += 1
            continue
        sources = []
        for raw in group.source_ids:
            source = raw.strip().upper()
            if source in ids and source != target and source not in used and source not in sources:  # fmt: skip
                sources.append(source)
            else:
                result.rejected += 1
        if not sources:
            continue
        used.update([target, *sources])
        result.proposals.append(
            MergeProposal(
                target=ids[target][1], sources=[ids[s][1] for s in sources], reason=group.reason
            )
        )
        result.pairs += [(ids[s][0], ids[target][0]) for s in sources]
    return result


class TopicAdminService:
    def __init__(
        self,
        sessions: SessionFactory,
        llm: LazyLLM,
        settings: Settings,
        *,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        self._sessions = sessions
        self._llm = llm
        self._settings = settings
        self._now = now

    async def import_seeds(self, seeds: Sequence[TopicSeed]) -> SeedImportSummary:
        async with self._sessions() as session, session.begin():
            await lock_taxonomy(session)
            return await TopicRegistry(session).import_seeds(seeds)

    async def merge(self, source_slug: str, target_slug: str, *, trigger: RunTrigger) -> MergeSummary:  # fmt: skip
        """Fold one topic into another (a person's decision; recorded as a run)."""
        async with self._sessions() as session, session.begin():
            await lock_taxonomy(session)
            source = await analysis_queries.get_topic(session, source_slug)
            target = await analysis_queries.get_topic(session, target_slug)
            if source is None or target is None:
                missing = source_slug if source is None else target_slug
                raise TopicMergeError(f"Unknown topic {missing!r}")
            summary = await TopicRegistry(session).merge(source, target)
            session.add(self._run(trigger, {"merges": [vars(summary)], "method": "manual"}))
        return summary

    async def consolidate(self, *, apply: bool, trigger: RunTrigger) -> ConsolidationResult:
        provider = self._llm.get()  # raises LLMConfigurationError before anything is recorded
        async with self._sessions() as session:
            topics = await analysis_queries.list_topics(session, limit=MAX_TOPICS_IN_PROMPT)
        if len(topics) < 2:
            return ConsolidationResult(run_id=None, applied=apply, proposals=[])
        ids = {
            f"T{index}": (topic.id, TopicRef(slug=topic.slug, name=topic.name))
            for index, topic in enumerate(topics, start=1)
        }
        async with self._sessions() as session, session.begin():
            run = self._run(trigger, {}, status=RunStatus.RUNNING)
            session.add(run)
            await session.flush()
            run_id = run.id
        llm = BudgetedLLM(provider, self._sessions, self._settings, run_id=run_id, now=self._now)
        try:
            response = await llm.structured(
                LLMRequest(
                    prompt=prompt.render(
                        topics=[
                            f"{ref} | {t.name} | {t.items}"
                            for ref, t in zip(ids, topics, strict=True)
                        ]
                    ),
                    system=prompt.SYSTEM,
                    model=self._settings.synthesis_model,
                    max_output_tokens=6_000,
                    reasoning_effort=self._settings.synthesis_reasoning_effort,
                ),
                prompt.ConsolidationOut,
                purpose=LLMPurpose.TOPIC_CONSOLIDATION,
                prompt_version=prompt.VERSION,
                items=len(topics),
            )
        except Exception as exc:
            await self._finish(run_id, RunStatus.FAILED, {"applied": False}, llm, error=f"{type(exc).__name__}: {exc}")  # fmt: skip
            raise
        validated = validate_proposal(response.data, ids)
        merges: list[dict[str, Any]] = []
        if apply and validated.pairs:
            async with self._sessions() as session, session.begin():
                await lock_taxonomy(session)
                registry = TopicRegistry(session)
                for source_id, target_id in validated.pairs:
                    source = await session.get_one(Topic, source_id)
                    target = await session.get_one(Topic, target_id)
                    merges.append(vars(await registry.merge(source, target)))
        summary = {
            "applied": apply,
            "method": "llm",
            "proposals": [p.model_dump() for p in validated.proposals],
            "rejected": validated.rejected,
            "merges": merges,
        }
        await self._finish(run_id, RunStatus.SUCCEEDED, summary, llm)
        return ConsolidationResult(
            run_id=run_id,
            applied=apply,
            proposals=validated.proposals,
            rejected=validated.rejected,
            merges=merges,
        )

    def _run(self, trigger: RunTrigger, summary: dict[str, Any], *, status: RunStatus = RunStatus.SUCCEEDED) -> Run:  # fmt: skip
        now = self._now()
        return Run(
            kind=RUN_KIND,
            trigger=trigger.value,
            status=status.value,
            competitor_id=None,
            summary=summary,
            created_at=now,
            started_at=now,
            finished_at=now if status is not RunStatus.RUNNING else None,
        )

    async def _finish(
        self,
        run_id: int,
        status: RunStatus,
        summary: dict[str, Any],
        llm: BudgetedLLM,
        *,
        error: str | None = None,
    ) -> None:
        async with self._sessions() as session, session.begin():
            run = await session.get_one(Run, run_id)
            run.status = status.value
            run.summary = summary
            run.stats = llm.usage.as_dict()
            run.error = error
            run.finished_at = self._now()
