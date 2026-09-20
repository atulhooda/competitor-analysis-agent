"""Model-written explanations of significant changes to positioning and pricing pages.

Which changes: ``pricing_changed`` events, and non-minor ``updated`` events on homepage,
pricing, product and landing pages (deterministic, from Phase 2). What the model sees:
the change facts and a deterministic diff (removed and added lines only). One summary per
version transition; already-summarized transitions are never sent again.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import and_, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.models import ChangeEvent, ChangeSummary, Competitor, ContentItem, ContentVersion
from app.db.session import SessionFactory
from app.domain.analysis import POSITIONING_TYPES, LLMPurpose
from app.domain.history import ChangeType
from app.llm import LLMRequest, LLMResponseError
from app.prompts import change_summary
from app.services.change_detection import diff_excerpt
from app.services.llm_usage import BudgetedLLM

LOOKBACK_DAYS = 30
_DIFF_MAX_CHARS = 6_000


@dataclass
class ChangeSummaryResult:
    summarized: int = 0
    failed: int = 0
    pending: int = 0  # eligible changes left for a later run (limit reached)


async def _pending(
    session: AsyncSession, competitor_id: int, since: datetime
) -> list[tuple[ChangeEvent, ContentItem]]:
    summarized = exists().where(ChangeSummary.to_version_id == ChangeEvent.to_version_id)
    pricing = ChangeEvent.change_type == ChangeType.PRICING_CHANGED.value
    rows = await session.execute(
        select(ChangeEvent, ContentItem)
        .join(ContentItem, ContentItem.id == ChangeEvent.content_item_id)
        .where(
            ChangeEvent.competitor_id == competitor_id,
            ChangeEvent.detected_at >= since,
            ChangeEvent.from_version_id.is_not(None),
            ChangeEvent.to_version_id.is_not(None),
            or_(
                pricing,
                and_(
                    ChangeEvent.change_type == ChangeType.UPDATED.value,
                    ChangeEvent.is_minor.is_(False),
                    ContentItem.content_type.in_([t.value for t in POSITIONING_TYPES]),
                ),
            ),
            ~summarized,
        )
        .order_by(ChangeEvent.detected_at.desc(), pricing.desc(), ChangeEvent.id.desc())
    )
    seen: set[int] = set()
    unique = []
    for event, item in rows:
        if event.to_version_id not in seen:  # prefer the pricing_changed event (ordered first)
            seen.add(event.to_version_id or 0)
            unique.append((event, item))
    return unique


def _facts(event: ChangeEvent, before: ContentVersion, after: ContentVersion) -> list[str]:
    details = event.details
    facts = []
    if (before.title or "") != (after.title or ""):
        facts += [f"Title before: {before.title or '(none)'}", f"Title after: {after.title or '(none)'}"]  # fmt: skip
    facts.append(f"Length: {before.word_count} → {after.word_count} words")
    if event.change_type == ChangeType.PRICING_CHANGED.value:
        facts.append(f"Prices before: {', '.join(details.get('prices_before', [])) or '(none)'}")
        facts.append(f"Prices after: {', '.join(details.get('prices_after', [])) or '(none)'}")
    facts.append(f"Detected: {event.detected_at:%Y-%m-%d}")
    return facts


async def summarize_changes(
    *,
    sessions: SessionFactory,
    llm: BudgetedLLM,
    settings: Settings,
    competitor: Competitor,
    run_id: int,
    now: datetime,
    limit: int,
) -> ChangeSummaryResult:
    """Summarize up to ``limit`` pending changes. Budget and provider errors propagate."""
    result = ChangeSummaryResult()
    async with sessions() as session:
        pending = await _pending(session, competitor.id, now - timedelta(days=LOOKBACK_DAYS))
        result.pending = max(len(pending) - limit, 0)
        work = []
        for event, item in pending[:limit]:
            before = await session.get_one(ContentVersion, event.from_version_id)
            after = await session.get_one(ContentVersion, event.to_version_id)
            work.append((event, item, before, after))

    for event, item, before, after in work:
        diff = diff_excerpt(before.text, after.text, max_chars=_DIFF_MAX_CHARS)
        prompt = change_summary.render(
            competitor=competitor.name,
            website=competitor.website,
            url=item.url,
            content_type=item.content_type,
            change_facts=_facts(event, before, after),
            diff=diff.replace("</diff", "<\\/diff") or "(no text lines changed)",
        )
        request = LLMRequest(
            prompt=prompt,
            system=change_summary.SYSTEM,
            model=settings.synthesis_model,
            max_output_tokens=2_000,
            reasoning_effort=settings.synthesis_reasoning_effort,
        )
        try:
            response = await llm.structured(
                request,
                change_summary.ChangeSummaryOut,
                purpose=LLMPurpose.CHANGE_SUMMARY,
                prompt_version=change_summary.VERSION,
            )
        except LLMResponseError:
            result.failed += 1  # unusable output for this one change; try the others
            continue
        out = response.data
        async with sessions() as session, session.begin():
            session.add(
                ChangeSummary(
                    competitor_id=competitor.id,
                    change_event_id=event.id,
                    content_item_id=item.id,
                    to_version_id=after.id,
                    run_id=run_id,
                    summary=out.summary,
                    significance=out.significance.value,
                    categories=[c.value for c in out.categories],
                    key_changes=out.key_changes,
                    model=response.raw.model,
                    prompt_version=change_summary.VERSION,
                    input_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                    created_at=now,
                )
            )
        result.summarized += 1
    return result
