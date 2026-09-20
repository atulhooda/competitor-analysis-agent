"""Budgeted, audited access to the LLM for one run.

Before each call: the estimated tokens must fit the per-run budget
(``LLM_MAX_TOKENS_PER_RUN``) and today's remaining budget (``LLM_DAILY_TOKEN_BUDGET``,
UTC day, summed from the ``llm_calls`` ledger). Otherwise ``LLMBudgetExceededError`` is
raised and nothing is sent. After each call, success or failure, one ``llm_calls`` row
records purpose, model, prompt version, token usage and latency.

Concurrent runs check the daily budget independently, so it can be overshot by at most
the calls in flight at that moment.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.timeutils import utcnow
from app.db.models import LLMCall
from app.db.session import SessionFactory
from app.domain.analysis import LLMCallStatus, LLMPurpose
from app.llm import (
    ImageRequest,
    ImageResponse,
    LLMBudgetExceededError,
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMResponseError,
    LLMUsage,
    StructuredResponse,
)
from app.services.digest import estimate_tokens

log = structlog.get_logger(__name__)


# In every budget error message (the scheduler reports such stops as skipped_due_to_budget).
BUDGET_REACHED = "LLM token budget reached"
# What one generated picture is budgeted at before it is made. Gemini bills an image as a
# fixed block of output tokens; the ledger then records what it actually reported.
IMAGE_OUTPUT_TOKENS = 2_000


@dataclass
class RunUsage:
    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "llm_calls": self.calls,
            "llm_failed_calls": self.failed_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


def utc_day_start(now: datetime) -> datetime:
    return now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


async def tokens_used_since(session: AsyncSession, since: datetime) -> int:
    used = await session.scalar(
        select(func.coalesce(func.sum(LLMCall.total_tokens), 0)).where(LLMCall.created_at >= since)
    )
    return int(used or 0)


class BudgetedLLM:
    def __init__(
        self,
        provider: LLMProvider,
        sessions: SessionFactory,
        settings: Settings,
        *,
        run_id: int | None,
        now: Callable[[], datetime] = utcnow,
        token_limit: int | None = None,
        token_limit_name: str = "LLM_MAX_TOKENS_PER_RUN",  # noqa: S107 - a setting name
    ) -> None:
        """``token_limit`` lowers the per-run budget (e.g. what's left of an article's
        budget); ``token_limit_name`` names it in the error."""
        self._provider = provider
        self._sessions = sessions
        self._settings = settings
        self._run_id = run_id
        self._now = now
        self._token_limit = token_limit
        self._token_limit_name = token_limit_name
        self.usage = RunUsage()

    async def structured[T: BaseModel](
        self,
        request: LLMRequest,
        schema: type[T],
        *,
        purpose: LLMPurpose,
        prompt_version: str,
        items: int = 1,
    ) -> StructuredResponse[T]:
        request_chars = len(request.prompt) + len(request.system or "")
        await self._check_budget(self.estimate(request))
        model = request.model or self._provider.default_model
        started = time.monotonic()
        try:
            result = await self._provider.generate_structured(request, schema)
        except LLMError as exc:
            usage = exc.usage if isinstance(exc, LLMResponseError) else None
            await self._record(
                purpose, prompt_version, model, items, request_chars, started,
                status=LLMCallStatus.FAILED, usage=usage, error=f"{type(exc).__name__}: {exc}",
            )  # fmt: skip
            raise
        await self._record(
            purpose, prompt_version, result.raw.model, items, request_chars, started,
            status=LLMCallStatus.SUCCEEDED, usage=result.raw.usage, response_id=result.raw.response_id,
        )  # fmt: skip
        return result

    async def image(
        self,
        request: ImageRequest,
        *,
        purpose: LLMPurpose,
        prompt_version: str,
        items: int = 1,
    ) -> ImageResponse:
        """One generated picture, budgeted and recorded exactly like a text call. The bytes
        never reach the ledger or the log: only the tokens they cost."""
        request_chars = len(request.prompt)
        await self._check_budget(estimate_tokens(request_chars) + IMAGE_OUTPUT_TOKENS)
        model = request.model or self._provider.default_image_model
        started = time.monotonic()
        try:
            response = await self._provider.generate_image(request)
        except LLMError as exc:
            usage = exc.usage if isinstance(exc, LLMResponseError) else None
            await self._record(
                purpose, prompt_version, model, items, request_chars, started,
                status=LLMCallStatus.FAILED, usage=usage, error=f"{type(exc).__name__}: {exc}",
            )  # fmt: skip
            raise
        await self._record(
            purpose, prompt_version, response.model, items, request_chars, started,
            status=LLMCallStatus.SUCCEEDED, usage=response.usage, response_id=response.response_id,
        )  # fmt: skip
        return response

    def estimate(self, request: LLMRequest) -> int:
        """Tokens a call is budgeted at before it's made: the prompt plus a quarter of the
        output ceiling. Tool output (pages read, search results) isn't known in advance."""
        request_chars = len(request.prompt) + len(request.system or "")
        return estimate_tokens(request_chars) + (request.max_output_tokens or 0) // 4

    async def _check_budget(self, estimate: int) -> None:
        run_budget, name = self._settings.llm_max_tokens_per_run, "LLM_MAX_TOKENS_PER_RUN"
        kind = "per-run"
        if self._token_limit is not None and self._token_limit < run_budget:
            run_budget, name, kind = self._token_limit, self._token_limit_name, "remaining"
        if self.usage.total_tokens + estimate > run_budget:
            raise LLMBudgetExceededError(
                f"{kind} {BUDGET_REACHED} ({self.usage.total_tokens:,} used this run, "
                f"next call ≈{estimate:,}, {name}={run_budget:,})"
            )
        daily_budget = self._settings.llm_daily_token_budget
        if daily_budget <= 0:
            return
        async with self._sessions() as session:
            used_today = await tokens_used_since(session, utc_day_start(self._now()))
        if used_today + estimate > daily_budget:
            raise LLMBudgetExceededError(
                f"daily {BUDGET_REACHED} ({used_today:,} used today (UTC), "
                f"next call ≈{estimate:,}, LLM_DAILY_TOKEN_BUDGET={daily_budget:,})"
            )

    async def _record(
        self,
        purpose: LLMPurpose,
        prompt_version: str,
        model: str,
        items: int,
        request_chars: int,
        started: float,
        *,
        status: LLMCallStatus,
        usage: LLMUsage | None = None,
        error: str | None = None,
        response_id: str | None = None,
    ) -> None:
        usage = usage or LLMUsage()
        latency_ms = int((time.monotonic() - started) * 1000)
        self.usage.calls += 1
        self.usage.failed_calls += status is LLMCallStatus.FAILED
        self.usage.input_tokens += usage.input_tokens
        self.usage.output_tokens += usage.output_tokens
        self.usage.total_tokens += usage.total_tokens
        async with self._sessions() as session, session.begin():
            session.add(
                LLMCall(
                    run_id=self._run_id,
                    created_at=self._now(),
                    purpose=purpose.value,
                    provider=self._provider.name,
                    model=model[:100],
                    prompt_version=prompt_version,
                    status=status.value,
                    error=error[:2000] if error else None,
                    items=items,
                    request_chars=request_chars,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    reasoning_tokens=usage.reasoning_tokens,
                    cached_input_tokens=usage.cached_input_tokens,
                    total_tokens=usage.total_tokens,
                    latency_ms=latency_ms,
                    response_id=response_id,
                )
            )
        log.info(
            "llm.call",
            run_id=self._run_id,
            purpose=purpose.value,
            model=model,
            status=status.value,
            items=items,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency_ms,
        )


def usage_window_start(now: datetime, days: int) -> datetime:
    return utc_day_start(now) - timedelta(days=days - 1)
