"""The article checkpoint primitives shared by generation (Phase 5) and validation (Phase 6).

Every step execution is an ``article_steps`` row holding a fingerprint of its inputs. A step
runs only when no succeeded row matches its current fingerprint; otherwise its stored output
is reused. Runs are ``runs`` rows with the article's id, and one article runs one step at a
time (its advisory lock).
"""

import hashlib
import json
from datetime import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ArticleStepRun, Run
from app.domain.articles import ArticleStep, StepStatus
from app.domain.history import RunStatus, RunTrigger


def digest(data: Any) -> str:
    """A stable fingerprint of JSON-serializable data."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


async def find_checkpoint(session: AsyncSession, article_id: int, step: ArticleStep, fingerprint: str) -> ArticleStepRun | None:  # fmt: skip
    """The latest succeeded execution of ``step`` with exactly these inputs."""
    row: ArticleStepRun | None = await session.scalar(
        select(ArticleStepRun)
        .where(
            ArticleStepRun.article_id == article_id,
            ArticleStepRun.step == step.value,
            ArticleStepRun.fingerprint == fingerprint,
            ArticleStepRun.status == StepStatus.SUCCEEDED.value,
        )
        .order_by(ArticleStepRun.id.desc())
        .limit(1)
    )
    return row


async def fail_running_steps(session: AsyncSession, article_id: int, now: datetime) -> None:
    """Steps left ``running`` by a process that stopped. Call while holding the article lock."""
    await session.execute(
        update(ArticleStepRun)
        .where(
            ArticleStepRun.article_id == article_id,
            ArticleStepRun.status == StepStatus.RUNNING.value,
        )
        .values(
            status=StepStatus.FAILED.value,
            error="interrupted: the process running this step stopped",
            finished_at=now,
        )
    )


def new_run(kind: str, article_id: int, trigger: RunTrigger, now: datetime, params: dict[str, Any]) -> Run:  # fmt: skip
    return Run(kind=kind, trigger=trigger.value, status=RunStatus.QUEUED.value, competitor_id=None, article_id=article_id, params={"article_id": article_id, **params}, created_at=now)  # fmt: skip
