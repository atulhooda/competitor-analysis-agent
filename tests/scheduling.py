"""Shared harness for the Phase 8 tests: the real job machinery and pipeline over a real
database, the offline fake site, the fake Gemini and a fake WordPress. No network.

``Rig`` starts from a scanned, analyzed site with one approved opportunity ("ai agents"); a
pipeline job then runs every stage for real: scan → analyze → opportunities → generate →
validate → approval → publish.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import inspect, select

from app.cms import LazyCMS
from app.config import Settings
from app.db.models import Article, Opportunity, OpportunityAssessment, OpportunityEvidence
from app.domain.jobs import JobStatus, JobTrigger, JobType, JobView
from app.domain.opportunities import InterpretationConfig, OpportunityStatus, ScoringConfig
from app.llm import LazyLLM
from app.prompts.seo import SEOOut
from app.scheduling.runtime import Scheduling, build_scheduling
from app.services.analysis import AnalysisService
from app.services.articles import ArticleService
from app.services.jobs import JobContext, JobResult
from app.services.opportunities import OpportunityService
from app.services.pipeline import PipelineServices
from app.services.publishing import PublishingService
from app.services.quality import QualityService
from app.services.scans import ScanService
from tests.fakellm import _seo
from tests.fakesite import make_settings, public_resolver
from tests.fakewordpress import BASE, PASSWORD, USERNAME, FakeWordPress
from tests.pipeline import Env

SCORING = ScoringConfig(interpretation=InterpretationConfig(candidates=10, min_score=0, batch_size=4))  # fmt: skip


async def no_sleep(_: float) -> None:
    return None


@dataclass
class Rig:
    env: Env
    wp: FakeWordPress
    opportunity_id: int  # "ai agents", approved

    # Automated publishing fully on (the tests turn switches off one at a time), one article
    # generated and published per day, only the approved opportunity eligible.
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "cms_provider": "wordpress",
        "wordpress_base_url": BASE,
        "wordpress_username": USERNAME,
        "wordpress_application_password": PASSWORD,
        "cms_max_retries": 1,
        "automated_publishing_enabled": True,
        "publish_allow_direct_publish": True,
        "publish_auto_approve": True,
        "pipeline_approve_opportunities": False,
        "max_articles_generated_per_day": 1,
        "max_articles_per_day": 1,
    }

    def settings(self, **overrides: Any) -> Settings:
        values = {**self.DEFAULTS, **overrides}
        return make_settings(database_url=self.env.settings.database_url.get_secret_value(), **values)  # fmt: skip

    def scheduling(self, **overrides: Any) -> Scheduling:
        s = self.settings(**overrides)
        env = self.env
        llm = LazyLLM(s, provider=env.fake)
        cms = LazyCMS(s, sleep=no_sleep)
        engine: Any = env.engine
        services = PipelineServices(
            scans=ScanService(engine, env.sessions, env.fetcher, s, now=env.wall),
            analyses=AnalysisService(engine, env.sessions, llm, s, now=env.wall),
            opportunities=OpportunityService(
                engine, env.sessions, llm, s, now=env.wall, scoring=SCORING
            ),
            articles=ArticleService(
                engine, env.sessions, llm, s, now=env.wall, resolver=public_resolver
            ),
            quality=QualityService(
                engine, env.sessions, llm, s, now=env.wall, resolver=public_resolver
            ),
            publishing=PublishingService(
                engine, env.sessions, s, cms, now=env.wall, sleep=no_sleep
            ),
            cms=cms,
            llm=llm,
        )
        return build_scheduling(engine, env.sessions, s, services, now=env.wall, heartbeat_seconds=3_600)  # fmt: skip

    async def run(self, job_type: JobType = JobType.FULL_PIPELINE, *, trigger: JobTrigger = JobTrigger.CLI, dry_run: bool = False, **overrides: Any) -> JobView:  # fmt: skip
        jobs = self.scheduling(**overrides).jobs
        view, _ = await jobs.enqueue(job_type, trigger=trigger, dry_run=dry_run)
        return await jobs.run(view.id)

    def distinct_slugs(self) -> None:
        """A different SEO slug per article (the fake Gemini's is the keyword): Phase 7
        rightly refuses a second post with a slug that is already taken."""
        counter = iter(range(1, 1_000))
        self.env.fake.answers[SEOOut] = lambda request: _seo(request.prompt, {"slug": f"ai-agents-{next(counter)}"})  # fmt: skip

    async def articles(self) -> list[Article]:
        async with self.env.sessions() as session:
            return list(await session.scalars(select(Article).order_by(Article.id)))

    async def opportunity(self, opportunity_id: int) -> Opportunity:
        async with self.env.sessions() as session:
            return await session.get_one(Opportunity, opportunity_id)

    async def clone_opportunity(self, *, score: float, status: OpportunityStatus = OpportunityStatus.APPROVED) -> int:  # fmt: skip
        """Another opportunity like "ai agents" (same evidence, so the same brief inputs),
        with its own score: several eligible opportunities without a bigger fake site."""
        async with self.env.sessions() as session, session.begin():
            source = await session.get_one(Opportunity, self.opportunity_id)
            assessment = await session.get_one(OpportunityAssessment, source.current_assessment_id)  # fmt: skip
            evidence = list(await session.scalars(select(OpportunityEvidence).where(OpportunityEvidence.assessment_id == assessment.id)))  # fmt: skip
            count = len(list(await session.scalars(select(Opportunity.id))))
            clone = _copy(source, key=f"{source.key}-copy-{count}", title=f"{source.title} (angle {count})", score=score, status=status.value, current_assessment_id=None)  # fmt: skip
            session.add(clone)
            await session.flush()
            copied = _copy(assessment, opportunity_id=clone.id, score=score, previous_assessment_id=None)  # fmt: skip
            session.add(copied)
            await session.flush()
            for row in evidence:
                session.add(_copy(row, assessment_id=copied.id))
            clone.current_assessment_id = copied.id
        self.distinct_slugs()
        return clone.id


def _copy[T](row: T, **overrides: Any) -> T:
    mapper = inspect(type(row))
    values = {c.key: getattr(row, c.key) for c in mapper.column_attrs if c.key != "id"}
    values.update(overrides)
    return type(row)(**values)


class FakeRunner:
    """A job runner for testing the job machinery alone: it records calls, can block until
    released, and returns (or raises) what the test says."""

    def __init__(self) -> None:
        self.calls: list[JobContext] = []
        self.results: list[JobResult | BaseException] = []
        self.gate: Callable[[JobContext], Awaitable[None]] | None = None

    async def run_job(self, ctx: JobContext) -> JobResult:
        self.calls.append(ctx)
        if self.gate is not None:
            await self.gate(ctx)
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return JobResult(JobStatus.COMPLETED)


def at(wall: Any, moment: datetime) -> None:
    wall.now = moment


__all__ = ["SCORING", "FakeRunner", "Rig", "at", "no_sleep"]
