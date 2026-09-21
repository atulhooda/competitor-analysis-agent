"""Importing an article written outside the agent, end to end: real PostgreSQL, the fake
GitHub, Vercel, website and Pexels. Nothing here needs Gemini — that is the point of the
import path — and nothing reaches a real site."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import respx
from sqlalchemy import delete, select, update

from app.cms import LazyCMS
from app.cms.github.mdx import split_frontmatter
from app.config import Settings
from app.db import article_queries, quality_queries
from app.db.models import (
    Article,
    ArticleCitation,
    ArticleQualityReport,
    ArticleSource,
    ArticleStepRun,
    ArticleVersion,
    CompanyProfileVersion,
    Opportunity,
    OpportunityEvent,
    OpportunityEvidence,
    Publication,
)
from app.db.session import SessionFactory, create_session_factory
from app.db.session import create_engine as create_async_db_engine
from app.domain.articles import ArticleOrigin, ArticleStatus, SectionKind
from app.domain.company import CompanyProfile
from app.domain.history import RunTrigger
from app.domain.opportunities import OpportunityStatus
from app.domain.publishing import ApprovalChannel, PublicationStatus, TargetStatus
from app.domain.quality import GateStatus
from app.llm import LazyLLM
from app.services.approvals import ApprovalRefusedError, ApprovalService
from app.services.article_file import ArticleFileError, parse_article_file
from app.services.article_import import ArticleImportService, ImportOutcome
from app.services.articles import ArticleConflictError, ArticleService
from app.services.company import save_company_profile
from app.services.covers import CoverService
from app.services.opportunities import NoCompanyProfileError
from app.services.publishing import PublishingConflictError, PublishingService
from tests.fakegithub import REPO, SITE, TOKEN, FakeGitHub
from tests.fakellm import FakeLLM
from tests.fakepexels import KEY, LANDSCAPE, NARROW, PHOTO_PNG, SQUARE, FakePexels
from tests.fakesite import NOW, make_settings, public_resolver
from tests.integration.test_github_publishing import Clock
from tests.pipeline import ARTICLE_COMPANY, WallClock
from tests.scheduling import no_sleep

GITHUB: dict[str, Any] = {"cms_provider": "github", "github_repo": REPO, "github_token": TOKEN, "publish_site_url": SITE, "cms_max_retries": 1, "github_deploy_poll_seconds": 1}  # fmt: skip
PEXELS: dict[str, Any] = {"publish_cover_images": True, "cover_image_source": "pexels", "pexels_api_key": KEY}  # fmt: skip
MIN_WORDS = 200  # ARTICLE_MIN_WORDS for these tests: the fixture article is ~290 words
DESCRIPTION = "description: A practical guide to splitting customer support work between an AI agent and the people who own the hard cases."  # fmt: skip

FRONTMATTER = """\
---
title: How small support teams hand work to AI agents
description: A practical guide to splitting customer support work between an AI agent and the people who own the hard cases.
primary_keyword: AI agents for customer support
target_audience: founders
content_type: guide
tags:
  - ai agents
  - customer support
  - handoff
slug: ai-agents-customer-support-handoff
sources:
  - label: S1
    title: Zendesk CX Trends
    url: https://example.com/cx-trends
  - label: S2
    title: Intercom support benchmark
    url: https://example.org/benchmark
---
"""

BODY = """\
Support teams of three or four people carry the same ticket volume as teams twice their
size, and the difference is almost always where the work is split. An AI agent that answers
the repetitive half of the queue buys a small team the hours it needs for the half that
actually needs judgement [S1].

## What an agent should answer first

Start with the tickets that already have one correct answer: order status, password resets,
plan changes. These are the questions your team answers from memory, and the ones where a
wrong answer is cheap to correct [S2].

- Order and delivery status
- Password and login problems
- Plan and billing changes that follow a published rule

### The rule of the single source

An agent is only as good as the page it reads from. Keep one canonical help centre article
per answer and let the agent quote it, so a correction reaches every future answer at once.

## Designing the handoff

The handoff is the part teams get wrong. A customer who has explained a problem once should
never have to explain it again, so the agent hands over the whole conversation, its own
confidence and the article it used [S1].

1. The agent says it is handing over, by name
2. The transcript and the agent's reasoning land in the ticket
3. The person answers with the context already in front of them

## What to measure after a month

Measure deflection honestly: a ticket the agent closed that comes back the next day was not
deflected. Track repeat contacts, the share of handoffs a person had to redo, and how long
the queue sat at its longest point of the day [S2].

Small teams that publish these numbers internally tend to keep the agent honest, because
everybody can see the cases it got wrong and the handoffs that felt abrupt to a customer.
"""

ARTICLE_FILE = FRONTMATTER + "\n" + BODY


def file_with(*, frontmatter: str = FRONTMATTER, body: str = BODY) -> str:
    return frontmatter + "\n" + body


@dataclass
class World:
    settings: Settings
    sessions: SessionFactory
    engine: Any
    wall: WallClock
    gh: FakeGitHub
    pex: FakePexels
    clock: Clock
    tmp: Path

    def options(self, **overrides: Any) -> Settings:
        values: dict[str, Any] = {"article_min_words": MIN_WORDS, **GITHUB}
        values.update(overrides)
        return make_settings(database_url=self.settings.database_url.get_secret_value(), **values)  # fmt: skip

    def imports(self, **overrides: Any) -> ArticleImportService:
        s = self.options(**overrides)
        return ArticleImportService(self.engine, self.sessions, s, now=self.wall)

    async def write(self, text: str, name: str = "article.md") -> Path:
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    async def import_text(self, text: str = ARTICLE_FILE, *, opportunity_id: int | None = None, name: str = "article.md", **overrides: Any) -> ImportOutcome:  # fmt: skip
        path = await self.write(text, name)
        return await self.imports(**overrides).import_path(path, opportunity_id=opportunity_id)

    def covers(self, **overrides: Any) -> CoverService:
        s = self.options(**overrides)
        # LazyLLM without a key: a Pexels cover must never need Gemini.
        return CoverService(self.sessions, s, LazyLLM(s), now=self.wall)

    def publishing(self, **overrides: Any) -> PublishingService:
        s = self.options(**overrides)
        covers = self.covers(**overrides)
        cms = LazyCMS(s, sleep=self.clock.sleep, clock=self.clock, covers=covers)
        return PublishingService(self.engine, self.sessions, s, cms, now=self.wall, sleep=no_sleep, covers=covers)  # type: ignore[arg-type]  # fmt: skip

    async def approve(self, article_id: int) -> None:
        await ApprovalService(self.sessions, self.options(), now=self.wall).approve(article_id, channel=ApprovalChannel.CLI, approver="atul", note="Read it myself.")  # fmt: skip

    async def article(self, article_id: int) -> Article:
        async with self.sessions() as session:
            return await session.get_one(Article, article_id)

    async def report(self, article_id: int) -> ArticleQualityReport:
        article = await self.article(article_id)
        async with self.sessions() as session:
            return await session.get_one(ArticleQualityReport, article.quality_report_id or 0)

    async def publication(self, publication_id: int) -> Publication:
        async with self.sessions() as session:
            return await session.get_one(Publication, publication_id)


@pytest.fixture
async def world(db_settings: Settings, tmp_path: Path) -> AsyncIterator[World]:
    engine = create_async_db_engine(db_settings, pooled=False)
    sessions = create_session_factory(engine)
    wall = WallClock(NOW)
    async with sessions() as session, session.begin():
        await save_company_profile(session, CompanyProfile.model_validate(ARTICLE_COMPANY), source="file", now=wall())  # fmt: skip
    gh = FakeGitHub()
    pex = FakePexels(default=[SQUARE, LANDSCAPE, NARROW])
    with respx.mock(assert_all_called=False) as router:
        gh.mount(router)
        pex.mount(router)
        yield World(db_settings, sessions, engine, wall, gh, pex, Clock(), tmp_path)
    await engine.dispose()


# ── the file format ──────────────────────────────────────────────────────────


def test_the_body_becomes_the_content_structure_the_writing_step_produces() -> None:
    parsed = parse_article_file(ARTICLE_FILE, min_words=MIN_WORDS)

    assert parsed.slug == "ai-agents-customer-support-handoff"
    assert parsed.tags == ["ai agents", "customer support", "handoff"]
    assert parsed.cited_labels == ["S1", "S2"]
    kinds = [s.kind for s in parsed.content.sections]
    assert kinds[0] is SectionKind.INTRODUCTION
    assert set(kinds[1:]) == {SectionKind.BODY}
    assert parsed.headings == ["What an agent should answer first", "Designing the handoff", "What to measure after a month"]  # fmt: skip
    second = parsed.content.sections[1]
    assert [b.type.value for b in second.blocks] == ["paragraph", "list", "subheading", "paragraph"]
    assert second.blocks[1].items == ["Order and delivery status", "Password and login problems", "Plan and billing changes that follow a published rule"]  # fmt: skip
    assert parsed.content.sections[2].blocks[1].ordered
    # Wrapped lines are one paragraph, and the citation marker stays where the claim is.
    assert "[S1]" in (parsed.content.sections[0].blocks[0].text or "")
    assert "\n" not in (parsed.content.sections[0].blocks[0].text or "")


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("no frontmatter", "YAML frontmatter"),
        ("no sources", "sources must list at least one"),
        ("two tags", "the site wants 3-6"),
        ("unknown format", "content_type"),
        ("no description", "description is required"),
        ("relative source url", "absolute http(s) url"),
    ],
)
def test_a_file_that_is_not_an_article_names_what_is_wrong(change: str, expected: str) -> None:
    no_tags = FRONTMATTER.replace("  - handoff\n", "").replace("  - customer support\n", "")
    no_description = FRONTMATTER.replace(DESCRIPTION, "description: '  '")
    files = {
        "no frontmatter": BODY,
        "no sources": file_with(frontmatter=FRONTMATTER.split("sources:")[0] + "---\n"),
        "two tags": file_with(frontmatter=no_tags),
        "unknown format": file_with(
            frontmatter=FRONTMATTER.replace("content_type: guide", "content_type: press_release")
        ),
        "no description": file_with(frontmatter=no_description),
        "relative source url": file_with(
            frontmatter=FRONTMATTER.replace("https://example.org/benchmark", "/benchmark")
        ),
    }
    with pytest.raises(ArticleFileError) as caught:
        parse_article_file(files[change], min_words=MIN_WORDS)
    assert expected in str(caught.value)


# ── the import ───────────────────────────────────────────────────────────────


async def test_a_file_becomes_a_ready_article_with_its_own_opportunity(world: World) -> None:
    outcome = await world.import_text()

    assert outcome.created
    assert outcome.word_count >= MIN_WORDS
    assert outcome.sources == 2
    article = await world.article(outcome.article_id)
    assert article.origin == ArticleOrigin.IMPORTED.value
    assert article.status == ArticleStatus.READY.value
    assert article.slug == "ai-agents-customer-support-handoff"
    assert article.recommended_version_id == outcome.version_id
    assert article.quality_report_id == outcome.quality_report_id
    assert article.tokens_used == 0
    assert article.quality_score is None  # there is no agent score, and none is invented
    assert article.validated_at is None
    async with world.sessions() as session:
        opportunity = await session.get_one(Opportunity, outcome.opportunity_id)
        events = list(await session.scalars(select(OpportunityEvent).where(OpportunityEvent.opportunity_id == opportunity.id).order_by(OpportunityEvent.id)))  # fmt: skip
        evidence = list(await session.scalars(select(OpportunityEvidence).where(OpportunityEvidence.assessment_id == opportunity.current_assessment_id)))  # fmt: skip
        steps = [(s.step, s.prompt_version, s.model, s.llm_calls, s.tokens) for s in await session.scalars(select(ArticleStepRun).where(ArticleStepRun.article_id == article.id).order_by(ArticleStepRun.id))]  # fmt: skip
        sources = list(await session.scalars(select(ArticleSource).where(ArticleSource.article_id == article.id).order_by(ArticleSource.id)))  # fmt: skip
        citations = list(await session.scalars(select(ArticleCitation).where(ArticleCitation.version_id == outcome.version_id)))  # fmt: skip
        version = await session.get_one(ArticleVersion, outcome.version_id)
    assert opportunity.key == "manual:how small support teams hand work to ai agent"
    assert opportunity.status == OpportunityStatus.APPROVED.value
    assert [e.kind for e in events] == ["created"]
    assert [e.actor for e in events] == ["import"]
    assert [e.kind for e in evidence] == ["company_profile"]
    assert [s[0] for s in steps] == ["brief", "research", "edit", "seo", "decision"]
    assert {s[1] for s in steps} == {"import/1"}
    assert {(s[2], s[3], s[4]) for s in steps} == {(None, 0, 0)}  # no model, no call, no token
    assert [(s.label, s.title) for s in sources] == [("S1", "Zendesk CX Trends"), ("S2", "Intercom support benchmark")]  # fmt: skip
    assert {c.source_id for c in citations} == {s.id for s in sources}
    assert version.prompt_version == "import/1"
    assert version.model is None
    assert version.kind == "final"


async def test_the_authored_report_records_the_gemini_gates_as_not_run(world: World) -> None:
    outcome = await world.import_text()

    report = await world.report(outcome.article_id)
    assert report.authored
    assert report.passed
    assert report.fact_check_step_id is None
    assert report.originality_step_id is None
    assert report.judge_step_id is None
    assert report.seo_step_id is not None
    gates = {g["name"]: g for g in report.gates}
    ran = {name for name, g in gates.items() if g.get("status") != GateStatus.NOT_RUN.value}
    assert ran == {"content_valid", "citation_integrity", "mdx_safe", "seo_fields"}
    not_run = {name for name, g in gates.items() if g.get("status") == GateStatus.NOT_RUN.value}
    assert not_run == {"no_contradicted_claims", "unsupported_claims", "uncited_claims", "originality", "minimum_score"}  # fmt: skip
    assert all(g["detail"] == "written by a person, not fact-checked by the agent" for name, g in gates.items() if name in not_run)  # fmt: skip
    async with world.sessions() as session:
        view = await quality_queries.quality_overview(session, outcome.article_id, token_budget=1)
        detail = await article_queries.get_article(session, outcome.article_id, token_budget=1)
        versions = await article_queries.list_versions(session, outcome.article_id)
    assert view is not None
    assert view.report is not None
    assert view.report.authored
    assert view.judge is None  # nothing claims a rubric that was never asked for
    assert detail is not None
    assert detail.origin is ArticleOrigin.IMPORTED
    assert versions is not None
    assert [v.authored for v in versions] == [True]


async def test_a_file_can_attach_to_an_opportunity_the_caller_names(world: World) -> None:
    """The caller can name an opportunity that already exists; the article attaches to it and
    the opportunity is approved, because a person decided to write it."""
    first = await world.import_text()
    async with world.sessions() as session, session.begin():
        opportunity = await session.get_one(Opportunity, first.opportunity_id)
        opportunity.status = OpportunityStatus.REVIEWED.value
        await session.execute(update(Article).where(Article.id == first.article_id).values(status=ArticleStatus.CANCELLED.value))  # fmt: skip

    second = await world.import_text(file_with(frontmatter=FRONTMATTER.replace("slug: ai-agents-customer-support-handoff", "slug: ai-agents-handoff-again")), opportunity_id=first.opportunity_id)  # fmt: skip

    assert second.created
    assert second.opportunity_id == first.opportunity_id
    assert second.article_id != first.article_id
    async with world.sessions() as session:
        opportunity = await session.get_one(Opportunity, first.opportunity_id)
        article = await session.get_one(Article, second.article_id)
    assert opportunity.status == OpportunityStatus.APPROVED.value
    assert article.attempt == 2


async def test_importing_the_same_file_twice_keeps_one_article(world: World) -> None:
    first = await world.import_text()

    again = await world.import_text(name="article-copy.md")

    assert not again.created
    assert again.article_id == first.article_id
    assert again.opportunity_id == first.opportunity_id
    assert "already has article" in (again.message or "")
    async with world.sessions() as session:
        articles = list(await session.scalars(select(Article.id).where(Article.opportunity_id == first.opportunity_id)))  # fmt: skip
        versions = list(await session.scalars(select(ArticleVersion.id).where(ArticleVersion.article_id == first.article_id)))  # fmt: skip
        opportunities = list(await session.scalars(select(Opportunity.id)))
    assert articles == [first.article_id]
    assert versions == [first.version_id]
    assert len(opportunities) == 1


async def test_a_file_without_a_company_profile_is_refused(world: World) -> None:
    async with world.sessions() as session, session.begin():
        await session.execute(delete(CompanyProfileVersion))

    with pytest.raises(NoCompanyProfileError, match="company import"):
        await world.import_text()


# ── the refusals ─────────────────────────────────────────────────────────────


async def test_a_citation_without_a_source_is_refused(world: World) -> None:
    body = BODY.replace("wrong answer is cheap to correct [S2]", "wrong answer is cheap to correct [S3]")  # fmt: skip

    with pytest.raises(ArticleFileError) as caught:
        await world.import_text(file_with(body=body))

    assert "the body cites S3" in str(caught.value)
    assert "no source in the frontmatter defines" in str(caught.value)
    async with world.sessions() as session:
        assert list(await session.scalars(select(Article.id))) == []
        assert list(await session.scalars(select(Opportunity.id))) == []


async def test_a_source_the_body_never_cites_is_refused(world: World) -> None:
    body = BODY.replace(" [S2]", "")

    with pytest.raises(ArticleFileError, match="listed but never cited"):
        await world.import_text(file_with(body=body))


async def test_a_file_under_the_minimum_word_count_is_refused(world: World) -> None:
    short = BODY.split("## Designing the handoff")[0]

    with pytest.raises(ArticleFileError) as caught:
        await world.import_text(file_with(body=short), article_min_words=600)

    assert "at least 600 are required" in str(caught.value)
    async with world.sessions() as session:
        assert list(await session.scalars(select(Article.id))) == []


@pytest.mark.parametrize(
    ("markup", "expected"),
    [
        ("<BlogCTA title='mine' />", "the body is prose, not markup"),
        ("export const meta = 1", "MDX would read it as code"),
        ("The value is {frontmatter.title}", "MDX would evaluate it"),
        ("# A second title", "the body has an H1"),
    ],
)
async def test_a_body_that_would_break_mdx_is_refused(world: World, markup: str, expected: str) -> None:  # fmt: skip
    with pytest.raises(ArticleFileError) as caught:
        await world.import_text(file_with(body=BODY + "\n" + markup + "\n"))

    assert expected in str(caught.value)
    async with world.sessions() as session:
        assert list(await session.scalars(select(Article.id))) == []


async def test_a_slug_the_site_cannot_use_is_refused(world: World) -> None:
    with pytest.raises(ArticleFileError, match="no usable slug"):
        await world.import_text(file_with(frontmatter=FRONTMATTER.replace("slug: ai-agents-customer-support-handoff", "slug: '。。。'")))  # fmt: skip


async def test_the_agent_never_rewrites_an_imported_article(world: World) -> None:
    outcome = await world.import_text()
    settings = world.options()
    llm = LazyLLM(settings, provider=FakeLLM())  # configured, and never called
    articles = ArticleService(world.engine, world.sessions, llm, settings, now=world.wall, resolver=public_resolver)  # type: ignore[arg-type]  # fmt: skip

    with pytest.raises(ArticleConflictError, match="written by a person and imported"):
        await articles.resume(outcome.article_id, trigger=RunTrigger.CLI)
    with pytest.raises(ArticleConflictError, match="never rewrites it"):
        await articles.create(outcome.opportunity_id, trigger=RunTrigger.CLI)


# ── the loophole the authored report must never become ───────────────────────


async def test_an_authored_report_never_lets_a_generated_article_skip_its_gates(world: World) -> None:  # fmt: skip
    """The same rows, with the article's origin flipped back to ``generated``: the report is
    then a claim that the agent wrote a piece and checked nothing, and nothing may publish it."""
    outcome = await world.import_text()
    await world.approve(outcome.article_id)
    async with world.sessions() as session, session.begin():
        await session.execute(update(Article).where(Article.id == outcome.article_id).values(origin=ArticleOrigin.GENERATED.value))  # fmt: skip

    with pytest.raises(ApprovalRefusedError, match="written by the agent"):
        await world.approve(outcome.article_id)
    with pytest.raises(PublishingConflictError, match="written by the agent"):
        await world.publishing().request(outcome.article_id, trigger=RunTrigger.CLI)
    report = await world.publishing().preflight(outcome.article_id)
    assert not report.ready
    assert not next(c for c in report.checks if c.name == "quality").passed
    assert world.gh.mutations == []


async def test_a_gate_recorded_as_not_run_needs_an_authored_report(world: World) -> None:
    """The other direction: an ordinary report may not carry a not-run gate either, so the
    marker alone can never buy a generated article a pass."""
    outcome = await world.import_text()
    async with world.sessions() as session, session.begin():
        await session.execute(update(ArticleQualityReport).where(ArticleQualityReport.article_id == outcome.article_id).values(authored=False))  # fmt: skip

    with pytest.raises(ApprovalRefusedError, match="records gate\\(s\\) as not run"):
        await world.approve(outcome.article_id)


# ── publishing an imported article ───────────────────────────────────────────


async def test_an_imported_article_publishes_end_to_end_with_its_cover(world: World) -> None:
    outcome = await world.import_text()
    await world.approve(outcome.article_id)

    service = world.publishing(**PEXELS, publish_allow_direct_publish=True)
    result, published = await service.publish_now(outcome.article_id, trigger=RunTrigger.CLI, target=TargetStatus.PUBLISH)  # fmt: skip

    assert published is not None
    assert published.run_status.value == "succeeded", published.error
    assert published.status is PublicationStatus.PUBLISHED
    assert published.url == f"{SITE}/blog/ai-agents-customer-support-handoff"
    [pr] = list(world.gh.pulls.values())
    assert pr.merged_at is not None
    live = world.gh.file("ai-agents-customer-support-handoff") or ""
    fields, body = split_frontmatter(live)
    assert fields is not None
    assert fields["title"] == "How small support teams hand work to AI agents"
    assert fields["tags"] == ["ai agents", "customer support", "handoff"]
    assert fields["draft"] is False
    assert fields["coverImage"] == "/blog/covers/ai-agents-customer-support-handoff.png"
    assert world.gh.image("public/blog/covers/ai-agents-customer-support-handoff.png") == PHOTO_PNG
    assert body.count("<BlogCTA") == 1
    assert "## What an agent should answer first" in body
    assert "[Zendesk CX Trends](https://example.com/cx-trends)" in body  # the citation is a link
    publication = await world.publication(result.publication_id or 0)
    assert publication.details["word_count"] == outcome.word_count


async def test_the_publication_and_its_pull_request_say_who_wrote_it(world: World) -> None:
    outcome = await world.import_text()
    await world.approve(outcome.article_id)

    result, published = await world.publishing().publish_now(outcome.article_id, trigger=RunTrigger.CLI)  # fmt: skip

    assert published is not None
    assert published.run_status.value == "succeeded", published.error
    [pr] = world.gh.open_pulls()
    assert "Written by hand" in pr.body
    assert "not** fact-checked" in pr.body
    assert "Quality score: none (written by a person, not scored by the agent)" in pr.body
    assert "Gemini article generation" not in pr.body
    publication = await world.publication(result.publication_id or 0)
    assert publication.details["authored"] is True
    provenance = next(c for c in (publication.preflight or {}).get("checks", []) if c["name"] == "provenance")  # fmt: skip
    assert "did not fact-check" in provenance["detail"]
    assert TOKEN not in str(publication.details)
