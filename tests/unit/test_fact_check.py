"""Fact-checking (Phase 6) without a database: evidence grounding, claim and uncited-claim
extraction, citation integrity, verdict aggregation, metrics, re-reading and caching, against
a fake Gemini."""

from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from app.domain.articles import ArticleContent
from app.domain.quality import ClaimVerdict, UncitedClaim
from app.llm import LLMRequest, LLMResponseError, StructuredResponse
from app.prompts import fact_check as prompt
from app.services.fact_check import (
    FactCheckConfig,
    PriorCheck,
    SourceMaterial,
    cited_claims,
    claim_key,
    claim_verdict,
    evidence_found,
    integrity_problems,
    metrics,
    run_fact_check,
    uncited_candidates,
)
from tests.fakellm import FakeLLM
from tests.fakesite import public_resolver

CONFIG = FactCheckConfig(model="fake-gemini", reasoning_effort="low", batch_size=8, max_rereads=4, max_uncited=30)  # fmt: skip
HANDOFF = SourceMaterial(
    source_id=11,
    label="S1",
    url="https://standards.example.org/ai-agents/handoff",
    title="Human handoff guidelines",
    publisher="Example Standards Body",
    facts=(
        (
            "Handoffs should pass the full conversation so customers never repeat themselves.",
            "pass the full conversation",
        ),
    ),
    excerpt="Every automated channel needs a documented escalation path.",
)
STUDY = SourceMaterial(
    source_id=12,
    label="S2",
    url="https://research.example.edu/papers/agent-evaluation",
    title="Evaluating AI support agents",
    publisher="Example University",
    facts=(
        (
            "In a study of 1,200 support conversations, AI agents resolved 64% of routine tickets.",
            "resolved 64% of routine tickets",
        ),
    ),
    excerpt=None,
)
SOURCES = {"S1": HANDOFF, "S2": STUDY}


def content(*paragraphs: str) -> ArticleContent:
    return ArticleContent.model_validate(
        {
            "title": "AI agents for founders",
            "description": "A guide.",
            "sections": [
                {
                    "kind": "introduction",
                    "heading": None,
                    "blocks": [{"type": "paragraph", "text": p} for p in paragraphs],
                },
                {
                    "kind": "conclusion",
                    "heading": "Getting started",
                    "blocks": [
                        {
                            "type": "list",
                            "ordered": False,
                            "items": ["Write the scope down first.", "Check the evidence [S1]."],
                        }
                    ],
                },
            ],
        }
    )


@dataclass
class StubLLM:
    """``BudgetedLLM.structured`` over the fake Gemini, without the ledger."""

    fake: FakeLLM = field(default_factory=FakeLLM)
    purposes: list[str] = field(default_factory=list)

    async def structured(self, request: LLMRequest, schema: type[BaseModel], **kw: Any) -> StructuredResponse[Any]:  # fmt: skip
        self.purposes.append(kw["purpose"].value)
        return await self.fake.generate_structured(request, schema)


# ── evidence ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("quote", "found"),
    [
        ("pass the full conversation", True),
        ("Handoffs should pass the full conversation, so customers never repeat themselves", True),
        ("handoffs SHOULD pass the full conversation so customers never repeat", True),
        ("Handoffs must always include a phone call within five minutes", False),  # invented
        ("the", False),  # too short to ground anything
        (None, False),
        ("", False),
    ],
)
def test_evidence_must_come_from_the_notes(quote: str | None, found: bool) -> None:
    assert evidence_found(quote, HANDOFF.text()) is found


def test_notes_include_facts_quotes_and_excerpt() -> None:
    text = HANDOFF.text()
    assert "documented escalation path" in text
    assert "pass the full conversation" in text


# ── claims ───────────────────────────────────────────────────────────────────


def test_cited_claims_are_grouped_by_location() -> None:
    rows = [(0, 0, None, "Agents help.", "S1"), (0, 0, None, "Agents help.", "S2"), (0, 0, None, "Agents help.", "S1"), (1, 0, 1, "Check the evidence.", "S1")]  # fmt: skip
    claims = cited_claims(rows)
    assert [(c.claim, c.labels) for c in claims] == [("Agents help.", ("S1", "S2")), ("Check the evidence.", ("S1",))]  # fmt: skip
    assert claims[0].key == claim_key("agents  HELP")  # normalized


def test_uncited_candidates_find_factual_looking_sentences() -> None:
    article = content(
        "AI agents resolved 64% of routine tickets [S2]. Start small and review transcripts every week with the team.",
        "In March 2025, the European Union adopted a regulation for automated customer service agents.",
        "A survey found that most founders now use chat tools for support.",
        "Is this the right time to adopt AI agents for support teams?",
        "Only 12 percent.",
    )
    found = uncited_candidates(article, limit=10)
    sentences = [c.sentence for c in found]
    assert sentences == [
        "In March 2025, the European Union adopted a regulation for automated customer service agents.",
        "A survey found that most founders now use chat tools for support.",
    ]
    assert {"number", "date", "named_entity", "legal"} <= set(found[0].signals)
    assert "research_reference" in found[1].signals
    assert uncited_candidates(article, limit=1)[0].sentence.startswith("In March 2025")  # strongest first  # fmt: skip


def test_citation_integrity() -> None:
    article = content("Agents help [S1]. They resolve tickets [S9].")
    rows = [(0, 0, None, "Agents help.", "S1"), (1, 0, 1, "Check the evidence.", "S1"), (0, 1, None, "Stale record.", "S2")]  # fmt: skip
    problems = integrity_problems(article, rows, {"S1", "S2"})
    assert problems == [
        "citation(s) S9 don't match a stored source",
        "1 citation(s) in the text have no stored claim → source record",
        "1 stored citation record(s) don't match the text",
    ]
    assert integrity_problems(content("Agents help [S1]."), [(0, 0, None, "Agents help.", "S1"), (1, 0, 1, "Check the evidence.", "S1")], {"S1"}) == []  # fmt: skip


def test_a_claim_is_as_good_as_its_best_source() -> None:
    assert claim_verdict(["unsupported", "supported"]) is ClaimVerdict.SUPPORTED
    assert claim_verdict(["contradicted", "partial"]) is ClaimVerdict.PARTIAL
    assert claim_verdict(["unsupported", "contradicted"]) is ClaimVerdict.CONTRADICTED
    assert claim_verdict([]) is ClaimVerdict.UNSUPPORTED


def test_metrics_are_computed_from_the_verdicts() -> None:
    from app.domain.quality import ClaimResult

    def claim(verdict: str) -> ClaimResult:
        return ClaimResult(key=verdict, section=0, block=0, item=None, claim="x", labels=["S1"], verdict=ClaimVerdict(verdict), checks=[])  # fmt: skip

    def uncited(verdict: str) -> UncitedClaim:
        return UncitedClaim(key=verdict, section=0, block=0, item=None, sentence="x", signals=["number"], verdict=ClaimVerdict(verdict), claim_type="statistic", reason="")  # fmt: skip

    m = metrics([claim("supported"), claim("supported"), claim("partial"), claim("contradicted")], [uncited("needs_verification"), uncited("not_required")], problems=[], rereads=2)  # fmt: skip
    assert (m.cited_claims, m.supported, m.partial, m.contradicted, m.unsupported) == (
        4,
        2,
        1,
        1,
        0,
    )
    assert m.uncited_candidates == 2
    assert m.uncited_factual == 1
    assert m.factual_claims == 5
    assert m.citation_coverage == 0.8
    assert (m.supported_claim_ratio, m.partial_claim_ratio, m.contradicted_claim_ratio) == (0.5, 0.25, 0.25)  # fmt: skip
    assert m.uncited_factual_claim_ratio == 0.2
    assert m.integrity_ok
    assert m.rereads == 2
    empty = metrics([], [], problems=["broken"], rereads=0)
    assert empty.citation_coverage == 1.0
    assert not empty.integrity_ok


# ── the Gemini parts ─────────────────────────────────────────────────────────


async def check(article: ArticleContent, rows: list[tuple[int, int, int | None, str, str]], llm: StubLLM, *, config: FactCheckConfig = CONFIG, prior: dict[tuple[str, int], PriorCheck] | None = None, resolver: Any = public_resolver) -> Any:  # fmt: skip
    return await run_fact_check(
        llm,  # type: ignore[arg-type]
        content=article, claims=cited_claims(rows), sources=SOURCES, config=config, resolver=resolver,
        integrity=[], prior_pairs=prior or {}, prior_uncited={},
    )  # fmt: skip


ROWS = [(0, 0, None, "AI agents resolved 64% of routine tickets.", "S2"), (1, 0, 1, "Check the evidence.", "S1")]  # fmt: skip
ARTICLE = content("AI agents resolved 64% of routine tickets [S2].")


async def test_supported_claims_need_verified_evidence() -> None:
    llm = StubLLM()
    outcome = await check(ARTICLE, ROWS, llm)
    assert [c.verdict for c in outcome.report.claims] == [ClaimVerdict.SUPPORTED] * 2
    assert all(p.evidence_verified and p.model == "fake-gemini" for p in outcome.pairs)
    assert llm.purposes == ["fact_check"]  # one batch, no re-read, nothing uncited


async def test_unverifiable_agreement_is_reread_then_settled() -> None:
    llm = StubLLM()
    llm.fake.verdicts = {"64%": "fabricated"}
    outcome = await check(ARTICLE, ROWS, llm)
    pair = next(p for p in outcome.pairs if p.label == "S2")
    assert pair.reread
    assert pair.verdict == "supported"
    assert not pair.evidence_verified
    assert "re-read from https://research.example.edu/papers/agent-evaluation" in pair.explanation
    assert outcome.report.metrics.rereads == 1
    reread = [r for r, _ in llm.fake.requests if r.tools]
    assert reread[0].system == prompt.REREAD_SYSTEM


@pytest.mark.parametrize(("status", "rereads", "explanation"), [("error", 4, "couldn't be re-read"), ("success", 0, "wasn't re-read")])  # fmt: skip
async def test_what_stays_unsettled_is_unsupported(status: str, rereads: int, explanation: str) -> None:  # fmt: skip
    llm = StubLLM()
    llm.fake.verdicts = {"64%": "insufficient"}
    llm.fake.reread_status = status
    outcome = await check(ARTICLE, ROWS, llm, config=FactCheckConfig(**{**CONFIG.__dict__, "max_rereads": rereads}))  # fmt: skip
    pair = next(p for p in outcome.pairs if p.label == "S2")
    assert pair.verdict == "unsupported"
    assert explanation in pair.explanation
    assert pair.model is None  # not decided by a model: not reused next time


async def test_a_reread_verdict_without_evidence_is_unsupported() -> None:
    llm = StubLLM()
    llm.fake.verdicts = {"64%": "insufficient"}
    llm.fake.reread_verdict = "unsupported"
    outcome = await check(ARTICLE, ROWS, llm)
    pair = next(p for p in outcome.pairs if p.label == "S2")
    assert pair.verdict == "unsupported"
    assert pair.reread


async def test_unsafe_urls_are_not_reread() -> None:
    async def private(host: str) -> list[str]:
        return ["10.0.0.5"]

    llm = StubLLM()
    llm.fake.verdicts = {"64%": "insufficient"}
    outcome = await check(ARTICLE, ROWS, llm, resolver=private)
    pair = next(p for p in outcome.pairs if p.label == "S2")
    assert pair.verdict == "unsupported"
    assert "safety check" in pair.explanation
    assert not [r for r, _ in llm.fake.requests if r.tools]


async def test_contradictions_keep_the_source_evidence() -> None:
    llm = StubLLM()
    llm.fake.verdicts = {"64%": "contradicted"}
    outcome = await check(ARTICLE, ROWS, llm)
    claim = next(c for c in outcome.report.claims if "64%" in c.claim)
    assert claim.verdict is ClaimVerdict.CONTRADICTED
    assert claim.checks[0].evidence == "resolved 64% of routine tickets"


async def test_earlier_verdicts_are_reused_without_a_call() -> None:
    llm = StubLLM()
    key = claim_key("AI agents resolved 64% of routine tickets.")
    prior = {(key, 12): PriorCheck(7, "partial", "earlier", "resolved 64%", True, 0.7, False, None, "fake-gemini")}  # fmt: skip
    outcome = await check(ARTICLE, [ROWS[0]], llm, prior=prior)
    assert llm.purposes == []
    assert outcome.pairs[0].reused_from == 7
    assert outcome.pairs[0].verdict == "partial"
    assert outcome.report.claims[0].checks[0].reused


async def test_an_unusable_batch_is_split_and_retried() -> None:
    llm = StubLLM()
    llm.fake.fail_schema[prompt.FactCheckOut] = [LLMResponseError("bad JSON (fake)"), LLMResponseError("bad JSON (fake)")]  # fmt: skip
    outcome = await check(ARTICLE, ROWS, llm)
    # 2 claims (sorted by source): the batch fails, then the first half fails (unusable),
    # and the second half works.
    assert llm.purposes == ["fact_check"] * 3
    verdicts = {p.claim.claim: p.verdict for p in outcome.pairs}
    assert verdicts == {"Check the evidence.": "unsupported", "AI agents resolved 64% of routine tickets.": "supported"}  # fmt: skip
    assert any("unusable output" in n for n in outcome.report.notes)


async def test_uncited_claims_are_classified_and_kept() -> None:
    llm = StubLLM()
    article = content("In 2025, 64 percent of founders said AI agents cut costs within six months.", "Founders should write the scope down before they start any rollout.")  # fmt: skip
    outcome = await check(article, [], llm)
    assert llm.purposes == ["claim_classification"]
    assert [(u.sentence[:8], u.verdict) for u in outcome.report.uncited] == [("In 2025,", ClaimVerdict.NEEDS_VERIFICATION)]  # fmt: skip
    assert outcome.report.metrics.citation_coverage == 0.0
    request = llm.fake.calls(prompt.ClassifyOut)[0]
    assert "<claims>" in request.prompt
    assert request.system == prompt.CLASSIFY_SYSTEM


async def test_a_failed_classification_falls_back_to_the_signals() -> None:
    llm = StubLLM()
    llm.fake.fail_schema[prompt.ClassifyOut] = [LLMResponseError("bad JSON (fake)")]
    article = content("In 2025, 64 percent of founders said AI agents cut costs within six months.")
    outcome = await check(article, [], llm)
    assert outcome.uncited[0].verdict is ClaimVerdict.NEEDS_VERIFICATION
    assert outcome.uncited[0].model is None
    assert any("classification returned unusable output" in n for n in outcome.report.notes)


async def test_a_label_without_a_source_is_unsupported() -> None:
    llm = StubLLM()
    outcome = await check(content("Agents help [S7]."), [(0, 0, None, "Agents help.", "S7")], llm)
    assert outcome.pairs[0].verdict == "unsupported"
    assert outcome.pairs[0].source_id is None
    assert llm.purposes == []
