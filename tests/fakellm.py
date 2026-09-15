"""A deterministic stand-in for Gemini, implementing ``app.llm.LLMProvider``.

It reads the prompt like the real model would (documents, evidence ids, topic ids) and
answers with valid structured output, so the whole pipeline runs offline. Tests can make
it fail, omit documents, or return specific answers.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.llm import LLMRequest, LLMResponse, LLMResponseError, LLMUsage, StructuredResponse
from app.prompts.change_summary import ChangeSummaryOut
from app.prompts.competitor_profile import ClaimOut, CompetitorProfileOut, PricingTierOut
from app.prompts.content_analysis import (
    ContentAnalysisResponse,
    DocumentAnalysisOut,
    EntityOut,
    TopicLabelOut,
)
from app.prompts.landscape import FindingOut, LandscapeOut, PositioningOut
from app.prompts.opportunity import OpportunityInterpretationOut, OpportunityOut
from app.prompts.topic_consolidation import ConsolidationOut, MergeGroupOut

_DOCUMENT = re.compile(r'<document id="(D\d+)">\n(.*?)\n</document>', re.DOTALL)
_TYPE_TO_FORMAT = {
    "blog_post": "article",
    "pricing": "pricing_page",
    "homepage": "landing_page",
    "case_study": "case_study",
    "product": "product_page",
    "landing_page": "landing_page",
}


def _field(body: str, name: str) -> str:
    match = re.search(rf"^{name}: (.*)$", body, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _topics(title: str, text: str) -> list[TopicLabelOut]:
    """Topic labels chosen from the page, with deliberately inconsistent spellings."""
    lowered = f"{title} {text}".lower()
    labels: list[TopicLabelOut] = []
    if "pricing" in title.lower():
        labels.append(TopicLabelOut(name="Pricing", relevance=0.9, subtopics=["Per-seat pricing"]))  # fmt: skip
    if "agentic" in lowered:
        labels.append(TopicLabelOut(name="Agentic AI", relevance=0.8))
    if "ai support agents" in title.lower():
        labels.append(TopicLabelOut(name="AI Agents", relevance=0.95, subtopics=["Ticket automation", "Human handoff"]))  # fmt: skip
    elif "old post" in title.lower():
        labels.append(TopicLabelOut(name="ai-agents", relevance=0.7, subtopics=["ticket automation"]))  # fmt: skip
    elif "agents" in lowered:
        labels.append(TopicLabelOut(name="AI agents", relevance=0.6))
    if "support" in lowered:
        labels.append(TopicLabelOut(name="Customer support", relevance=0.7))
    if "automation" in title.lower():
        labels.append(TopicLabelOut(name="Automation", relevance=0.9))
    if "launch" in title.lower():
        labels.append(TopicLabelOut(name="Product updates", relevance=0.9))
    return labels or [TopicLabelOut(name="General", relevance=0.5)]


def analyze_document(ref: str, body: str) -> DocumentAnalysisOut:
    title = _field(body, "Title")
    page_type = _field(body, "Page type \\(from URL and markup\\)")
    text = body.split("\nText:\n", 1)[-1]
    return DocumentAnalysisOut.model_validate(
        {
            "document_id": ref,
            "content_quality": "substantive",
            "summary": f"{title or 'The page'} explains how teams use AI agents in support.",
            "topics": [t.model_dump() for t in _topics(title, text)],
            "content_format": _TYPE_TO_FORMAT.get(page_type, "article"),
            "target_audiences": ["Customer support teams", "customer support team", "founders"],
            "intent": "commercial" if page_type in ("pricing", "homepage") else "informational",
            "funnel_stage": "decision" if page_type == "pricing" else "awareness",
            "primary_angle": "AI agents should augment support teams, not replace them",
            "key_themes": ["Faster resolution", "Human handoff", "faster resolution"],
            "keywords": ["ai support agents", "ticket automation"],
            "positioning_claims": ["Resolves tickets with AI agents"]
            if page_type == "homepage"
            else [],
            "entities": [EntityOut(name="Acme", type="company").model_dump()],  # type: ignore[arg-type]
            "language": "en",
            "confidence": 0.8,
        }
    )


@dataclass
class FakeLLM:
    model: str = "fake-gemini"
    requests: list[tuple[LLMRequest, type[BaseModel]]] = field(default_factory=list)
    # Exceptions raised by the next calls, in order (None = answer normally).
    failures: list[Exception | None] = field(default_factory=list)
    # A content-analysis batch containing a document with this text raises LLMResponseError.
    poison: str | None = None
    # Document URLs to leave out of the next content-analysis answer (once each).
    omit_once: set[str] = field(default_factory=set)
    # Custom answers per schema: a callable taking the request.
    answers: dict[type[BaseModel], Callable[[LLMRequest], BaseModel]] = field(default_factory=dict)  # fmt: skip
    # Opportunity interpretations: add a sentence with a number that isn't in the evidence.
    fabricate_numbers: bool = False
    # Opportunity topics (by label) to leave out of the answer.
    omit_topics: set[str] = field(default_factory=set)
    closed: bool = False

    @property
    def name(self) -> str:
        return "fake"

    @property
    def default_model(self) -> str:
        return self.model

    def calls(self, schema: type[BaseModel]) -> list[LLMRequest]:
        return [request for request, s in self.requests if s is schema]

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    async def generate_structured[T: BaseModel](
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        self.requests.append((request, schema))
        usage = LLMUsage(
            input_tokens=len(request.prompt) // 4,
            output_tokens=200,
            total_tokens=len(request.prompt) // 4 + 200,
        )
        if self.failures:
            failure = self.failures.pop(0)
            if failure is not None:
                raise failure
        if schema in self.answers:
            data: Any = self.answers[schema](request)
        elif schema is ContentAnalysisResponse:
            documents = _DOCUMENT.findall(request.prompt)
            if self.poison and any(self.poison in body for _, body in documents):
                raise LLMResponseError("Gemini output does not match (fake)", usage=usage)
            analyses = []
            for ref, body in documents:
                url = _field(body, "URL")
                if url in self.omit_once:
                    self.omit_once.discard(url)
                    continue
                analyses.append(analyze_document(ref, body))
            data = ContentAnalysisResponse(analyses=analyses)
        elif schema is ChangeSummaryOut:
            data = ChangeSummaryOut.model_validate(
                {
                    "summary": "Acme raised its plan prices.",
                    "significance": "high",
                    "categories": ["pricing", "bogus-category"],
                    "key_changes": ["Starter price changed"],
                }
            )
        elif schema is CompetitorProfileOut:
            data = _profile(request.prompt)
        elif schema is LandscapeOut:
            data = _landscape(request.prompt)
        elif schema is ConsolidationOut:
            data = _consolidation(request.prompt)
        elif schema is OpportunityInterpretationOut:
            data = _opportunities(request.prompt, self.fabricate_numbers, self.omit_topics)
        else:  # pragma: no cover - a new schema needs an answer here
            raise AssertionError(f"FakeLLM has no answer for {schema.__name__}")
        return StructuredResponse(
            data=data,
            raw=LLMResponse(
                text=data.model_dump_json(),
                provider="fake",
                model=request.model or self.model,
                usage=usage,
                finish_reason="completed",
                response_id=f"fake-{len(self.requests)}",
            ),
        )

    async def aclose(self) -> None:
        self.closed = True


def _profile(prompt: str) -> CompetitorProfileOut:
    evidence = re.findall(r"^(E\d+) \| (\w+) \|", prompt, re.MULTILINE)
    changes = re.findall(r"^(C\d+) \|", prompt, re.MULTILINE)
    home = next((ref for ref, kind in evidence if kind == "homepage"), evidence[0][0])
    pricing = next((ref for ref, kind in evidence if kind == "pricing"), None)
    return CompetitorProfileOut(
        tagline=ClaimOut(text="Resolve support tickets with AI agents", evidence=[home]),
        description=ClaimOut(
            text="Acme sells an AI customer support platform.", evidence=[home, "E1"]
        ),
        positioning_statement=ClaimOut(
            text="AI-first support automation for growing teams.", evidence=[home]
        ),
        target_audiences=[ClaimOut(text="Customer support teams", evidence=[home])],
        value_propositions=[
            ClaimOut(text="Faster first response", evidence=[home]),
            ClaimOut(text="Invented claim with no evidence", evidence=[]),
            ClaimOut(text="Claim citing a missing item", evidence=["E99"]),
        ],
        pricing_model=ClaimOut(text="Per-agent monthly plans", evidence=[pricing])
        if pricing
        else None,
        pricing_tiers=[
            PricingTierOut(name="Starter", price="$29 per agent per month", evidence=[pricing])
        ]
        if pricing
        else [],
        notable_changes=[ClaimOut(text="Raised prices", evidence=[changes[0]])] if changes else [],
        confidence=1.4,  # clamped to 1.0
    )


def _landscape(prompt: str) -> LandscapeOut:
    competitors = re.findall(r"^- ([a-z0-9-]+) \([^)]*\): \d+ analyzed pages", prompt, re.MULTILINE)  # fmt: skip
    topics = re.findall(r"^- ([a-z0-9-]+) \| ", prompt, re.MULTILINE)
    return LandscapeOut(
        summary="Competitors focus on AI agents for customer support.",
        patterns=[
            FindingOut(text="AI agents dominate.", topics=topics[:1], competitors=competitors[:1]),
            FindingOut(text="Cites a topic that doesn't exist.", topics=["quantum-knitting"]),
        ],
        rising_subjects=[FindingOut(text="Support automation is rising.", topics=topics[:1])],
        positioning=[
            PositioningOut(
                competitor=c, positioning="AI support platform", focus=[*topics[:1], "nope"]
            )
            for c in competitors
        ],
    )


def _consolidation(prompt: str) -> ConsolidationOut:
    rows = re.findall(r"^(T\d+) \| (.+?) \| \d+$", prompt, re.MULTILINE)
    by_name = {name.casefold(): ref for ref, name in rows}
    merges = []
    if "agentic ai" in by_name and "ai agents" in by_name:
        merges.append(
            MergeGroupOut(
                target_id=by_name["ai agents"],
                source_ids=[by_name["agentic ai"], "T999"],
                reason="same subject",
            )
        )
    return ConsolidationOut(merges=merges)


def _opportunities(prompt: str, fabricate: bool, omit: set[str]) -> OpportunityInterpretationOut:
    blocks = re.split(r"^(?=O\d+ \| topic: )", prompt, flags=re.MULTILINE)[1:]
    answers = []
    for block in blocks:
        header = re.match(r"(O\d+) \| topic: (.+?) \| score (\S+)/100", block)
        assert header is not None, block[:200]
        ref, topic, score = header.groups()
        if topic in omit:
            continue
        pages = re.findall(r"^(E\d+) \|", block, re.MULTILINE)
        why = f"Competitors keep publishing on {topic}, and it scored {score}."
        title = f"{topic}: the practical guide for founders"
        if fabricate:
            why += " Competitors published 987 posts on it last week."
            title = f"Why 73% of founders get {topic} wrong"
        answers.append(
            OpportunityOut(
                opportunity_id=ref,
                title=title,
                recommended_angle=f"Show founders how {topic} works in practice.",
                why_now=why,
                target_audience="founders",
                recommended_format="comparison",
                search_intent="comparison",
                differentiation_strategy="Go deeper than the cited pages, with worked examples.",
                strategic_rationale=f"It ties {topic} to the startup's core topics.",
                evidence=[*pages[:1], "E999"],
                confidence=0.7,
            )
        )
    return OpportunityInterpretationOut(opportunities=answers)
