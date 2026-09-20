"""Article research (Phase 5): Gemini with Google Search grounding proposes sources, and
Gemini's URL context tool reads them. This process never fetches a page itself.

    discover   one grounded call: the research questions (claims needing evidence) and
               candidate sources (URLs the model saw in search results)
    screen     deterministic: each URL must be a public http(s) address (the Phase 1 SSRF
               guard: no credentials, no private, loopback or link-local hosts, the domain
               must resolve), deduplicated, typed (your site and competitors' sites are
               recognized by domain), most authoritative first, capped
    read       URL context calls (budgeted, capped): Gemini retrieves the pages and extracts
               facts with supporting excerpts

A candidate becomes a source only if the URL tool reports that page as retrieved: a URL the
model made up, or one that fails to load, never reaches the article. Every fact is tied to
the source it was read from.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Self
from urllib.parse import urlsplit

from app.config import Settings
from app.core.errors import PermanentError
from app.crawling.errors import FetchError, UnsafeDestinationError
from app.crawling.netguard import Resolver, ensure_public_destination
from app.crawling.urls import normalize_url
from app.domain.analysis import LLMPurpose
from app.domain.articles import (
    ATTRIBUTION_REQUIRED,
    SOURCE_PRIORITY,
    ArticleBrief,
    CandidateDisposition,
    ResearchFact,
    ResearchQuestion,
    ResearchResult,
    ResearchSourceData,
    SourceType,
)
from app.llm import (
    LLMBudgetExceededError,
    LLMRequest,
    LLMResponseError,
    ReasoningEffort,
    RetrievedURL,
)
from app.prompts import article_research as prompt
from app.services.article_brief import domain_of
from app.services.llm_usage import BudgetedLLM

MAX_URL_LENGTH = 2_000
MAX_URLS_PER_CALL = 20  # Gemini's URL context limit per request
MAX_FACTS = 40
MAX_EXCERPT = 1_000


class ResearchFailedError(PermanentError):
    """Research couldn't produce enough usable sources. ``result`` is what it found."""

    def __init__(self, message: str, result: ResearchResult) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class ResearchConfig:
    model: str
    reasoning_effort: ReasoningEffort
    max_queries: int
    max_sources: int
    max_url_context_calls: int
    max_tokens: int
    min_sources: int

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            model=settings.writing_model,
            reasoning_effort=settings.research_reasoning_effort,
            max_queries=settings.article_research_max_queries,
            max_sources=settings.article_research_max_sources,
            max_url_context_calls=settings.article_research_max_url_context_calls,
            max_tokens=settings.article_research_max_tokens,
            min_sources=settings.article_research_min_sources,
        )

    def fingerprint_data(self) -> dict[str, Any]:
        """What the research result depends on (the token cap only limits it)."""
        return {
            "model": self.model,
            "reasoning": self.reasoning_effort,
            "max_queries": self.max_queries,
            "max_sources": self.max_sources,
            "max_url_context_calls": self.max_url_context_calls,
            "min_sources": self.min_sources,
        }


@dataclass
class _Candidate:
    url: str
    title: str | None
    publisher: str | None
    source_type: SourceType
    order: int


@dataclass
class _Read:
    candidate: _Candidate
    url: str  # after redirects
    status: str
    page: prompt.PageOut
    call: int


async def safe_public_url(raw: str, resolver: Resolver) -> tuple[str | None, str | None]:
    """(normalized URL, None) for a public http(s) URL; (None, reason) otherwise."""
    raw = raw.strip()
    if len(raw) > MAX_URL_LENGTH:
        return None, "URL too long"
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError:
        return None, "malformed URL"
    if parts.scheme.lower() not in ("http", "https"):
        return None, f"not an http(s) URL ({parts.scheme or 'no scheme'})"
    if parts.username or parts.password:
        return None, "the URL carries credentials"
    if port not in (None, 80, 443):
        return None, f"non-standard port {port}"
    url = normalize_url(raw)
    if url is None:
        return None, "malformed URL"
    host = urlsplit(url).hostname or ""
    try:
        await ensure_public_destination(url, host, resolver)
    except UnsafeDestinationError as exc:
        return None, f"unsafe destination ({exc.detail})"
    except FetchError:
        return None, "the domain does not resolve"
    return url, None


def classify(url: str, suggested: SourceType, competitor_domains: Sequence[str], company_domain: str | None) -> SourceType:  # fmt: skip
    """Your site and competitors' sites are recognized by domain; the model's suggestion
    decides the rest (it can't label anything company or competitor itself)."""
    host = domain_of(url)

    def under(domain: str) -> bool:
        return host == domain or host.endswith("." + domain)

    if company_domain and under(company_domain):
        return SourceType.COMPANY
    if any(under(d) for d in competitor_domains):
        return SourceType.COMPETITOR
    if suggested in ATTRIBUTION_REQUIRED:
        return SourceType.OTHER
    return suggested


def match_retrievals(requested: Sequence[str], results: Sequence[RetrievedURL]) -> dict[str, tuple[str, str]]:  # fmt: skip
    """requested URL → (URL the page was read at, status), from the URL tool's report.

    The tool reports a redirected page under its final URL (status "success") and the
    requested URL as failed. Such an unclaimed success is matched to a failed request on the
    same site, most similar path first, so a broken URL can't borrow another page's success.
    """

    def key(url: str) -> str:
        return normalize_url(url) or url

    status: dict[str, str] = {}
    for result in results:
        k = key(result.url)
        if status.get(k) != "success":
            status[k] = result.status
    requested_keys = {key(u) for u in requested}
    out: dict[str, tuple[str, str]] = {}
    failed = []
    for url in requested:
        if status.get(key(url)) == "success":
            out[url] = (key(url), "success")
        else:
            failed.append(url)
    redirects = list(dict.fromkeys(key(r.url) for r in results if r.status == "success" and key(r.url) not in requested_keys))  # fmt: skip
    pairs = sorted(
        (
            (SequenceMatcher(None, urlsplit(url).path, urlsplit(target).path).ratio(), i, j)
            for i, url in enumerate(failed)
            for j, target in enumerate(redirects)
            if domain_of(url) == domain_of(target)
        ),
        reverse=True,
    )
    claimed_requests: set[int] = set()
    claimed_targets: set[int] = set()
    for _, i, j in pairs:
        if i not in claimed_requests and j not in claimed_targets:
            claimed_requests.add(i)
            claimed_targets.add(j)
            out[failed[i]] = (redirects[j], "success")
    for url in failed:
        out.setdefault(url, (key(url), status.get(key(url), "not_retrieved")))
    return out


async def research(
    llm: BudgetedLLM,
    brief: ArticleBrief,
    config: ResearchConfig,
    *,
    resolver: Resolver,
    company_domain: str | None,
) -> ResearchResult:
    """Research for ``brief``. Raises ``ResearchFailedError`` with fewer than
    ``config.min_sources`` usable sources; LLM budget and availability errors propagate."""
    start = llm.usage.total_tokens
    notes: list[str] = []
    calls = {"discover": 0, "read": 0}

    def spent() -> int:
        return llm.usage.total_tokens - start

    def fits(request: LLMRequest) -> bool:
        return spent() + llm.estimate(request) <= config.max_tokens

    discover = LLMRequest(
        prompt=prompt.render_discover(
            brief, max_questions=config.max_queries, max_sources=config.max_sources
        ),
        system=prompt.DISCOVER_SYSTEM,
        model=config.model,
        max_output_tokens=prompt.DISCOVER_MAX_OUTPUT_TOKENS,
        reasoning_effort=config.reasoning_effort,
        tools=("google_search",),
    )
    if not fits(discover):
        raise LLMBudgetExceededError(f"ARTICLE_RESEARCH_MAX_TOKENS={config.max_tokens:,} is too small for the search call")  # fmt: skip
    found = await llm.structured(discover, prompt.DiscoverOut, purpose=LLMPurpose.ARTICLE_RESEARCH, prompt_version=prompt.VERSION)  # fmt: skip
    calls["discover"] += 1
    queries = list(found.raw.grounding.search_queries)
    if not queries:
        notes.append("Gemini ran no Google Search query; its suggested URLs were still checked by reading them")  # fmt: skip
    # Our own ids (Q1, Q2, ...): the read prompt lists them and facts must cite them.
    questions = [ResearchQuestion(id=f"Q{i}", question=q.question, claim=q.claim) for i, q in enumerate(found.data.questions[: config.max_queries], start=1)]  # fmt: skip

    dispositions: list[CandidateDisposition] = []
    accepted: list[_Candidate] = []
    seen: set[str] = set()
    company = company_domain
    for order, c in enumerate(found.data.sources):
        url, reason = await safe_public_url(c.url, resolver)
        kind = classify(url or c.url, c.source_type, brief.competitor_domains, company)
        if url is None:
            dispositions.append(CandidateDisposition(url=c.url[:500], title=c.title, source_type=kind, outcome="rejected", reason=reason))  # fmt: skip
        elif url in seen:
            dispositions.append(CandidateDisposition(url=url, title=c.title, source_type=kind, outcome="skipped", reason="duplicate of another candidate"))  # fmt: skip
        else:
            seen.add(url)
            accepted.append(_Candidate(url, c.title, c.publisher, kind, order))
    accepted.sort(key=lambda a: (SOURCE_PRIORITY[a.source_type], a.order))
    for extra in accepted[config.max_sources :]:
        dispositions.append(CandidateDisposition(url=extra.url, title=extra.title, source_type=extra.source_type, outcome="skipped", reason="over ARTICLE_RESEARCH_MAX_SOURCES"))  # fmt: skip
    chosen = accepted[: config.max_sources]

    reads: list[_Read] = []
    if chosen:
        size = min(MAX_URLS_PER_CALL, math.ceil(len(chosen) / config.max_url_context_calls))
        batches = [chosen[i : i + size] for i in range(0, len(chosen), size)]
        for index, batch in enumerate(batches):
            if index >= config.max_url_context_calls:
                dispositions += [CandidateDisposition(url=c.url, title=c.title, source_type=c.source_type, outcome="skipped", reason="over ARTICLE_RESEARCH_MAX_URL_CONTEXT_CALLS") for c in batch]  # fmt: skip
                continue
            urls = [c.url for c in batch]
            request = LLMRequest(
                prompt=prompt.render_read(questions, urls),
                system=prompt.READ_SYSTEM,
                model=config.model,
                max_output_tokens=prompt.read_max_output_tokens(len(urls)),
                reasoning_effort=config.reasoning_effort,
                tools=("url_context",),
            )
            if not fits(request):
                pending = [c for b in batches[index:] for c in b]
                notes.append(f"research token budget reached (ARTICLE_RESEARCH_MAX_TOKENS={config.max_tokens:,}): {len(pending)} page(s) not read")  # fmt: skip
                dispositions += [CandidateDisposition(url=c.url, title=c.title, source_type=c.source_type, outcome="skipped", reason="research token budget reached") for c in pending]  # fmt: skip
                break
            calls["read"] += 1
            try:
                read = await llm.structured(request, prompt.ReadOut, purpose=LLMPurpose.ARTICLE_RESEARCH, prompt_version=prompt.VERSION)  # fmt: skip
            except LLMResponseError as exc:
                notes.append(f"page-reading call {calls['read']} returned unusable output; its {len(urls)} page(s) weren't used ({exc})")  # fmt: skip
                dispositions += [CandidateDisposition(url=c.url, title=c.title, source_type=c.source_type, outcome="not_retrieved", reason="unusable model output") for c in batch]  # fmt: skip
                continue
            if not read.raw.grounding.retrieved_urls:
                notes.append(f"page-reading call {calls['read']}: the URL context tool reported no retrieved page")  # fmt: skip
            retrieval = match_retrievals(urls, read.raw.grounding.retrieved_urls)
            pages: dict[str, prompt.PageOut] = {}
            for page in read.data.pages:
                pages.setdefault(normalize_url(page.url) or page.url, page)
            for candidate in batch:
                final_url, status = retrieval[candidate.url]
                page_out = pages.get(candidate.url) or pages.get(final_url)
                if status != "success":
                    reason = f"URL context status: {status}"
                elif page_out is None or not page_out.readable:
                    reason = "retrieved, but the model couldn't read it"
                elif not any(f.statement.strip() for f in page_out.facts):
                    reason = "retrieved, but no relevant facts"
                else:
                    reads.append(_Read(candidate, final_url, status, page_out, calls["read"]))
                    continue
                dispositions.append(CandidateDisposition(url=candidate.url, title=candidate.title, source_type=candidate.source_type, outcome="not_retrieved", reason=reason))  # fmt: skip

    result = _assemble(reads, questions, queries, dispositions, notes, calls)
    if len(result.sources) < config.min_sources:
        raise ResearchFailedError(
            f"research produced {len(result.sources)} usable source(s); ARTICLE_RESEARCH_MIN_SOURCES={config.min_sources}",
            result,
        )
    return result


def _assemble(
    reads: Sequence[_Read],
    questions: Sequence[ResearchQuestion],
    queries: Sequence[str],
    dispositions: list[CandidateDisposition],
    notes: list[str],
    calls: dict[str, int],
) -> ResearchResult:
    question_ids = {q.id for q in questions}

    def relevance(r: _Read) -> float:
        answered = {q for f in r.page.facts for q in f.question_ids if q in question_ids}
        return round(len(answered) / len(question_ids), 2) if question_ids else 0.0

    ranked = sorted(reads, key=lambda r: (SOURCE_PRIORITY[r.candidate.source_type], -relevance(r), r.candidate.order))  # fmt: skip
    sources: list[ResearchSourceData] = []
    facts: list[ResearchFact] = []
    seen_pages: dict[str, str] = {}
    for r in ranked:
        page_key = normalize_url(r.url) or r.url
        if page_key in seen_pages:
            dispositions.append(CandidateDisposition(url=r.candidate.url, title=r.candidate.title, source_type=r.candidate.source_type, outcome="skipped", reason=f"same page as {seen_pages[page_key]}"))  # fmt: skip
            continue
        label = f"S{len(sources) + 1}"
        seen_pages[page_key] = label
        mine = []
        for fact in r.page.facts:
            if len(facts) >= MAX_FACTS or not fact.statement.strip():
                continue
            mine.append(ResearchFact(id=f"F{len(facts) + 1}", source=label, statement=fact.statement, excerpt=fact.excerpt, question_ids=[q for q in fact.question_ids if q in question_ids], kind=fact.kind or "finding"))  # fmt: skip
            facts.append(mine[-1])
        excerpt = " … ".join(f.excerpt for f in mine if f.excerpt)[:MAX_EXCERPT] or None
        sources.append(
            ResearchSourceData(
                label=label, url=r.url, requested_url=r.candidate.url, domain=domain_of(r.url),
                title=r.page.title or r.candidate.title, publisher=r.page.publisher or r.candidate.publisher,
                published=r.page.published, source_type=r.candidate.source_type, relevance=relevance(r),
                attribution_required=r.candidate.source_type in ATTRIBUTION_REQUIRED,
                retrieval_status=r.status, excerpt=excerpt,
            )
        )  # fmt: skip
        dispositions.append(CandidateDisposition(url=r.candidate.url, title=r.candidate.title, source_type=r.candidate.source_type, outcome="retrieved", reason=f"{label} (read in call {r.call})"))  # fmt: skip
    return ResearchResult(
        questions=list(questions),
        search_queries=list(queries),
        candidates=dispositions,
        sources=sources,
        facts=facts,
        notes=notes,
        calls=dict(calls),
    )
