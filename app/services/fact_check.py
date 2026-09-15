"""Fact-checking (Phase 6): every cited claim against the source it cites, and a search for
factual claims that cite nothing.

Cited claims come from ``article_citations`` (claim → source). For each (claim, source) pair:

1. Gemini judges the claim against the source's stored research notes (facts and verbatim
   excerpts). A supporting or contradicting verdict needs an evidence quote, and the quote
   must be found in those notes (checked in code). Otherwise the claim isn't settled.
2. Unsettled claims are checked against the page itself: Gemini's URL context tool re-reads
   the source (the URL is screened by the Phase 1 SSRF guard first; this process fetches
   nothing). The tool must report the page as retrieved, and a verdict still needs evidence.
3. Whatever remains unsettled is ``unsupported``: agreement alone never verifies a claim.

Uncited claims: sentences without citations that look factual (numbers, percentages,
dates, quantities, research references, legal statements, named organizations) are
extracted in code; Gemini classifies which of them need a source. They are stored, never
deleted: the revision step decides what to do with them.

An identical (claim, source) pair keeps its earlier verdict (same prompt and model), so a
revision only re-checks what changed. Only verdicts a model actually decided are reused
(``model`` is set on them); fallbacks such as "not re-read" are decided again next time. All
ratios are computed here from the verdicts.
"""

import re
import unicodedata
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Self

from app.config import Settings
from app.crawling.netguard import Resolver
from app.domain.analysis import LLMPurpose
from app.domain.articles import ArticleContent
from app.domain.quality import (
    ClaimResult,
    ClaimVerdict,
    FactCheckMetrics,
    FactCheckReport,
    SourceCheck,
    UncitedClaim,
)
from app.llm import LLMRequest, LLMResponseError, ReasoningEffort
from app.prompts import fact_check as prompt
from app.services.article_content import citations, split_sentences, strip_markers, text_blocks
from app.services.checkpoints import digest
from app.services.llm_usage import BudgetedLLM
from app.services.research import match_retrievals, safe_public_url

INSUFFICIENT = "insufficient"  # internal: not settled yet
_RANK = {ClaimVerdict.SUPPORTED: 0, ClaimVerdict.PARTIAL: 1, ClaimVerdict.CONTRADICTED: 2, ClaimVerdict.UNSUPPORTED: 3}  # fmt: skip
_WORD = re.compile(r"[a-z0-9]+")
_SIGNALS: tuple[tuple[str, int, re.Pattern[str]], ...] = (
    ("percentage", 3, re.compile(r"\d\s?%|\bper ?cent\b", re.IGNORECASE)),
    ("number", 2, re.compile(r"\d")),
    (
        "date",
        2,
        re.compile(
            r"\b(?:january|february|march|april|june|july|august|september|october|november|december|may\s+\d)",
            re.IGNORECASE,
        ),
    ),
    (
        "quantity",
        1,
        re.compile(
            r"\b(?:million|billion|thousand|majority|minority|twice|double|triple|half of)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "research_reference",
        2,
        re.compile(
            r"\b(?:study|studies|survey|surveys|report|reports|research|researchers|analysis|statistics|according to|found that|shows that|estimates?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "legal",
        1,
        re.compile(
            r"\b(?:law|laws|regulation|regulations|directive|act|requires|mandates?|court|ruling|judgment|fines?|penalt(?:y|ies))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "named_entity",
        1,
        re.compile(
            r"(?:\b[A-Z]{3,}\b|(?<=\s)[A-Z][a-z]+(?:\s+(?:of|for|and|the)?\s*[A-Z][a-zA-Z]+)+)"
        ),
    ),
)


@dataclass(frozen=True)
class FactCheckConfig:
    model: str
    reasoning_effort: ReasoningEffort
    batch_size: int
    max_rereads: int
    max_uncited: int

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            model=settings.quality_model,
            reasoning_effort=settings.quality_reasoning_effort,
            batch_size=settings.fact_check_batch_size,
            max_rereads=settings.fact_check_max_rereads,
            max_uncited=settings.fact_check_max_uncited_candidates,
        )

    def fingerprint_data(self) -> dict[str, Any]:
        return {"model": self.model, "reasoning": self.reasoning_effort, "batch": self.batch_size, "rereads": self.max_rereads, "uncited": self.max_uncited}  # fmt: skip


@dataclass(frozen=True)
class SourceMaterial:
    source_id: int
    label: str
    url: str
    title: str | None
    publisher: str | None
    facts: tuple[tuple[str, str | None], ...]  # (statement, verbatim excerpt)
    excerpt: str | None

    def text(self) -> str:
        """The stored notes an evidence quote must come from."""
        parts = [self.title or "", self.excerpt or ""]
        for statement, quote in self.facts:
            parts += [statement, quote or ""]
        return "\n".join(p for p in parts if p)


@dataclass(frozen=True)
class CitedClaim:
    key: str
    section: int
    block: int
    item: int | None
    claim: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class UncitedCandidate:
    key: str
    section: int
    block: int
    item: int | None
    sentence: str
    signals: tuple[str, ...]


@dataclass
class PairResult:
    claim: CitedClaim
    label: str
    source_id: int | None
    verdict: str
    explanation: str
    evidence: str | None = None
    evidence_verified: bool = False
    confidence: float = 0.0
    reread: bool = False
    reused_from: int | None = None
    model: str | None = None


@dataclass
class UncitedResult:
    candidate: UncitedCandidate
    verdict: ClaimVerdict
    claim_type: str
    reason: str
    reused_from: int | None = None
    model: str | None = None


@dataclass(frozen=True)
class PriorCheck:
    id: int
    verdict: str
    explanation: str
    evidence: str | None
    evidence_verified: bool
    confidence: float | None
    reread: bool
    claim_type: str | None
    model: str | None = None


@dataclass
class FactCheckOutcome:
    report: FactCheckReport
    pairs: list[PairResult]
    uncited: list[UncitedResult]
    notes: list[str] = field(default_factory=list)


def claim_key(text: str) -> str:
    return digest(" ".join(_WORD.findall(text.lower())))


def _normalize(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text).lower().replace("\u2019", "'")
    return _WORD.findall(folded)


def evidence_found(evidence: str | None, material: str) -> bool:
    """Whether the quote comes from the source notes: found word for word, or with at least
    80% of it as one contiguous run (tolerating punctuation and small edits)."""
    quote = _normalize(evidence or "")
    if len(quote) < 3:
        return False
    notes = _normalize(material)
    if not notes:
        return False
    match = SequenceMatcher(None, quote, notes, autojunk=False).find_longest_match(0, len(quote), 0, len(notes))  # fmt: skip
    return match.size >= 0.8 * len(quote)


def cited_claims(rows: Sequence[tuple[int, int, int | None, str, str]]) -> list[CitedClaim]:
    """(section, block, item, claim, label) rows → one claim per location and text."""
    grouped: dict[tuple[int, int, int | None, str], list[str]] = defaultdict(list)
    for section, block, item, claim, label in rows:
        labels = grouped[(section, block, item, claim)]
        if label not in labels:
            labels.append(label)
    return [CitedClaim(claim_key(claim), s, b, i, claim, tuple(labels)) for (s, b, i, claim), labels in grouped.items()]  # fmt: skip


def uncited_candidates(content: ArticleContent, *, limit: int) -> list[UncitedCandidate]:
    """Factual-looking sentences that cite nothing, strongest signals first."""
    found: list[tuple[int, int, UncitedCandidate]] = []
    order = 0
    for s, b, i, text in text_blocks(content):
        block = content.sections[s].blocks[b]
        if block.type.value == "subheading":
            continue
        for sentence in split_sentences(text):
            if "[S" in sentence:  # cited: checked against its sources instead
                continue
            plain = strip_markers(sentence).strip()
            if len(plain.split()) < 6 or plain.endswith("?"):
                continue
            signals = [(name, weight) for name, weight, pattern in _SIGNALS if pattern.search(plain)]  # fmt: skip
            if not signals:
                continue
            strength = sum(weight for _, weight in signals)
            found.append((-strength, order, UncitedCandidate(claim_key(plain), s, b, i, plain, tuple(name for name, _ in signals))))  # fmt: skip
            order += 1
    chosen = sorted(found)[:limit]
    return [c for _, _, c in sorted(chosen, key=lambda t: t[1])]


def integrity_problems(content: ArticleContent, rows: Sequence[tuple[int, int, int | None, str, str]], labels: set[str]) -> list[str]:  # fmt: skip
    """Citation integrity: markers in the text, stored claim → source records, and stored
    sources must agree."""
    marked = citations(content)
    expected = {(c.section, c.block, c.item, label) for c in marked for label in c.labels}
    stored = {(s, b, i, label) for s, b, i, _, label in rows}
    problems = []
    unknown = sorted({label for c in marked for label in c.labels if label not in labels})
    if unknown:
        problems.append(f"citation(s) {', '.join(unknown)} don't match a stored source")
    if expected - stored:
        problems.append(f"{len(expected - stored)} citation(s) in the text have no stored claim → source record")  # fmt: skip
    if stored - expected:
        problems.append(f"{len(stored - expected)} stored citation record(s) don't match the text")  # fmt: skip
    return problems


def claim_verdict(verdicts: Sequence[str]) -> ClaimVerdict:
    """A claim is as good as its best-supported citation."""
    known = [ClaimVerdict(v) for v in verdicts if v in {k.value for k in _RANK}]
    return min(known, key=lambda v: _RANK[v]) if known else ClaimVerdict.UNSUPPORTED


def metrics(claims: Sequence[ClaimResult], uncited: Sequence[UncitedClaim], *, problems: list[str], rereads: int) -> FactCheckMetrics:  # fmt: skip
    counts = {v: sum(1 for c in claims if c.verdict is v) for v in _RANK}
    cited = len(claims)
    uncited_factual = sum(1 for u in uncited if u.verdict is ClaimVerdict.NEEDS_VERIFICATION)
    factual = cited + uncited_factual

    def share(n: int) -> float:
        return round(n / cited, 4) if cited else 0.0

    return FactCheckMetrics(
        cited_claims=cited,
        supported=counts[ClaimVerdict.SUPPORTED],
        partial=counts[ClaimVerdict.PARTIAL],
        unsupported=counts[ClaimVerdict.UNSUPPORTED],
        contradicted=counts[ClaimVerdict.CONTRADICTED],
        uncited_candidates=len(uncited),
        uncited_factual=uncited_factual,
        factual_claims=factual,
        citation_coverage=round(cited / factual, 4) if factual else 1.0,
        supported_claim_ratio=share(counts[ClaimVerdict.SUPPORTED]),
        partial_claim_ratio=share(counts[ClaimVerdict.PARTIAL]),
        unsupported_claim_ratio=share(counts[ClaimVerdict.UNSUPPORTED]),
        contradicted_claim_ratio=share(counts[ClaimVerdict.CONTRADICTED]),
        uncited_factual_claim_ratio=round(uncited_factual / factual, 4) if factual else 0.0,
        rereads=rereads,
        integrity_ok=not problems,
        integrity_problems=problems,
    )


async def run_fact_check(
    llm: BudgetedLLM,
    *,
    content: ArticleContent,
    claims: Sequence[CitedClaim],
    sources: Mapping[str, SourceMaterial],
    config: FactCheckConfig,
    resolver: Resolver,
    integrity: list[str],
    prior_pairs: Mapping[tuple[str, int], PriorCheck],
    prior_uncited: Mapping[str, PriorCheck],
) -> FactCheckOutcome:
    notes: list[str] = []
    pairs: list[PairResult] = []
    pending: list[PairResult] = []
    for claim in claims:
        for label in claim.labels:
            source = sources.get(label)
            if source is None:
                pairs.append(PairResult(claim, label, None, ClaimVerdict.UNSUPPORTED.value, f"[{label}] isn't a stored source"))  # fmt: skip
                continue
            prior = prior_pairs.get((claim.key, source.source_id))
            if prior is not None:
                pairs.append(PairResult(claim, label, source.source_id, prior.verdict, prior.explanation, prior.evidence, prior.evidence_verified, prior.confidence or 0.0, prior.reread, prior.id, prior.model))  # fmt: skip
                continue
            result = PairResult(claim, label, source.source_id, INSUFFICIENT, "")
            pairs.append(result)
            pending.append(result)

    pending.sort(key=lambda p: (p.label, p.claim.section, p.claim.block))
    await _check_batches(llm, pending, sources, config, notes)
    rereads = await _reread(llm, [p for p in pending if p.verdict == INSUFFICIENT], sources, config, resolver, notes)  # fmt: skip
    for p in pairs:
        if p.verdict == INSUFFICIENT:
            p.verdict, p.model = ClaimVerdict.UNSUPPORTED.value, None
            p.explanation = (p.explanation + " " if p.explanation else "") + "Not settled by the stored research, and the page wasn't re-read (FACT_CHECK_MAX_REREADS)."  # fmt: skip

    uncited = await _classify(llm, uncited_candidates(content, limit=config.max_uncited), config, prior_uncited, notes)  # fmt: skip
    results = _claim_results(claims, pairs)
    uncited_views = [
        UncitedClaim(
            key=u.candidate.key,
            section=u.candidate.section,
            block=u.candidate.block,
            item=u.candidate.item,
            sentence=u.candidate.sentence,
            signals=list(u.candidate.signals),
            verdict=u.verdict,
            claim_type=u.claim_type,
            reason=u.reason,
            reused=u.reused_from is not None,
        )
        for u in uncited
    ]
    report = FactCheckReport(
        claims=results,
        uncited=uncited_views,
        metrics=metrics(results, uncited_views, problems=integrity, rereads=rereads),
        notes=notes,
    )
    return FactCheckOutcome(report, pairs, uncited, notes)


def _claim_results(claims: Sequence[CitedClaim], pairs: Sequence[PairResult]) -> list[ClaimResult]:  # fmt: skip
    by_claim: dict[tuple[int, int, int | None, str], list[PairResult]] = defaultdict(list)
    for p in pairs:
        by_claim[(p.claim.section, p.claim.block, p.claim.item, p.claim.claim)].append(p)
    results = []
    for c in claims:
        checks = by_claim[(c.section, c.block, c.item, c.claim)]
        results.append(
            ClaimResult(
                key=c.key, section=c.section, block=c.block, item=c.item, claim=c.claim, labels=list(c.labels),
                verdict=claim_verdict([p.verdict for p in checks]),
                checks=[SourceCheck(label=p.label, source_id=p.source_id or 0, verdict=ClaimVerdict(p.verdict), explanation=p.explanation, evidence=p.evidence, evidence_verified=p.evidence_verified, confidence=p.confidence, reread=p.reread, reused=p.reused_from is not None) for p in checks],
            )
        )  # fmt: skip
    return results


async def _check_batches(llm: BudgetedLLM, pending: Sequence[PairResult], sources: Mapping[str, SourceMaterial], config: FactCheckConfig, notes: list[str]) -> None:  # fmt: skip
    queue = deque(list(pending[i : i + config.batch_size]) for i in range(0, len(pending), config.batch_size))  # fmt: skip
    while queue:
        batch = queue.popleft()
        labels = list(dict.fromkeys(p.label for p in batch))
        blocks = [prompt.source_block(label, title=sources[label].title, url=sources[label].url, publisher=sources[label].publisher, facts=sources[label].facts, excerpt=sources[label].excerpt) for label in labels]  # fmt: skip
        ids = {f"C{i}": p for i, p in enumerate(batch, start=1)}
        request = LLMRequest(
            prompt=prompt.render_check(
                blocks, [(cid, p.label, p.claim.claim) for cid, p in ids.items()]
            ),
            system=prompt.CHECK_SYSTEM,
            model=config.model,
            max_output_tokens=prompt.max_output_tokens(len(batch)),
            reasoning_effort=config.reasoning_effort,
        )
        try:
            response = await llm.structured(request, prompt.FactCheckOut, purpose=LLMPurpose.FACT_CHECK, prompt_version=prompt.VERSION, items=len(batch))  # fmt: skip
        except LLMResponseError:
            if len(batch) > 1:  # retry in halves
                middle = len(batch) // 2
                queue.appendleft(batch[middle:])
                queue.appendleft(batch[:middle])
            else:
                batch[0].verdict = ClaimVerdict.UNSUPPORTED.value
                batch[0].explanation = "The model returned unusable output for this claim."
                notes.append(f"a fact-check call returned unusable output for: {batch[0].claim.claim[:80]}")  # fmt: skip
            continue
        answers = {c.claim_id.strip().upper(): c for c in response.data.checks}
        for cid, p in ids.items():
            answer = answers.get(cid)
            if answer is None:
                p.explanation = "The model gave no verdict."
                continue
            p.explanation, p.confidence = answer.explanation, answer.confidence
            if answer.verdict in (ClaimVerdict.SUPPORTED.value, ClaimVerdict.PARTIAL.value, ClaimVerdict.CONTRADICTED.value):  # fmt: skip
                if evidence_found(answer.evidence, sources[p.label].text()):
                    p.verdict, p.evidence, p.evidence_verified = answer.verdict, answer.evidence, True  # fmt: skip
                    p.model = response.raw.model
                else:  # a verdict it can't ground in the notes doesn't count
                    p.verdict = INSUFFICIENT
                    p.explanation = f"Claimed {answer.verdict}, but the quoted evidence isn't in the stored notes. {answer.explanation}"  # fmt: skip
            elif answer.verdict == ClaimVerdict.UNSUPPORTED.value:
                p.verdict, p.model = ClaimVerdict.UNSUPPORTED.value, response.raw.model
            else:
                p.verdict = INSUFFICIENT


async def _reread(llm: BudgetedLLM, unsettled: Sequence[PairResult], sources: Mapping[str, SourceMaterial], config: FactCheckConfig, resolver: Resolver, notes: list[str]) -> int:  # fmt: skip
    by_source: dict[str, list[PairResult]] = defaultdict(list)
    for p in unsettled:
        by_source[p.label].append(p)
    rereads = 0
    for label, group in sorted(by_source.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if rereads >= config.max_rereads:
            break
        source = sources[label]
        url, reason = await safe_public_url(source.url, resolver)
        if url is None:
            for p in group:
                p.verdict, p.explanation = ClaimVerdict.UNSUPPORTED.value, f"The source URL failed the safety check ({reason}), so it wasn't re-read."  # fmt: skip
            continue
        rereads += 1
        ids = {f"C{i}": p for i, p in enumerate(group, start=1)}
        request = LLMRequest(
            prompt=prompt.render_reread(
                url, label, source.title, [(cid, label, p.claim.claim) for cid, p in ids.items()]
            ),
            system=prompt.REREAD_SYSTEM,
            model=config.model,
            max_output_tokens=prompt.max_output_tokens(len(group)),
            reasoning_effort=config.reasoning_effort,
            tools=("url_context",),
        )
        try:
            response = await llm.structured(request, prompt.FactCheckOut, purpose=LLMPurpose.FACT_CHECK, prompt_version=prompt.VERSION, items=len(group))  # fmt: skip
        except LLMResponseError:
            for p in group:
                p.verdict, p.explanation = ClaimVerdict.UNSUPPORTED.value, "Re-reading the source returned unusable output."  # fmt: skip
            continue
        final_url, status = match_retrievals([url], response.raw.grounding.retrieved_urls)[url]
        if status != "success":
            notes.append(f"{label} couldn't be re-read ({status})")
            for p in group:
                p.verdict, p.explanation, p.reread = ClaimVerdict.UNSUPPORTED.value, f"The page couldn't be re-read (URL context status: {status}).", True  # fmt: skip
            continue
        answers = {c.claim_id.strip().upper(): c for c in response.data.checks}
        for cid, p in ids.items():
            answer = answers.get(cid)
            p.reread, p.model = True, response.raw.model if answer is not None else None
            grounded = answer is not None and answer.verdict in (ClaimVerdict.SUPPORTED.value, ClaimVerdict.PARTIAL.value, ClaimVerdict.CONTRADICTED.value) and bool((answer.evidence or "").strip())  # fmt: skip
            if answer is not None and grounded:
                p.verdict, p.evidence, p.confidence = answer.verdict, answer.evidence, answer.confidence  # fmt: skip
                p.explanation = f"{answer.explanation} (re-read from {final_url})"
            else:
                p.verdict = ClaimVerdict.UNSUPPORTED.value
                p.explanation = (answer.explanation if answer else "No verdict on re-reading.") + " The page doesn't support it with quotable evidence."  # fmt: skip
                p.confidence = answer.confidence if answer else 0.0
    return rereads


async def _classify(llm: BudgetedLLM, candidates: Sequence[UncitedCandidate], config: FactCheckConfig, prior: Mapping[str, PriorCheck], notes: list[str]) -> list[UncitedResult]:  # fmt: skip
    results: list[UncitedResult] = []
    todo: list[UncitedCandidate] = []
    for c in candidates:
        cached = prior.get(c.key)
        if cached is not None:
            results.append(UncitedResult(c, ClaimVerdict(cached.verdict), cached.claim_type or "other", cached.explanation, cached.id, cached.model))  # fmt: skip
        else:
            todo.append(c)
    if todo:
        ids = {f"U{i}": c for i, c in enumerate(todo, start=1)}
        request = LLMRequest(
            prompt=prompt.render_classify([(sid, c.sentence) for sid, c in ids.items()]),
            system=prompt.CLASSIFY_SYSTEM,
            model=config.model,
            max_output_tokens=prompt.max_output_tokens(len(todo)),
            reasoning_effort=config.reasoning_effort,
        )
        try:
            response = await llm.structured(request, prompt.ClassifyOut, purpose=LLMPurpose.CLAIM_CLASSIFICATION, prompt_version=prompt.VERSION, items=len(todo))  # fmt: skip
            answers = {a.sentence_id.strip().upper(): a for a in response.data.sentences}
            model: str | None = response.raw.model
        except LLMResponseError:
            answers, model = {}, None
            notes.append("uncited-claim classification returned unusable output: sentences with numbers are treated as needing a source")  # fmt: skip
        for sid, c in ids.items():
            answer = answers.get(sid)
            if answer is None:  # conservative: anything with a figure needs a source
                needs = bool({"number", "percentage", "date"} & set(c.signals))
                results.append(UncitedResult(c, ClaimVerdict.NEEDS_VERIFICATION if needs else ClaimVerdict.NOT_REQUIRED, "statistic" if needs else "other", "not classified; decided by its signals", None, None))  # fmt: skip
            else:
                verdict = ClaimVerdict.NEEDS_VERIFICATION if answer.requires_citation else ClaimVerdict.NOT_REQUIRED  # fmt: skip
                results.append(UncitedResult(c, verdict, answer.claim_type, answer.reason, None, model))  # fmt: skip
    order = {c.key: i for i, c in enumerate(candidates)}
    return sorted(results, key=lambda r: order[r.candidate.key])
