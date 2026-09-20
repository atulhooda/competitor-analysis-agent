"""Originality (Phase 6): deterministic text similarity between an article and the stored
pages of competitors and of your own site. No LLM. It's a similarity signal, not a
plagiarism verdict.

Method (word shingles with common-phrase filtering):

1. Text is normalized: citation markers removed, Unicode folded to ASCII, lowercased,
   punctuation dropped.
2. Every run of ``ORIGINALITY_NGRAM_SIZE`` consecutive words (default 8) is a shingle,
   hashed stably (BLAKE2b; Python's ``hash`` isn't stable across runs).
3. Shingles that can't indicate copying are ignored: those that appear in at least
   ``ORIGINALITY_COMMON_DOC_FREQUENCY`` stored pages (boilerplate, stock phrases, product
   names, industry terminology in fixed expressions), and those made mostly (75%+) of
   stopwords ("one of the most important things to").
4. For each passage (a paragraph or list item of at least ``ORIGINALITY_MIN_PASSAGE_WORDS``
   words), similarity to a page is the share of the passage's remaining shingles found in
   it (containment). The best-matching page and its longest shared run of words are
   recorded.
5. A passage is flagged at ``ORIGINALITY_FLAG_THRESHOLD`` similarity; at
   ``ORIGINALITY_MAX_OVERLAP`` the overlap is severe.

Article level: the highest and the mean passage similarity, and the share of all its
distinctive shingles found anywhere in the corpus. The score goes from 1 (at or below the
flag threshold) to 0 (at the severe level), linearly. The same inputs always give the same
report.
"""

import hashlib
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

from app.config import Settings
from app.domain.articles import ArticleContent, BlockType
from app.domain.quality import OriginalityFlag, OriginalityReport, SimilaritySourceKind
from app.services.article_content import strip_markers, text_blocks
from app.services.checkpoints import digest

ALGORITHM_VERSION = "originality/1"
MAX_DOCUMENT_CHARS = 60_000
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "one",
        "ones",
        "get",
        "gets",
        "got",
        "make",
        "makes",
        "way",
        "ways",
        "thing",
        "things",
        "lot",
        "lots",
        "many",
        "much",
    ]
)


@dataclass(frozen=True)
class OriginalityConfig:
    ngram_size: int
    flag_threshold: float
    max_overlap: float
    common_doc_frequency: int
    min_passage_words: int

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            ngram_size=settings.originality_ngram_size,
            flag_threshold=settings.originality_flag_threshold,
            max_overlap=settings.originality_max_overlap,
            common_doc_frequency=settings.originality_common_doc_frequency,
            min_passage_words=settings.originality_min_passage_words,
        )

    def fingerprint_data(self) -> dict[str, Any]:
        return {"algorithm": ALGORITHM_VERSION, "n": self.ngram_size, "flag": self.flag_threshold, "max": self.max_overlap, "common": self.common_doc_frequency, "min_words": self.min_passage_words}  # fmt: skip


@dataclass(frozen=True)
class CorpusDocument:
    key: int  # content version id
    kind: SimilaritySourceKind
    label: str  # competitor slug, or "company"
    url: str
    content_item_id: int
    text: str


def corpus_fingerprint(documents: Sequence[CorpusDocument]) -> str:
    return digest(sorted((d.key, d.kind.value) for d in documents))


def _words(text: str) -> tuple[list[str], list[str]]:
    """(normalized tokens, the original words they came from), aligned."""
    tokens, originals = [], []
    for word in text.split():
        folded = unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode("ascii").lower()  # fmt: skip
        token = "".join(_TOKEN.findall(folded))
        if token:
            tokens.append(token)
            originals.append(word)
    return tokens, originals


def _shingle(tokens: Sequence[str]) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(tokens).encode(), digest_size=8).digest(), "big")  # fmt: skip


def _shingles(tokens: Sequence[str], n: int) -> list[int]:
    return [_shingle(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class _Index:
    """shingle → the documents containing it."""

    def __init__(self, documents: Sequence[CorpusDocument], n: int) -> None:
        self.postings: dict[int, set[int]] = defaultdict(set)
        for index, document in enumerate(documents):
            tokens, _ = _words(document.text[:MAX_DOCUMENT_CHARS])
            for shingle in set(_shingles(tokens, n)):
                self.postings[shingle].add(index)

    def frequency(self, shingle: int) -> int:
        return len(self.postings.get(shingle, ()))


def analyze(content: ArticleContent, documents: Sequence[CorpusDocument], config: OriginalityConfig) -> OriginalityReport:  # fmt: skip
    n = config.ngram_size
    ordered = sorted(documents, key=lambda d: d.key)  # the report doesn't depend on load order
    index = _Index(ordered, n)
    flags: list[OriginalityFlag] = []
    similarities: list[float] = []
    distinct: set[int] = set()
    matched: set[int] = set()
    ignored: set[int] = set()
    for s, b, i, text in text_blocks(content):
        if content.sections[s].blocks[b].type is BlockType.SUBHEADING:
            continue
        tokens, originals = _words(strip_markers(text))
        if len(tokens) < config.min_passage_words or len(tokens) < n:
            continue
        usable: list[tuple[int, int]] = []  # (position, shingle)
        for position, shingle in enumerate(_shingles(tokens, n)):
            window = tokens[position : position + n]
            stopword_share = sum(1 for t in window if t in _STOPWORDS) / n
            if index.frequency(shingle) >= config.common_doc_frequency or stopword_share >= 0.75:  # fmt: skip
                ignored.add(shingle)
            else:
                usable.append((position, shingle))
        passage = {shingle for _, shingle in usable}
        if not passage:
            continue
        distinct |= passage
        matched |= {h for h in passage if h in index.postings}
        hits: Counter[int] = Counter(doc for h in passage for doc in index.postings.get(h, ()))
        best: tuple[float, int, int, int, int] | None = None  # (similarity, run, -key, doc, start)
        for doc, count in hits.items():
            run, start = _longest_run(usable, doc, index)
            candidate = (count / len(passage), run, -ordered[doc].key, doc, start)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        similarity = round(best[0], 4) if best else 0.0
        similarities.append(similarity)
        if best is not None and similarity >= config.flag_threshold:
            _, run, _, doc, start = best
            document = ordered[doc]
            words = run + n - 1
            flags.append(
                OriginalityFlag(
                    section=s, block=b, item=i, passage=" ".join(originals)[:2_000],
                    source_kind=document.kind, source_label=document.label, url=document.url,
                    content_item_id=document.content_item_id, similarity=similarity,
                    overlap_words=words, overlap_text=" ".join(originals[start : start + words])[:2_000],
                )
            )  # fmt: skip
    max_similarity = max(similarities, default=0.0)
    overall = round(len(matched) / len(distinct), 4) if distinct else 0.0
    signal = max(max_similarity, overall)
    if signal <= config.flag_threshold:
        score = 1.0
    elif signal >= config.max_overlap:
        score = 0.0
    else:
        score = 1 - (signal - config.flag_threshold) / (config.max_overlap - config.flag_threshold)
    return OriginalityReport(
        ngram_size=n,
        documents=len(ordered),
        competitor_documents=sum(1 for d in ordered if d.kind is SimilaritySourceKind.COMPETITOR),
        company_documents=sum(1 for d in ordered if d.kind is SimilaritySourceKind.COMPANY),
        passages_checked=len(similarities),
        common_ngrams_ignored=len(ignored),
        max_similarity=max_similarity,
        avg_similarity=round(statistics.fmean(similarities), 4) if similarities else 0.0,
        overall_overlap=overall,
        flagged=flags,
        severe=max_similarity >= config.max_overlap,
        score=round(score, 4),
        corpus_fingerprint=corpus_fingerprint(ordered),
    )


def _longest_run(usable: Sequence[tuple[int, int]], doc: int, index: _Index) -> tuple[int, int]:
    """(length in shingles, start word position) of the longest run of consecutive passage
    shingles all found in ``doc``."""
    best_length = best_start = length = start = 0
    previous = -2
    for position, shingle in usable:
        if doc in index.postings.get(shingle, ()):
            if position == previous + 1 and length:
                length += 1
            else:
                length, start = 1, position
            previous = position
            if length > best_length:
                best_length, best_start = length, start
        else:
            length = 0
    return best_length, best_start
