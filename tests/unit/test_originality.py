"""Originality (Phase 6): deterministic shingle similarity against stored pages."""

from app.domain.articles import ArticleContent
from app.domain.quality import SimilaritySourceKind
from app.services.originality import CorpusDocument, OriginalityConfig, analyze, corpus_fingerprint

CONFIG = OriginalityConfig(ngram_size=8, flag_threshold=0.25, max_overlap=0.5, common_doc_frequency=3, min_passage_words=12)  # fmt: skip
OURS = (
    "Small support teams should route billing disputes to a named person within one working "
    "day, and keep a written log of every case the agent could not settle without help [S1]."
)
OTHER = (
    "Quarterly roadmap reviews help product managers align engineering capacity with the "
    "commitments sales teams made to enterprise customers during renewal negotiations."
)


def article(*paragraphs: str) -> ArticleContent:
    return ArticleContent.model_validate(
        {
            "title": "Support playbook",
            "description": "A guide.",
            "sections": [
                {
                    "kind": "body",
                    "heading": "Routing",
                    "blocks": [{"type": "paragraph", "text": p} for p in paragraphs],
                }
            ],
        }
    )


def page(key: int, text: str, *, kind: SimilaritySourceKind = SimilaritySourceKind.COMPETITOR, label: str = "acme") -> CorpusDocument:  # fmt: skip
    return CorpusDocument(key=key, kind=kind, label=label, url=f"https://{label}.test/p/{key}", content_item_id=100 + key, text=text)  # fmt: skip


def test_a_copied_passage_is_flagged_with_where_it_came_from() -> None:
    copied = OURS.replace(" [S1]", "")
    report = analyze(article(OURS, OTHER), [page(1, f"Intro text here. {copied} More text."), page(2, "Unrelated page about pricing plans and seats.")], CONFIG)  # fmt: skip

    assert report.passages_checked == 2
    assert report.max_similarity == 1.0
    assert report.severe
    assert report.score == 0.0
    [flag] = report.flagged
    assert (flag.section, flag.block, flag.source_label, flag.content_item_id) == (
        0,
        0,
        "acme",
        101,
    )
    assert flag.url == "https://acme.test/p/1"
    assert flag.source_kind is SimilaritySourceKind.COMPETITOR
    assert flag.overlap_words == len(copied.split())
    assert flag.overlap_text == copied
    assert "[S1]" not in flag.passage  # citation markers aren't text


def test_original_text_scores_one() -> None:
    report = analyze(article(OURS), [page(1, OTHER)], CONFIG)
    assert report.flagged == []
    assert report.max_similarity == 0.0
    assert report.score == 1.0
    assert not report.severe
    assert report.overall_overlap == 0.0


def test_partial_overlap_is_flagged_but_not_severe() -> None:
    half = " ".join(OURS.split()[:14])  # the first half of the passage
    report = analyze(article(OURS), [page(1, f"Other words. {half} and then something else entirely.")], CONFIG)  # fmt: skip
    [flag] = report.flagged
    assert 0.25 <= flag.similarity < 0.5
    assert not report.severe
    assert 0 < report.score < 1


def test_phrases_on_many_pages_are_common_not_copied() -> None:
    copied = OURS.replace(" [S1]", "")
    corpus = [page(i, f"Page {i}. {copied}") for i in range(1, 4)]  # on 3 pages: boilerplate
    report = analyze(article(OURS), corpus, CONFIG)
    assert report.flagged == []
    assert report.common_ngrams_ignored > 0
    assert analyze(article(OURS), corpus, OriginalityConfig(**{**CONFIG.__dict__, "common_doc_frequency": 4})).severe  # fmt: skip


def test_stopword_runs_are_ignored() -> None:
    filler = "it is one of the most important things to do and it is one of the things that we can do for them"  # fmt: skip
    report = analyze(article(filler), [page(1, filler)], CONFIG)
    assert report.flagged == []
    assert report.passages_checked == 0


def test_short_passages_are_skipped() -> None:
    report = analyze(article("Route billing disputes to a named person."), [page(1, "Route billing disputes to a named person.")], CONFIG)  # fmt: skip
    assert report.passages_checked == 0
    assert report.score == 1.0


def test_company_pages_are_their_own_kind() -> None:
    copied = OURS.replace(" [S1]", "")
    report = analyze(article(OURS), [page(1, copied, kind=SimilaritySourceKind.COMPANY, label="company")], CONFIG)  # fmt: skip
    assert report.company_documents == 1
    assert report.competitor_documents == 0
    assert report.flagged[0].source_kind is SimilaritySourceKind.COMPANY


def test_the_report_is_deterministic() -> None:
    copied = OURS.replace(" [S1]", "")
    corpus = [page(1, copied), page(2, OTHER), page(3, copied + " " + OTHER)]
    first = analyze(article(OURS, OTHER), corpus, CONFIG)
    second = analyze(article(OURS, OTHER), list(reversed(corpus)), CONFIG)
    assert first == second
    assert first.flagged[0].content_item_id == 101  # ties go to the earliest page
    assert corpus_fingerprint(corpus) == corpus_fingerprint(list(reversed(corpus)))
    assert corpus_fingerprint(corpus) != corpus_fingerprint(corpus[:2])


def test_smaller_shingles_are_stricter() -> None:
    fragment = " ".join(OURS.split()[3:9])  # a six-word run
    corpus = [page(1, f"Unrelated start. {fragment} unrelated end.")]
    assert analyze(article(OURS), corpus, CONFIG).flagged == []
    small = OriginalityConfig(**{**CONFIG.__dict__, "ngram_size": 4, "flag_threshold": 0.1})
    assert analyze(article(OURS), corpus, small).flagged
