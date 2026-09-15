"""Quality judge (Phase 6): a rubric review of one article version. The model scores each
dimension 1-5 with reasons and issues; it doesn't compute an overall score, and the
deterministic metrics and fact-check results it's given are authoritative."""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, BeforeValidator, Field

from app.prompts.article_common import SECURITY
from app.prompts.fields import texts, truncate

VERSION = "quality-judge/1"
MAX_OUTPUT_TOKENS = 4_000
DIMENSIONS = (
    "factual_support",
    "audience_value",
    "clarity",
    "structure",
    "originality",
    "search_intent_alignment",
    "strategic_alignment",
    "readability",
)

SYSTEM = f"""\
You are a senior editor reviewing an article before it can be marked ready for publication.

{SECURITY}

Score each dimension from 1 to 5:
- factual_support: are claims backed by the cited sources? Use the fact-check results; don't \
re-check facts yourself. 5 = every claim supported; 1 = contradicted or many unsupported claims.
- audience_value: does it give the target reader concrete, useful help? 5 = specific and \
actionable; 1 = generic filler.
- clarity: is it easy to follow, precise and free of jargon the reader wouldn't know?
- structure: do headings and sections flow logically, with a clear introduction and conclusion?
- originality: does it offer its own angle and insight rather than restating competitors? Use \
the originality results.
- search_intent_alignment: does it serve the reader's search intent from the brief?
- strategic_alignment: does it follow the brief's angle and differentiation, and connect to \
the company where it helps the reader, without a hard sell?
- readability: sentence length, paragraphing and scannability for this audience.

Anchors: 5 excellent, 4 good (minor issues), 3 acceptable (clear issues), 2 weak, 1 poor.

Rules:
- The metrics and fact-check results are computed by the system and are authoritative: don't \
recompute or contradict them.
- Base every judgment on the article, the brief, the research and those results. Don't invent \
evidence, facts or problems.
- For each dimension give a one- or two-sentence explanation and the specific issues (empty if \
none), each pointing at the passage or section concerned.
- Return all eight dimensions. Don't give an overall score.
"""


def _score(value: object) -> object:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return min(max(round(value), 1), 5)
    return value


class DimensionOut(BaseModel):
    dimension: Annotated[str, AfterValidator(truncate(40))]
    score: Annotated[int, Field(ge=1, le=5), BeforeValidator(_score)]
    explanation: Annotated[str, AfterValidator(truncate(500))]
    issues: Annotated[list[str], AfterValidator(texts(6, 300))] = Field(default_factory=list)


class JudgeOut(BaseModel):
    dimensions: list[DimensionOut] = Field(default_factory=list)
    summary: Annotated[str, AfterValidator(truncate(600))] = ""


def render(*, brief: str, research: str, article: str, fact_check: str, originality: str, metrics: str) -> str:  # fmt: skip
    return "\n\n".join([brief, research, "Fact-check results (authoritative):", fact_check, "Originality results (authoritative):", originality, "Deterministic metrics (authoritative):", metrics, article, "Review the article."])  # fmt: skip
