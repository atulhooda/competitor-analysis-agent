"""Article draft (Phase 5): brief + research + outline + company profile → the article as
structured content, with inline citation markers ([S1]) for research-backed statements."""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, BeforeValidator, Field

from app.domain.articles import BlockType, SectionKind
from app.prompts.article_common import SECURITY
from app.prompts.fields import cap, lenient_enum, optional_text, truncate

VERSION = "article-draft/1"

SYSTEM = f"""\
You are an expert writer for the company described below. You write a blog article from an \
approved brief, an outline and verified research.

{SECURITY}

Rules:
- Follow the outline section by section. Keep its headings unless a clearer wording says \
the same thing.
- Write for the target audience, in the company's tone of voice, in your own words.
- Citations: right after each statement that relies on a research fact, put its source \
label in square brackets, before the sentence's final punctuation: "... within 72 hours \
[S2]." For several sources: [S1][S3]. Use only labels listed in the research.
- Never invent a source, URL, statistic, quote, study, customer, price or product \
capability. Use only numbers that appear in the research or the brief.
- Facts from competitor or company sources must be attributed in the sentence ("According \
to Acme's documentation, ...") and cited.
- The competitor context is only there so you can differentiate: never copy, closely \
paraphrase or restructure competitor pages, and don't present their claims as facts.
- State only the company facts listed in the company block. Mention the company where it \
genuinely helps the reader (usually once, in the conclusion); no hard sell.
- If the research doesn't support a point, make it as reasoning or practical advice \
without figures, or leave it out.
- Length: about the number of words in the brief.

Output: title, description (one or two sentences), and sections. Each section has kind \
(introduction, body or conclusion), heading (null for the introduction) and blocks. A \
block is a paragraph (text), a list (items, ordered true or false) or a subheading (text).
"""


def max_output_tokens(target_words: int) -> int:
    """Room for the article and its JSON structure (billing is by actual use)."""
    return 6_000 + target_words * 4


def _stripped(value: str | None) -> str | None:
    return value.strip() or None if value else None


def _items(values: list[str]) -> list[str]:
    return [v.strip() for v in values if v and v.strip()][:30]


class BlockOut(BaseModel):
    type: Annotated[BlockType, BeforeValidator(lenient_enum(BlockType, BlockType.PARAGRAPH))] = BlockType.PARAGRAPH  # fmt: skip
    text: Annotated[str | None, AfterValidator(_stripped)] = None
    items: Annotated[list[str], AfterValidator(_items)] = Field(default_factory=list)
    ordered: bool = False


class SectionOut(BaseModel):
    kind: Annotated[SectionKind, BeforeValidator(lenient_enum(SectionKind, SectionKind.BODY))] = SectionKind.BODY  # fmt: skip
    heading: Annotated[str | None, AfterValidator(optional_text(200))] = None
    blocks: Annotated[list[BlockOut], AfterValidator(cap(60))] = Field(default_factory=list)


class ArticleContentOut(BaseModel):
    title: Annotated[str, AfterValidator(truncate(200))]
    description: Annotated[str, AfterValidator(truncate(500))] = ""
    sections: Annotated[list[SectionOut], AfterValidator(cap(20))] = Field(default_factory=list)


def render(*, brief: str, company: str, outline: str, research: str, competitors: str) -> str:
    return "\n\n".join([brief, company, outline, research, competitors, "Write the article."])
