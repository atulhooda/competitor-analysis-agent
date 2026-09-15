"""Article outline (Phase 5): the approved brief + verified research → a structured outline."""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

from app.prompts.article_common import SECURITY
from app.prompts.fields import cap, optional_text, texts, truncate

VERSION = "article-outline/1"
MAX_OUTPUT_TOKENS = 6_000

SYSTEM = f"""\
You are a senior content strategist. You outline a blog article for a company, from an \
approved brief and verified research.

{SECURITY}

The outline must follow the brief: its angle, target audience, search intent, content \
type, key points, the competitor weaknesses to address and the differentiation strategy. \
Don't drift into a generic article on the topic, and don't mirror the structure of \
competitor pages: the competitor context shows what already exists so you can do \
something different and more useful.

Rules:
- Use research facts only through their source labels (S1, S2, ...): list in source_ids \
the sources a section relies on. Never invent a label.
- Don't plan claims the research doesn't support. Plan them as reasoning or practical \
advice without figures, or leave them out.
- Facts from competitor or company sources can only be used with attribution.
- Every section needs a clear purpose and concrete value for the target audience.
- 3 to 7 body sections, plus an introduction and a conclusion. The conclusion may connect \
the topic to the company's products where that genuinely helps the reader.

Fields: title (at most 14 words), description (one or two sentences on what the reader \
gets), introduction, sections and conclusion. Each has heading (null for the \
introduction), purpose, key_points (2-5), source_ids and audience_value.
"""


class OutlineSectionOut(BaseModel):
    heading: Annotated[str | None, AfterValidator(optional_text(160))] = None
    purpose: Annotated[str, AfterValidator(truncate(400))]
    key_points: Annotated[list[str], AfterValidator(texts(6, 300))] = Field(default_factory=list)
    source_ids: Annotated[list[str], AfterValidator(texts(12, 8))] = Field(default_factory=list)
    audience_value: Annotated[str, AfterValidator(truncate(300))] = ""


class OutlineOut(BaseModel):
    title: Annotated[str, AfterValidator(truncate(200))]
    description: Annotated[str, AfterValidator(truncate(400))]
    introduction: OutlineSectionOut
    sections: Annotated[list[OutlineSectionOut], AfterValidator(cap(10))]
    conclusion: OutlineSectionOut


def render(*, brief: str, company: str, research: str, competitors: str) -> str:
    return "\n\n".join([brief, company, research, competitors, "Write the outline."])
