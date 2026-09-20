"""Editorial pass (Phase 5): the draft → an edited article, the editor's change notes, and
flags for claims it removed, qualified or couldn't support. It adds no facts."""

from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, BeforeValidator, Field

from app.prompts.article_common import SECURITY
from app.prompts.article_draft import ArticleContentOut
from app.prompts.fields import cap, optional_text, texts, truncate

VERSION = "article-edit/1"
FLAG_ACTIONS = ("removed", "qualified", "flagged")

SYSTEM = f"""\
You are the editor. You improve a draft blog article before a person reviews it.

{SECURITY}

Check and fix: coherence, repetition, unsupported claims, fit for the target audience, \
tone of voice, clarity, logical flow, source attribution, and alignment with the brief and \
the outline.

Rules:
- You may rewrite, reorder, merge or cut. You must not add facts, numbers, sources, \
quotes, named companies or claims that aren't in the draft or the research.
- Keep valid citations ([S1]) with the statements they support, before the sentence's \
final punctuation. Never invent a label.
- An unsupported claim is a factual statement with no citation that isn't common \
knowledge, a number that isn't in the research or the brief, or an unattributed competitor \
claim. Remove it, qualify it (make it clearly an opinion or a general observation, without \
figures), or keep it and flag it. Record each one in flags, with the action you took.
- State only the company facts in the company block.
- Keep roughly the brief's length (within 20%) and the outline's structure.
- Record your main changes as short notes in changes.

Output: article (the same structure as the draft), changes (at most 12 notes), and flags \
(each with excerpt, issue, action and note). issue is one of unsupported_claim, \
missing_citation, unattributed_competitor_claim, off_brief, tone, other; action is one of \
removed, qualified, flagged.
"""


def _action(value: Any) -> Any:
    return value if value in FLAG_ACTIONS else "flagged"


class FlagOut(BaseModel):
    excerpt: Annotated[str, AfterValidator(truncate(300))] = ""
    issue: Annotated[str, AfterValidator(truncate(40))] = "other"
    action: Annotated[str, BeforeValidator(_action)] = "flagged"
    note: Annotated[str | None, AfterValidator(optional_text(300))] = None


class EditOut(BaseModel):
    article: ArticleContentOut
    changes: Annotated[list[str], AfterValidator(texts(12, 300))] = Field(default_factory=list)
    flags: Annotated[list[FlagOut], AfterValidator(cap(40))] = Field(default_factory=list)


def render(*, brief: str, company: str, outline: str, research: str, draft: str) -> str:
    return "\n\n".join([brief, company, outline, research, draft, "Edit the draft."])
