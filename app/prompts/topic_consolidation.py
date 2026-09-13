"""Find duplicate topics in the taxonomy (same subject, different names)."""

from collections.abc import Sequence
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

from app.prompts.fields import cap, truncate

VERSION = "topic-consolidation/2"

SYSTEM = """\
You maintain the topic taxonomy of a competitive content analysis system. Find topics that \
are duplicates: the same subject under different names (synonyms, abbreviations, singular \
and plural, word order), e.g. "Agentic AI" and "AI agents", or "Customer service" and \
"Customer support".

Rules:
- Merge only topics that mean the same subject. Related is not the same: "Pricing" and \
"Billing" are related; "AI" and "AI agents" differ in scope. When unsure, don't merge.
- For each group, choose as target the clearest, most general name, preferring the topic \
with more items.
- Use the ids exactly as listed. Each id may appear in at most one group.
- Return an empty list when there are no duplicates.
"""


class MergeGroupOut(BaseModel):
    target_id: str = Field(description='Id of the topic to keep, e.g. "T3"')
    source_ids: Annotated[list[str], AfterValidator(cap(20))] = Field(
        description="Ids of the duplicates to fold into the target"
    )
    reason: Annotated[str, AfterValidator(truncate(200))] = ""


class ConsolidationOut(BaseModel):
    merges: list[MergeGroupOut] = Field(default_factory=list)


def render(*, topics: Sequence[str]) -> str:
    return "Topics (id | name | analyzed items):\n" + "\n".join(topics)
