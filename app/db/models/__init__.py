"""ORM models, grouped by data layer (see MIGRATION_PLAN.md §7).

- config:      Competitor
- raw:         RawDocument (HTML exactly as fetched)
- normalized:  ContentItem, ContentVersion, ChangeEvent (facts extracted from sources)
- ops:         Run, RunEvent (what ran, when, and what happened)

AI-generated analysis (Phase 3) and recommendations (Phase 4+) get their own tables later;
nothing here is produced by an LLM.
"""

from app.db.models.config import Competitor
from app.db.models.content import ChangeEvent, ContentItem, ContentVersion
from app.db.models.ops import Run, RunEvent
from app.db.models.raw import RawDocument

__all__ = [
    "ChangeEvent",
    "Competitor",
    "ContentItem",
    "ContentVersion",
    "RawDocument",
    "Run",
    "RunEvent",
]
