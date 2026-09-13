"""ORM models, grouped by data layer (see MIGRATION_PLAN.md §7).

- config:      Competitor
- raw:         RawDocument (HTML exactly as fetched)
- normalized:  ContentItem, ContentVersion, ChangeEvent (facts extracted from sources)
- analysis:    Topic, TopicAlias, ContentAnalysis, ContentAnalysisTopic, ChangeSummary,
               CompetitorProfileSnapshot, LandscapeReport (interpretations; Phase 3)
- ops:         Run, RunEvent, LLMCall (what ran, when, what happened, what it cost)

Only the analysis layer holds LLM-produced data, and every such row records its run,
model and prompt version. Recommendations (Phase 4+) get their own tables later.
"""

from app.db.models.analysis import (
    ChangeSummary,
    CompetitorProfileSnapshot,
    ContentAnalysis,
    ContentAnalysisTopic,
    LandscapeReport,
    Topic,
    TopicAlias,
)
from app.db.models.config import Competitor
from app.db.models.content import ChangeEvent, ContentItem, ContentVersion
from app.db.models.ops import LLMCall, Run, RunEvent
from app.db.models.raw import RawDocument

__all__ = [
    "ChangeEvent",
    "ChangeSummary",
    "Competitor",
    "CompetitorProfileSnapshot",
    "ContentAnalysis",
    "ContentAnalysisTopic",
    "ContentItem",
    "ContentVersion",
    "LLMCall",
    "LandscapeReport",
    "RawDocument",
    "Run",
    "RunEvent",
    "Topic",
    "TopicAlias",
]
