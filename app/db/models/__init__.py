"""ORM models, grouped by data layer (see MIGRATION_PLAN.md §7).

- config:      Competitor
- raw:         RawDocument (HTML exactly as fetched)
- normalized:  ContentItem, ContentVersion, ChangeEvent (facts extracted from sources)
- config:      also CompanyProfileVersion (your own company, versioned; Phase 4)
- analysis:    Topic, TopicAlias, ContentAnalysis, ContentAnalysisTopic, ChangeSummary,
               CompetitorProfileSnapshot, LandscapeReport (interpretations; Phase 3)
- recommendation: Opportunity, OpportunityAssessment, OpportunityEvidence,
               OpportunityEvent (content opportunities; Phase 4)
- ops:         Run, RunEvent, LLMCall (what ran, when, what happened, what it cost)

LLM output lives only in the analysis layer and in assessments' ``interpretation``, and
every such row records its run, model and prompt version. Opportunity *scores* are
deterministic.
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
from app.db.models.config import CompanyProfileVersion, Competitor
from app.db.models.content import ChangeEvent, ContentItem, ContentVersion
from app.db.models.ops import LLMCall, Run, RunEvent
from app.db.models.raw import RawDocument
from app.db.models.recommendation import (
    Opportunity,
    OpportunityAssessment,
    OpportunityEvent,
    OpportunityEvidence,
)

__all__ = [
    "ChangeEvent",
    "ChangeSummary",
    "CompanyProfileVersion",
    "Competitor",
    "CompetitorProfileSnapshot",
    "ContentAnalysis",
    "ContentAnalysisTopic",
    "ContentItem",
    "ContentVersion",
    "LLMCall",
    "LandscapeReport",
    "Opportunity",
    "OpportunityAssessment",
    "OpportunityEvent",
    "OpportunityEvidence",
    "RawDocument",
    "Run",
    "RunEvent",
    "Topic",
    "TopicAlias",
]
