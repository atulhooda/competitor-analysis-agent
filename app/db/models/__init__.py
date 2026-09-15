"""ORM models, grouped by data layer (see MIGRATION_PLAN.md §7).

- config:      Competitor
- raw:         RawDocument (HTML exactly as fetched)
- normalized:  ContentItem, ContentVersion, ChangeEvent (facts extracted from sources)
- config:      also CompanyProfileVersion (your own company, versioned; Phase 4)
- analysis:    Topic, TopicAlias, ContentAnalysis, ContentAnalysisTopic, ChangeSummary,
               CompetitorProfileSnapshot, LandscapeReport (interpretations; Phase 3)
- recommendation: Opportunity, OpportunityAssessment, OpportunityEvidence,
               OpportunityEvent (content opportunities; Phase 4)
- generation:  Article, ArticleStepRun, ArticleVersion, ArticleSource, ArticleCitation
               (article drafts; Phase 5), ArticleClaimCheck, ArticleOriginalityFlag,
               ArticleQualityReport (validation; Phase 6)
- publishing:  ArticleApproval, Publication, PublicationAttempt (Phase 7: approved, ready
               versions only; drafts by default)
- scheduling:  Job, SchedulerState (Phase 8: jobs, checkpoints, the pause switch)
- ops:         Run, RunEvent, LLMCall (what ran, when, what happened, what it cost)

LLM output lives in the analysis layer, in assessments' ``interpretation`` and in the
generation layer, and every such row records its run, model and prompt version.
Opportunity *scores* and article briefs are deterministic.
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
from app.db.models.article import (
    Article,
    ArticleCitation,
    ArticleClaimCheck,
    ArticleOriginalityFlag,
    ArticleQualityReport,
    ArticleSource,
    ArticleStepRun,
    ArticleVersion,
)
from app.db.models.config import CompanyProfileVersion, Competitor
from app.db.models.content import ChangeEvent, ContentItem, ContentVersion
from app.db.models.jobs import Job, SchedulerState
from app.db.models.ops import LLMCall, Run, RunEvent
from app.db.models.publishing import ArticleApproval, Publication, PublicationAttempt
from app.db.models.raw import RawDocument
from app.db.models.recommendation import (
    Opportunity,
    OpportunityAssessment,
    OpportunityEvent,
    OpportunityEvidence,
)

__all__ = [
    "Article",
    "ArticleApproval",
    "ArticleCitation",
    "ArticleClaimCheck",
    "ArticleOriginalityFlag",
    "ArticleQualityReport",
    "ArticleSource",
    "ArticleStepRun",
    "ArticleVersion",
    "ChangeEvent",
    "ChangeSummary",
    "CompanyProfileVersion",
    "Competitor",
    "CompetitorProfileSnapshot",
    "ContentAnalysis",
    "ContentAnalysisTopic",
    "ContentItem",
    "ContentVersion",
    "Job",
    "LLMCall",
    "LandscapeReport",
    "Opportunity",
    "OpportunityAssessment",
    "OpportunityEvent",
    "OpportunityEvidence",
    "Publication",
    "PublicationAttempt",
    "RawDocument",
    "Run",
    "RunEvent",
    "SchedulerState",
    "Topic",
    "TopicAlias",
]
