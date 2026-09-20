"""Seed taxonomy file format (``config/topics.yaml``). Optional: topics are also created
as the analyzer meets new subjects, then kept tidy by aliases and merges."""

from pydantic import BaseModel, ConfigDict, Field


class TopicSeed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=500)
    aliases: list[str] = Field(default_factory=list, description="Other names for this topic")
    subtopics: list[str] = Field(default_factory=list)


class TopicsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topics: list[TopicSeed] = Field(default_factory=list)
