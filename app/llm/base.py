"""Provider-agnostic LLM interface.

Agents and services depend only on these types, never on a provider SDK. Get the
configured provider with ``app.llm.get_llm()``.
"""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

ReasoningEffort = Literal["minimal", "low", "medium", "high"]
# Built-in tools the provider runs itself: web search and reading the pages at given URLs.
Tool = Literal["google_search", "url_context"]


@dataclass(frozen=True, slots=True)
class LLMRequest:
    prompt: str
    system: str | None = None
    model: str | None = None  # None → the provider's default (GEMINI_MODEL)
    max_output_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None  # how hard the model should think
    tools: tuple[Tool, ...] = ()  # none by default: the model can only read the prompt

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("LLMRequest.prompt must not be empty")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("LLMRequest.max_output_tokens must be positive")


@dataclass(frozen=True, slots=True)
class Citation:
    """A source the provider attributed a span of its output to (search grounding)."""

    url: str
    title: str | None = None
    start_index: int | None = None
    end_index: int | None = None


@dataclass(frozen=True, slots=True)
class RetrievedURL:
    """A URL the provider tried to read with its URL tool, and how that went."""

    url: str
    status: str  # "success", "error", "paywall", "unsafe", ...


@dataclass(frozen=True, slots=True)
class Grounding:
    """What the provider's tools actually did during a call: the evidence that search and
    page reads happened, independent of what the model's text claims."""

    search_queries: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()
    requested_urls: tuple[str, ...] = ()
    retrieved_urls: tuple[RetrievedURL, ...] = ()


@dataclass(frozen=True, slots=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: LLMUsage
    finish_reason: str | None = None
    response_id: str | None = None
    grounding: Grounding = Grounding()  # empty unless the request enabled tools


@dataclass(frozen=True, slots=True)
class StructuredResponse[T: BaseModel]:
    data: T
    raw: LLMResponse


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def default_model(self) -> str: ...

    async def generate(self, request: LLMRequest) -> LLMResponse: ...

    async def generate_structured[T: BaseModel](
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        """Generate JSON matching ``schema`` and return it validated."""
        ...

    async def aclose(self) -> None: ...
