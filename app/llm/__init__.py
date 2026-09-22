"""Provider-agnostic LLM layer. Import from here, never from a provider module::

    from app.llm import LLMRequest, get_llm

    response = await get_llm().generate(LLMRequest(prompt="Summarize ..."))

Importing this package does not import any provider SDK.
"""

from app.llm.base import (
    Citation,
    Grounding,
    ImageRequest,
    ImageResponse,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ReasoningEffort,
    RetrievedURL,
    StructuredResponse,
    Tool,
)
from app.llm.errors import (
    LLMAuthenticationError,
    LLMBillingError,
    LLMBudgetExceededError,
    LLMConfigurationError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMRequestRejectedError,
    LLMResponseError,
    LLMUnavailableError,
)
from app.llm.factory import LazyLLM, create_llm_provider, get_llm
from app.llm.router import WritingRouter, unrecoverable

__all__ = [
    "Citation",
    "Grounding",
    "ImageRequest",
    "ImageResponse",
    "LLMAuthenticationError",
    "LLMBillingError",
    "LLMBudgetExceededError",
    "LLMConfigurationError",
    "LLMError",
    "LLMInvalidRequestError",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMRequestRejectedError",
    "LLMResponse",
    "LLMResponseError",
    "LLMUnavailableError",
    "LLMUsage",
    "LazyLLM",
    "ReasoningEffort",
    "RetrievedURL",
    "StructuredResponse",
    "Tool",
    "WritingRouter",
    "create_llm_provider",
    "get_llm",
    "unrecoverable",
]
