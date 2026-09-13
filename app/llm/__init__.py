"""Provider-agnostic LLM layer. Import from here, never from a provider module::

    from app.llm import LLMRequest, get_llm

    response = await get_llm().generate(LLMRequest(prompt="Summarize ..."))

Importing this package does not import any provider SDK.
"""

from app.llm.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ReasoningEffort,
    StructuredResponse,
)
from app.llm.errors import (
    LLMAuthenticationError,
    LLMBudgetExceededError,
    LLMConfigurationError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMResponseError,
    LLMUnavailableError,
)
from app.llm.factory import LazyLLM, create_llm_provider, get_llm

__all__ = [
    "LLMAuthenticationError",
    "LLMBudgetExceededError",
    "LLMConfigurationError",
    "LLMError",
    "LLMInvalidRequestError",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMResponseError",
    "LLMUnavailableError",
    "LLMUsage",
    "LazyLLM",
    "ReasoningEffort",
    "StructuredResponse",
    "create_llm_provider",
    "get_llm",
]
