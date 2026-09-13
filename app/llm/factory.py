"""Builds the configured LLM provider.

The ``get_llm()`` factory shape follows Str1nX03/Competitor-Research
(MIT, © 2026 Dravin Kumar Sharma). See THIRD_PARTY_NOTICES.md.
"""

from functools import lru_cache

from app.config import Settings, get_settings
from app.llm.base import LLMProvider
from app.llm.errors import LLMConfigurationError


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Return the Gemini provider for ``settings``, or raise if no API key is set."""
    if settings.gemini_api_key is None:
        raise LLMConfigurationError(
            "GEMINI_API_KEY is not set. LLM features (Phase 3 onward) require it; "
            "Phase 1 website scanning does not."
        )
    # Lazy import: the Gemini SDK is loaded only when an LLM is actually used.
    from app.llm.gemini import GeminiProvider

    return GeminiProvider(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


@lru_cache(maxsize=1)
def get_llm() -> LLMProvider:
    """Process-wide provider built from environment settings. Clear with ``get_llm.cache_clear()``."""
    return create_llm_provider(get_settings())
