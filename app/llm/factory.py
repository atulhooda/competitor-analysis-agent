"""Builds the configured LLM providers.

The ``get_llm()`` factory shape follows Str1nX03/Competitor-Research
(MIT, © 2026 Dravin Kumar Sharma). See THIRD_PARTY_NOTICES.md.
"""

from functools import lru_cache

from app.config import CLAUDE, GEMINI, Settings, get_settings
from app.llm.base import LLMProvider
from app.llm.errors import LLMConfigurationError


def create_llm_provider(settings: Settings, provider: str = GEMINI) -> LLMProvider:
    """Return ``provider`` (gemini or claude) built from ``settings``, or raise if its API
    key is not set. The provider SDKs are imported lazily, so a process that never calls
    one never loads it."""
    if provider == CLAUDE:
        if settings.anthropic_api_key is None:
            raise LLMConfigurationError(
                "ANTHROPIC_API_KEY is not set: Claude cannot write "
                "(WRITING_PROVIDER=claude or split needs it)"
            )
        from app.llm.claude import ClaudeProvider

        return ClaudeProvider(
            api_key=settings.anthropic_api_key.get_secret_value(),
            model=settings.claude_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    if provider != GEMINI:
        raise LLMConfigurationError(f"Unknown LLM provider {provider!r}")
    if settings.gemini_api_key is None:
        raise LLMConfigurationError(
            "GEMINI_API_KEY is not set. LLM features (Phase 3 onward) require it; "
            "Phase 1 website scanning does not."
        )
    from app.llm.gemini import GeminiProvider

    return GeminiProvider(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_model,
        image_model=settings.gemini_image_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


@lru_cache(maxsize=1)
def get_llm() -> LLMProvider:
    """Process-wide provider built from environment settings. Clear with ``get_llm.cache_clear()``."""
    return create_llm_provider(get_settings())


class LazyLLM:
    """Builds one provider on first use.

    The API server and CLI start (and scan) without ``GEMINI_API_KEY``; only features
    that call the LLM fail, with a clear ``LLMConfigurationError``.

    ``name`` says which provider it builds (gemini by default, so every existing call site
    keeps its meaning). An injected ``provider`` is used as-is and is never rebuilt.
    """

    def __init__(
        self, settings: Settings, provider: LLMProvider | None = None, name: str = GEMINI
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._owned = provider is None
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def configured(self) -> bool:
        return self._provider is not None or self._settings.provider_configured(self._name)

    def get(self) -> LLMProvider:
        if self._provider is None:
            self._provider = create_llm_provider(self._settings, self._name)
        return self._provider

    async def aclose(self) -> None:
        if self._provider is not None and self._owned:
            await self._provider.aclose()
            self._provider = None
