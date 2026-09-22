"""Which provider writes an article, and which one it falls back to.

Routing is a property of the *article*, not of a call: one article is researched, outlined,
drafted and edited by one provider, so its steps stay consistent with each other and its
step fingerprints stay stable across resumes.

``WRITING_PROVIDER`` decides:

- ``gemini`` (the default): Gemini writes everything — today's behaviour, unchanged.
- ``claude``: Claude writes everything.
- ``split``: articles from *editorial* opportunities are written by Claude, articles from
  competitor opportunities by Gemini. That is the owner's "10 from Claude, 4 from Gemini"
  in the system's own terms, because the two opportunity origins have their own daily
  allowances (MAX_EDITORIAL_ARTICLES_PER_DAY and MAX_ARTICLES_GENERATED_PER_DAY).
  ``CLAUDE_ARTICLE_SHARE`` caps how many articles Claude writes in a day; past the cap the
  editorial articles go to Gemini too.

A provider without an API key is never chosen and never fallen back to, so a deployment
that hasn't set ``ANTHROPIC_API_KEY`` behaves exactly as it did before.
"""

from app.config import CLAUDE, GEMINI, Settings
from app.llm.base import LLMProvider
from app.llm.errors import LLMConfigurationError
from app.llm.factory import LazyLLM

# Errors that mean "this provider will answer every later call the same way": trying the
# same provider again is pointless, so the article falls back to the other one.
UNRECOVERABLE = (
    "LLMAuthenticationError",
    "LLMBillingError",
    "LLMConfigurationError",
    "LLMInvalidRequestError",
)


def unrecoverable(error: BaseException | None) -> bool:
    """True for a provider failure that no retry on that provider fixes."""
    if error is None:
        return False
    return bool({cls.__name__ for cls in type(error).__mro__} & set(UNRECOVERABLE))


class WritingRouter:
    """Picks the provider for an article and hands out the built providers.

    The Gemini side is the ``LazyLLM`` the service was given, so an injected provider (the
    API server's shared client, a test's fake) keeps being used.
    """

    def __init__(
        self,
        settings: Settings,
        gemini: LazyLLM,
        claude: LazyLLM | None = None,
    ) -> None:
        self._settings = settings
        self._lazy = {GEMINI: gemini, CLAUDE: claude or LazyLLM(settings, name=CLAUDE)}

    def configured(self, provider: str) -> bool:
        lazy = self._lazy.get(provider)
        return lazy is not None and lazy.configured

    @property
    def providers(self) -> tuple[str, ...]:
        """The providers that have a key, in the order they are preferred."""
        return tuple(name for name in (GEMINI, CLAUDE) if self.configured(name))

    def get(self, provider: str) -> LLMProvider:
        lazy = self._lazy.get(provider)
        if lazy is None:
            raise LLMConfigurationError(f"Unknown writing provider {provider!r}")
        return lazy.get()

    def fallback(self, provider: str) -> str | None:
        """The other configured provider, or None when there isn't one."""
        other = CLAUDE if provider == GEMINI else GEMINI
        return other if self.configured(other) else None

    def choose(self, *, editorial: bool, claude_written_today: int) -> str:
        """The provider that should write an article from this kind of opportunity.

        Falls straight through to whichever provider is configured when the preferred one
        isn't, so a missing key degrades instead of failing.
        """
        preferred = self._preferred(editorial=editorial, claude_written_today=claude_written_today)
        if self.configured(preferred):
            return preferred
        return self.fallback(preferred) or preferred

    def _preferred(self, *, editorial: bool, claude_written_today: int) -> str:
        mode = self._settings.writing_provider
        if mode == CLAUDE:
            return CLAUDE
        if mode != "split":
            return GEMINI
        if not editorial:
            return GEMINI
        # The daily cap: past it, the rest of the day's editorial articles go to Gemini.
        if claude_written_today >= self._settings.claude_article_share:
            return GEMINI
        return CLAUDE

    def caps_claude(self) -> bool:
        """True when the daily Claude cap is worth counting (only ``split`` uses it)."""
        return self._settings.writing_provider == "split"

    async def aclose(self) -> None:
        for lazy in self._lazy.values():
            await lazy.aclose()
