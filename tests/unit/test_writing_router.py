"""Which provider writes an article (WRITING_PROVIDER), which one it falls back to, and
which failures are worth falling back over."""

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.llm import (
    LLMAuthenticationError,
    LLMBillingError,
    LLMBudgetExceededError,
    LLMConfigurationError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMResponseError,
    LLMUnavailableError,
    WritingRouter,
    unrecoverable,
)
from app.llm.factory import LazyLLM
from tests.fakellm import FakeLLM
from tests.fakesite import make_settings


def router(*, gemini: bool = True, claude: bool = True, **overrides: object) -> WritingRouter:
    settings: Settings = make_settings(
        gemini_api_key=SecretStr("g") if gemini else None,
        anthropic_api_key=SecretStr("a") if claude else None,
        **overrides,
    )
    return WritingRouter(settings, LazyLLM(settings))


# ── routing ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("editorial", [True, False])
def test_the_default_sends_everything_to_gemini(editorial: bool) -> None:
    assert router().choose(editorial=editorial, claude_written_today=0) == "gemini"


@pytest.mark.parametrize("editorial", [True, False])
def test_writing_provider_claude_sends_everything_to_claude(editorial: bool) -> None:
    chosen = router(writing_provider="claude").choose(editorial=editorial, claude_written_today=99)

    assert chosen == "claude"  # the daily share caps split only


def test_split_sends_editorial_articles_to_claude_and_competitor_ones_to_gemini() -> None:
    split = router(writing_provider="split")

    assert split.choose(editorial=True, claude_written_today=0) == "claude"
    assert split.choose(editorial=False, claude_written_today=0) == "gemini"


def test_the_daily_share_caps_how_many_articles_claude_writes() -> None:
    split = router(writing_provider="split", claude_article_share=10)

    assert split.choose(editorial=True, claude_written_today=9) == "claude"
    assert split.choose(editorial=True, claude_written_today=10) == "gemini"
    assert split.choose(editorial=True, claude_written_today=11) == "gemini"


def test_a_share_of_zero_turns_claude_off_without_unsetting_its_key() -> None:
    split = router(writing_provider="split", claude_article_share=0)

    assert split.choose(editorial=True, claude_written_today=0) == "gemini"
    assert split.configured("claude")


def test_only_split_counts_the_daily_share() -> None:
    assert router(writing_provider="split").caps_claude()
    assert not router(writing_provider="claude").caps_claude()
    assert not router().caps_claude()


# ── missing keys ─────────────────────────────────────────────────────────────


def test_without_an_anthropic_key_nothing_is_routed_to_claude_or_falls_back_to_it() -> None:
    only_gemini = router(claude=False, writing_provider="split")

    assert only_gemini.providers == ("gemini",)
    assert only_gemini.choose(editorial=True, claude_written_today=0) == "gemini"
    assert only_gemini.fallback("gemini") is None


def test_without_a_gemini_key_claude_writes_and_gemini_is_no_fallback() -> None:
    only_claude = router(gemini=False, writing_provider="split")

    assert only_claude.providers == ("claude",)
    assert only_claude.choose(editorial=False, claude_written_today=0) == "claude"
    assert only_claude.fallback("claude") is None


def test_each_provider_falls_back_to_the_other_when_both_have_keys() -> None:
    both = router()

    assert both.providers == ("gemini", "claude")
    assert both.fallback("gemini") == "claude"
    assert both.fallback("claude") == "gemini"


def test_an_injected_provider_counts_as_configured_without_any_key() -> None:
    settings = make_settings(gemini_api_key=None, anthropic_api_key=None)
    fake = FakeLLM()
    both = WritingRouter(settings, LazyLLM(settings, provider=fake), LazyLLM(settings, provider=FakeLLM(provider="claude"), name="claude"))  # type: ignore[arg-type]  # fmt: skip

    assert both.providers == ("gemini", "claude")
    assert both.get("gemini") is fake
    assert both.get("claude").name == "claude"


def test_an_unknown_provider_is_a_configuration_error() -> None:
    with pytest.raises(LLMConfigurationError, match="Unknown writing provider"):
        router().get("llama")


# ── which failures are worth another provider ────────────────────────────────


@pytest.mark.parametrize(
    "error",
    [
        LLMAuthenticationError("bad key"),
        LLMBillingError("prepayment credits are depleted"),
        LLMConfigurationError("no key"),
        LLMInvalidRequestError("unknown model"),
    ],
)
def test_failures_no_retry_fixes_are_worth_the_other_provider(error: Exception) -> None:
    assert unrecoverable(error)


@pytest.mark.parametrize(
    "error",
    [
        LLMRateLimitError("429"),
        LLMUnavailableError("503"),
        LLMResponseError("bad json"),
        LLMBudgetExceededError("budget"),
        RuntimeError("something else"),
        None,
    ],
)
def test_everything_else_keeps_the_provider_it_started_with(error: Exception | None) -> None:
    assert not unrecoverable(error)
