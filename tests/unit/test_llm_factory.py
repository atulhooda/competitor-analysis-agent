import pytest

from app.config import DEFAULT_GEMINI_MODEL, Settings
from app.llm import LLMConfigurationError, LLMProvider, create_llm_provider, get_llm
from tests.fakesite import make_settings


def env_settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_missing_key_raises_a_clear_error_mentioning_phase_1() -> None:
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY is not set") as caught:
        create_llm_provider(make_settings())
    assert "Phase 1" in str(caught.value)


def test_empty_key_in_environment_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "")
    settings = env_settings()
    assert settings.gemini_api_key is None
    assert not settings.llm_configured


def test_whitespace_key_counts_as_unset() -> None:
    assert make_settings(gemini_api_key="   ").gemini_api_key is None


def test_default_model_and_empty_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "")
    assert env_settings().gemini_model == DEFAULT_GEMINI_MODEL


def test_model_is_configurable_through_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    provider = create_llm_provider(env_settings())
    assert isinstance(provider, LLMProvider)
    assert provider.name == "gemini"
    assert provider.default_model == "gemini-3.5-flash-lite"


def test_get_llm_is_cached_and_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr("app.llm.factory.get_settings", env_settings)
    assert get_llm() is get_llm()


def test_the_key_is_never_exposed_by_settings_repr() -> None:
    settings = make_settings(gemini_api_key="super-secret-value")
    assert settings.llm_configured
    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in settings.model_dump_json()
