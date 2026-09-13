"""Guards for the committed .env.example: it must parse and contain placeholders only."""

import re
from pathlib import Path

from app.config import DEFAULT_GEMINI_MODEL, DEFAULT_USER_AGENT, Settings

ENV_EXAMPLE = Path(__file__).parents[2] / ".env.example"


def test_env_example_parses_and_leaves_secrets_unset() -> None:
    settings = Settings(_env_file=ENV_EXAMPLE)  # type: ignore[call-arg]
    assert settings.gemini_api_key is None
    assert settings.api_key is None
    assert not settings.llm_configured
    assert settings.gemini_model == DEFAULT_GEMINI_MODEL
    assert settings.crawler_user_agent == DEFAULT_USER_AGENT


def test_env_example_declares_the_gemini_settings() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert re.search(r"^GEMINI_API_KEY=$", text, re.MULTILINE)
    assert re.search(r"^GEMINI_MODEL=$", text, re.MULTILINE)


def test_env_example_contains_no_secret_values() -> None:
    secret_names = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if any(part in name for part in secret_names):
            assert value == "", f"{name} must be an empty placeholder in .env.example"
