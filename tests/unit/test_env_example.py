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
    secret_words = {"KEY", "TOKEN", "SECRET", "PASSWORD", "URL"}  # URLs may embed credentials
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if secret_words & set(name.split("_")):
            assert value == "", f"{name} must be an empty placeholder in .env.example"


def test_env_example_documents_every_setting() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    internal = {"database_echo", "log_json"}
    documented = {
        name
        for name in Settings.model_fields
        if name not in internal and not name.startswith("crawler_")
    }
    missing = [n for n in sorted(documented) if not re.search(rf"^#? ?{n.upper()}=", text, re.MULTILINE)]  # fmt: skip
    assert not missing, f"undocumented in .env.example: {missing}"
