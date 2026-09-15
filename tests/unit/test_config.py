from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import load_competitors
from app.core.errors import ConfigurationError
from app.domain.content import ContentType

EXAMPLE = Path(__file__).parents[2] / "config" / "competitors.example.yaml"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "competitors.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_committed_example_file_is_valid() -> None:
    competitors = load_competitors(EXAMPLE)
    assert competitors
    assert all(c.slug for c in competitors)


def test_full_competitor_entry(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
competitors:
  - slug: acme
    name: Acme Inc.
    website: https://www.acme.test
    feeds: [https://www.acme.test/blog/feed.xml]
    tracked_pages: [https://www.acme.test/pricing]
    allowed_domains: [WWW.Acme-Blog.test]
    exclude_patterns: ["/tag/"]
    exclude_types: [listing, docs]
""",
    )
    (acme,) = load_competitors(path)
    assert acme.allowed_domains == ("acme-blog.test",)
    assert acme.exclude_types == frozenset({ContentType.LISTING, ContentType.DOCS})


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("competitors:\n  - {slug: 'Bad Slug', name: X, website: https://x.test}", "slug"),
        ("competitors:\n  - {slug: a, name: A, website: not-a-url}", "website"),
        (
            "competitors:\n  - {slug: a, name: A, website: https://a.test, exclude_patterns: ['(']}",
            "regex",
        ),
        (
            "competitors:\n  - {slug: a, name: A, website: https://a.test}\n  - {slug: a, name: B, website: https://b.test}",
            "duplicate",
        ),
        (
            "competitors:\n  - {slug: a, name: A, website: https://a.test, typo_field: 1}",
            "typo_field",
        ),
        ("competitors: [unclosed", "Invalid YAML"),
    ],
)
def test_invalid_files_fail_with_useful_messages(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_competitors(write(tmp_path, body))


def test_missing_file_explains_how_to_fix(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match=r"competitors\.example\.yaml"):
        load_competitors(tmp_path / "nope.yaml")


def test_database_url_is_secret_and_displayed_masked() -> None:
    from tests.fakesite import make_settings

    settings = make_settings(database_url="postgresql+psycopg://app:s3cret@db.internal:5432/intel")
    assert "s3cret" not in repr(settings)
    assert settings.database_url_display == "postgresql+psycopg://app:***@db.internal:5432/intel"


def test_analysis_settings_defaults_and_routes() -> None:
    from tests.fakesite import make_settings

    settings = make_settings()
    assert settings.analysis_model == settings.gemini_model == "gemini-3.8-flash"
    assert settings.synthesis_model == "gemini-3.8-flash"
    routed = make_settings(gemini_analysis_model="gemini-3.5-flash-lite")
    assert routed.analysis_model == "gemini-3.5-flash-lite"
    assert routed.synthesis_model == "gemini-3.8-flash"
    assert {t.value for t in settings.analysis_exclude_types} == {"careers", "legal", "listing"}


def test_an_item_must_fit_in_a_batch() -> None:
    from tests.fakesite import make_settings

    with pytest.raises(ValidationError, match="ANALYSIS_ITEM_MAX_CHARS"):
        make_settings(analysis_item_max_chars=50_000, analysis_batch_max_chars=10_000)


def test_article_settings_defaults_and_consistency() -> None:
    from app.config import Settings

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.writing_model == settings.gemini_model
    assert (settings.article_max_tokens, settings.article_target_words, settings.article_min_words) == (400_000, 1_500, 600)  # fmt: skip
    assert Settings(_env_file=None, gemini_writing_model="gemini-x").writing_model == "gemini-x"  # type: ignore[call-arg]
    for bad in ({"article_min_words": 2_000, "article_target_words": 1_000}, {"article_research_max_tokens": 500_000}, {"article_research_min_sources": 12}):  # fmt: skip
        with pytest.raises(ValueError, match="ARTICLE_"):
            Settings(_env_file=None, **bad)  # type: ignore[arg-type]
