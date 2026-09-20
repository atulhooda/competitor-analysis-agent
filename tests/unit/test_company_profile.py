from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import load_company_profile, load_scoring_config
from app.core.errors import ConfigurationError
from app.domain.company import CompanyProfile
from app.domain.opportunities import ScoringConfig

ROOT = Path(__file__).parents[2]


def test_the_committed_examples_are_valid() -> None:
    company = load_company_profile(ROOT / "config" / "company.example.yaml")
    assert company.name == "Your Startup"
    assert company.products[0].name == "Your Product"
    assert load_scoring_config(ROOT / "config" / "scoring.example.yaml") == ScoringConfig()


def test_scoring_defaults_apply_without_a_file(tmp_path: Path) -> None:
    assert load_scoring_config(tmp_path / "missing.yaml") == ScoringConfig()
    bad = tmp_path / "scoring.yaml"
    bad.write_text("scoring:\n  weights:\n    momentum: -5\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid scoring file"):
        load_scoring_config(bad)


def test_profile_normalization_and_validation() -> None:
    profile = CompanyProfile.model_validate(
        {
            "name": "Acme",
            "description": "We help.",
            "products": ["Widget", {"name": "Gadget", "description": "Does things"}],
            "core_topics": ["AI", "  ai ", "Automation"],
        }
    )
    assert [p.name for p in profile.products] == ["Widget", "Gadget"]
    assert profile.core_topics == ["AI", "Automation"]
    with pytest.raises(ValidationError):
        CompanyProfile.model_validate({"name": "Acme", "description": "x", "unknown": 1})
    with pytest.raises(ValidationError):
        CompanyProfile.model_validate({"name": "Acme", "description": "x", "preferred_formats": ["poem"]})  # fmt: skip


def test_scoring_fingerprint_ignores_fields_that_dont_affect_scores() -> None:
    base = CompanyProfile(name="Acme", description="We help.", core_topics=["AI"])
    tone = base.model_copy(update={"tone": "playful"})
    topics = base.model_copy(update={"core_topics": ["AI", "Automation"]})
    assert tone.fingerprint != base.fingerprint
    assert tone.scoring_fingerprint == base.scoring_fingerprint
    assert topics.scoring_fingerprint != base.scoring_fingerprint
