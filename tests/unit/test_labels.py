import pytest

from app.services.labels import clean_label, label_key, slugify


@pytest.mark.parametrize(
    ("label", "key"),
    [
        ("AI agents", "ai agent"),
        ("AI Agents", "ai agent"),
        ("AI-agents", "ai agent"),
        ("ai‐agents", "ai agent"),  # unicode hyphen  # noqa: RUF001
        ("AI — Agents", "ai agent"),
        ("Artificial Intelligence agents", "ai agent"),
        ("The AI agents", "ai agent"),
        ("Customer support", "customer support"),
        ("customer_support", "customer support"),
        ("E-commerce", "ecommerce"),
        ("Search Engine Optimization", "seo"),
        ("Large language models", "llm"),
        ("LLMs", "llm"),
        ("APIs", "api"),
        ("Businesses", "business"),
        ("Strategies", "strategy"),
        ("Personas", "persona"),
        ("Analytics", "analytics"),
        ("SaaS", "saas"),
        ("DevOps", "devops"),
        ("Data analysis", "data analysis"),
        ("Customer success", "customer success"),
        ("Node.js", "node.js"),
        ("C++", "c++"),
        ("Sales & Marketing", "sales and marketing"),
        ("🚀", ""),
        ("  ", ""),
    ],
)
def test_label_key_unifies_spellings_but_not_meanings(label: str, key: str) -> None:
    assert label_key(label) == key


def test_different_subjects_keep_different_keys() -> None:
    assert label_key("AI agents") != label_key("Agentic AI")  # synonyms are merged, not guessed
    assert label_key("AI") != label_key("AI agents")
    assert label_key("Pricing") != label_key("Billing")


def test_clean_label_keeps_casing_and_trims_noise() -> None:
    assert clean_label('  "AI   agents". ') == "AI agents"
    assert clean_label("iOS apps") == "iOS apps"
    assert clean_label("x" * 100, max_length=10) == "x" * 10


def test_slugify() -> None:
    assert slugify("AI agents & automation") == "ai-agents-automation"
    assert slugify("Café Déjà vu") == "cafe-deja-vu"
    assert slugify("🚀") == "topic"
