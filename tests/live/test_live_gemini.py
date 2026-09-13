"""Opt-in: one real Gemini call through the production prompt and schema (costs ~3k tokens).

    uv run pytest -m llm_live

Needs GEMINI_API_KEY (environment or .env). Skipped otherwise.
"""

import pytest

from app.config import Settings
from app.domain.content import ContentType
from app.llm import LLMRequest
from app.llm.gemini import GeminiProvider
from app.prompts import content_analysis
from app.services.digest import DigestSource, build_digest

pytestmark = pytest.mark.llm_live

# Read at import, before the test fixtures isolate the environment.
_SETTINGS = Settings()

PAGES = {
    "https://example.test/blog/ai-support-agents": (
        ContentType.BLOG_POST,
        "How AI agents are changing customer support",
        "Customer support teams are adopting AI agents to resolve routine tickets faster, while "
        "escalating complex conversations to people. The biggest gains come from automating "
        "password resets, order status questions and refund checks. Teams that succeed start "
        "with a narrow scope and measure resolution quality weekly. Handoffs matter: customers "
        "should never repeat themselves when an automated agent passes them to a human.",
    ),
    "https://example.test/pricing": (
        ContentType.PRICING,
        "Pricing",
        "Starter: $29 per agent per month, email support. Growth: $79 per agent per month with "
        "automation workflows, analytics and priority support. Enterprise: custom pricing, SSO, "
        "SOC 2 Type II reports and a dedicated success manager. Start a 14-day free trial.",
    ),
}


async def test_real_gemini_returns_valid_structured_analyses() -> None:
    if _SETTINGS.gemini_api_key is None:
        pytest.skip("GEMINI_API_KEY is not set")
    provider = GeminiProvider(
        api_key=_SETTINGS.gemini_api_key.get_secret_value(),
        model=_SETTINGS.analysis_model,
        timeout_seconds=_SETTINGS.llm_timeout_seconds,
        max_retries=_SETTINGS.llm_max_retries,
    )
    digests = [
        build_digest(
            DigestSource(
                content_item_id=i, content_version_id=i, url=url, content_type=kind, title=title,
                description=None, author=None, published_at=None, published_at_source=None,
                categories=[], tags=[], headings=[], text=text, word_count=len(text.split()),
            ),
            max_chars=4_000,
        )
        for i, (url, (kind, title, text)) in enumerate(PAGES.items(), start=1)
    ]  # fmt: skip
    prompt = content_analysis.render(
        competitor="Example",
        website="https://example.test/",
        taxonomy=[],
        documents=[d.render(f"D{i}") for i, d in enumerate(digests, start=1)],
    )
    try:
        result = await provider.generate_structured(
            LLMRequest(
                prompt=prompt,
                system=content_analysis.SYSTEM,
                max_output_tokens=content_analysis.max_output_tokens(len(digests)),
                reasoning_effort="low",
            ),
            content_analysis.ContentAnalysisResponse,
        )
    finally:
        await provider.aclose()

    analyses = {a.document_id: a for a in result.data.analyses}
    assert set(analyses) == {"D1", "D2"}
    assert analyses["D1"].topics, "the article should get at least one topic"
    assert analyses["D2"].content_format == "pricing_page"
    assert result.raw.usage.total_tokens > 0
    print(
        "\n".join(
            f"{ref}: {a.content_format} {[t.name for t in a.topics]} {a.intent}"
            for ref, a in analyses.items()
        ),
        f"\ntokens: {result.raw.usage.total_tokens}",
    )
