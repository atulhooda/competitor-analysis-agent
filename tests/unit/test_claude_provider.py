"""The Anthropic provider: structured output, the two built-in tools, grounding, usage and
the SDK error taxonomy. No network and no API key — the SDK client is a fake."""

from typing import Any

import pytest
from pydantic import BaseModel

from app.llm import (
    Grounding,
    LLMAuthenticationError,
    LLMBillingError,
    LLMConfigurationError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponseError,
    LLMUnavailableError,
)
from app.llm.claude import WEB_FETCH_TOOL, WEB_SEARCH_TOOL, ClaudeProvider
from tests.fakeanthropic import (
    FakeAnthropic,
    api_error,
    fetch_call,
    message,
    search_call,
)


class Answer(BaseModel):
    headline: str
    points: list[str]


def provider(client: FakeAnthropic, **kwargs: Any) -> ClaudeProvider:
    return ClaudeProvider(api_key="sk-test", model="claude-sonnet-5", client=client, **kwargs)


ANSWER_JSON = '{"headline": "Missed calls", "points": ["one", "two"]}'


# ── structured output ────────────────────────────────────────────────────────


async def test_a_structured_call_returns_the_validated_model() -> None:
    client = FakeAnthropic(reply=message(text=ANSWER_JSON))

    result = await provider(client).generate_structured(
        LLMRequest(prompt="Write about missed calls", system="Be useful", max_output_tokens=900),
        Answer,
    )

    assert result.data == Answer(headline="Missed calls", points=["one", "two"])
    assert (result.raw.provider, result.raw.model) == ("claude", "claude-sonnet-5")
    [body] = client.bodies
    assert body["max_tokens"] == 900
    assert body["system"] == "Be useful"
    assert body["messages"] == [{"role": "user", "content": "Write about missed calls"}]
    assert body["thinking"] == {"type": "adaptive"}
    assert "tools" not in body


async def test_the_text_is_validated_when_the_sdk_could_not_parse_it() -> None:
    # The SDK leaves parsed_output unset when the turn ended before the JSON did.
    client = FakeAnthropic(reply=message(text=ANSWER_JSON), parse_output=False)

    result = await provider(client).generate_structured(LLMRequest(prompt="go"), Answer)

    assert result.data.headline == "Missed calls"


async def test_output_that_does_not_match_the_schema_is_a_response_error_with_its_usage() -> None:
    client = FakeAnthropic(reply=message(text='{"headline": "only this"}'), parse_output=False)

    with pytest.raises(LLMResponseError, match="does not match Answer") as caught:
        await provider(client).generate_structured(LLMRequest(prompt="go"), Answer)

    assert caught.value.usage is not None
    assert caught.value.usage.total_tokens == 120  # the call was billed anyway


async def test_reasoning_effort_becomes_claudes_effort_setting() -> None:
    client = FakeAnthropic(reply=message(text=ANSWER_JSON))

    await provider(client).generate_structured(
        LLMRequest(prompt="go", reasoning_effort="minimal"), Answer
    )

    assert client.bodies[0]["output_config"] == {"effort": "low"}  # Claude has no "minimal"


async def test_plain_text_generation_returns_every_text_block() -> None:
    client = FakeAnthropic(reply=message(text="the answer"))

    response = await provider(client).generate(LLMRequest(prompt="go"))

    assert response.text == "the answer"
    assert response.finish_reason == "end_turn"
    assert response.grounding == Grounding()


async def test_a_refusal_is_a_response_error_naming_its_category() -> None:
    details = {"type": "refusal", "category": "cyber", "explanation": "no"}
    client = FakeAnthropic(reply=message(text="", stop_reason="refusal", stop_details=details))

    with pytest.raises(LLMResponseError, match=r"refusal.*cyber"):
        await provider(client).generate(LLMRequest(prompt="go"))


# ── the built-in tools and grounding ─────────────────────────────────────────


async def test_the_projects_tools_map_onto_claudes_web_tools() -> None:
    client = FakeAnthropic(reply=message(text=ANSWER_JSON))

    await provider(client).generate_structured(
        LLMRequest(prompt="go", tools=("google_search", "url_context")), Answer
    )

    assert client.bodies[0]["tools"] == [
        {"type": WEB_SEARCH_TOOL, "name": "web_search"},
        {"type": WEB_FETCH_TOOL, "name": "web_fetch"},
    ]


async def test_grounding_reports_the_searches_the_reads_and_the_citations() -> None:
    blocks = [
        *search_call("missed calls clinics", tool_use_id="s1"),
        *search_call("missed calls clinics", tool_use_id="s2"),  # the same query, once
        *fetch_call("https://example.org/a", tool_use_id="f1", final_url="https://example.org/a/"),
        *fetch_call("https://paywall.test/b", tool_use_id="f2", error_code="unavailable"),
    ]
    reply = message(text=ANSWER_JSON, blocks=blocks)
    reply["content"][-1]["citations"] = [
        {
            "type": "web_search_result_location",
            "url": "https://example.org/a/",
            "title": "A",
            "cited_text": "…",
            "encrypted_index": "z",
        }
    ]
    client = FakeAnthropic(reply=reply)

    result = await provider(client).generate_structured(
        LLMRequest(prompt="go", tools=("google_search", "url_context")), Answer
    )

    grounding = result.raw.grounding
    assert grounding.search_queries == ("missed calls clinics",)
    assert grounding.requested_urls == ("https://example.org/a", "https://paywall.test/b")
    assert [(r.url, r.status) for r in grounding.retrieved_urls] == [
        ("https://example.org/a/", "success"),  # the final URL, after a redirect
        ("https://paywall.test/b", "unavailable"),  # matched back through its tool_use id
    ]
    assert [(c.url, c.title) for c in grounding.citations] == [("https://example.org/a/", "A")]


async def test_grounding_stays_empty_when_the_request_asked_for_no_tool() -> None:
    client = FakeAnthropic(reply=message(text=ANSWER_JSON, blocks=search_call("q")))

    result = await provider(client).generate_structured(LLMRequest(prompt="go"), Answer)

    assert result.raw.grounding == Grounding()


# ── usage ────────────────────────────────────────────────────────────────────


async def test_usage_counts_everything_the_call_was_billed_for() -> None:
    usage = {
        "input_tokens": 100,
        "cache_read_input_tokens": 400,
        "cache_creation_input_tokens": 50,
        "output_tokens": 30,
        "output_tokens_details": {"thinking_tokens": 12},
    }
    client = FakeAnthropic(reply=message(text=ANSWER_JSON, usage=usage))

    result = await provider(client).generate_structured(LLMRequest(prompt="go"), Answer)

    billed = result.raw.usage
    assert billed.input_tokens == 550  # uncached + cache reads + cache writes
    assert (billed.cached_input_tokens, billed.output_tokens, billed.reasoning_tokens) == (400, 30, 12)  # fmt: skip
    assert billed.total_tokens == 580  # thinking is inside output, never counted twice


# ── errors ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "kind", "expected"),
    [
        (401, "authentication_error", LLMAuthenticationError),
        (403, "permission_error", LLMAuthenticationError),
        (402, "billing_error", LLMBillingError),
        (400, "invalid_request_error", LLMInvalidRequestError),
        (404, "not_found_error", LLMInvalidRequestError),
        (429, "rate_limit_error", LLMRateLimitError),
        (500, "api_error", LLMUnavailableError),
        (529, "overloaded_error", LLMUnavailableError),
    ],
)
async def test_sdk_errors_map_onto_the_projects_taxonomy(status: int, kind: str, expected: type[Exception]) -> None:  # fmt: skip
    client = FakeAnthropic(failures=[api_error(status, kind=kind)])

    with pytest.raises(expected):
        await provider(client).generate_structured(LLMRequest(prompt="go"), Answer)


async def test_a_billing_failure_is_not_a_malformed_request() -> None:
    client = FakeAnthropic(failures=[api_error(402, kind="billing_error", text="credits depleted")])  # fmt: skip

    with pytest.raises(LLMBillingError) as caught:
        await provider(client).generate_structured(LLMRequest(prompt="go"), Answer)

    assert not isinstance(caught.value, LLMInvalidRequestError)
    assert "credits depleted" in str(caught.value)


async def test_an_unreachable_api_is_unavailable_not_permanent() -> None:
    import anthropic
    import httpx2

    failure = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))  # fmt: skip
    client = FakeAnthropic(failures=[failure])

    with pytest.raises(LLMUnavailableError, match="Could not reach Anthropic"):
        await provider(client).generate_structured(LLMRequest(prompt="go"), Answer)


async def test_an_empty_key_or_model_is_a_configuration_error() -> None:
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        ClaudeProvider(api_key="  ", model="claude-sonnet-5", client=FakeAnthropic())
    with pytest.raises(LLMConfigurationError, match="CLAUDE_MODEL"):
        ClaudeProvider(api_key="sk-test", model="", client=FakeAnthropic())


async def test_claude_draws_no_cover_images() -> None:
    from app.llm import ImageRequest

    with pytest.raises(LLMConfigurationError, match="no images"):
        await provider(FakeAnthropic()).generate_image(ImageRequest(prompt="a clinic"))


async def test_closing_the_provider_closes_the_client() -> None:
    client = FakeAnthropic()

    await provider(client).aclose()

    assert client.closed
