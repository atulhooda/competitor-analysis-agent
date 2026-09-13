"""GeminiProvider tests. The Gemini API is mocked at the HTTP level with respx, so the real
SDK builds and parses every request, but no network call is made and no real key is used."""

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx
from pydantic import BaseModel

from app.llm import (
    LLMAuthenticationError,
    LLMInvalidRequestError,
    LLMProvider,
    LLMRateLimitError,
    LLMRequest,
    LLMResponseError,
    LLMUnavailableError,
)
from app.llm.gemini import GeminiProvider

FAKE_KEY = "test-key-not-real"
INTERACTIONS = r"https://generativelanguage\.googleapis\.com/.+/interactions"
NO_WAIT = {"retry-after-ms": "0"}  # the SDK honors this, so retries don't sleep in tests


def interaction(
    text: str = "Hello", status: str = "completed", **extra: object
) -> dict[str, object]:
    body: dict[str, object] = {
        "id": "int_test_1",
        "status": status,
        "model": "gemini-test-model",
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
        "usage": {
            "total_input_tokens": 12,
            "total_output_tokens": 5,
            "total_thought_tokens": 3,
            "total_cached_tokens": 2,
            "total_tokens": 20,
        },
    }
    body.update(extra)
    return body


def api_error(status: int) -> httpx.Response:
    return httpx.Response(
        status, headers=NO_WAIT, json={"error": {"code": status, "message": f"test error {status}"}}
    )


@pytest.fixture
async def provider() -> AsyncIterator[GeminiProvider]:
    p = GeminiProvider(
        api_key=FAKE_KEY, model="gemini-test-model", timeout_seconds=5, max_retries=1
    )
    yield p
    await p.aclose()


class Topic(BaseModel):
    name: str
    score: int


def test_implements_the_provider_interface(provider: GeminiProvider) -> None:
    assert isinstance(provider, LLMProvider)
    assert provider.name == "gemini"
    assert provider.default_model == "gemini-test-model"


async def test_generate_builds_a_stateless_interactions_request(provider: GeminiProvider) -> None:
    with respx.mock() as router:
        route = router.post(url__regex=INTERACTIONS).respond(200, json=interaction("Hi there"))
        response = await provider.generate(
            LLMRequest(
                prompt="Say hi", system="Be brief", max_output_tokens=64, reasoning_effort="low"
            )
        )

    sent = route.calls.last.request
    body = json.loads(sent.content)
    assert body["model"] == "gemini-test-model"
    assert body["input"] == "Say hi"
    assert body["system_instruction"] == "Be brief"
    assert body["generation_config"] == {"max_output_tokens": 64, "thinking_level": "low"}
    assert body["store"] is False
    assert sent.headers["x-goog-api-key"] == FAKE_KEY
    assert response.text == "Hi there"
    assert response.provider == "gemini"
    assert response.finish_reason == "completed"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 5
    assert response.usage.reasoning_tokens == 3
    assert response.usage.cached_input_tokens == 2


async def test_per_request_model_override(provider: GeminiProvider) -> None:
    with respx.mock() as router:
        route = router.post(url__regex=INTERACTIONS).respond(200, json=interaction())
        await provider.generate(LLMRequest(prompt="x", model="gemini-other-model"))
    assert json.loads(route.calls.last.request.content)["model"] == "gemini-other-model"


async def test_generate_structured_sends_schema_and_validates(provider: GeminiProvider) -> None:
    with respx.mock() as router:
        route = router.post(url__regex=INTERACTIONS).respond(
            200, json=interaction('{"name": "AI support agents", "score": 87}')
        )
        result = await provider.generate_structured(LLMRequest(prompt="Top topic?"), Topic)

    response_format = json.loads(route.calls.last.request.content)["response_format"]
    assert response_format["mime_type"] == "application/json"
    assert response_format["schema"]["properties"].keys() == {"name", "score"}
    assert result.data == Topic(name="AI support agents", score=87)
    assert result.raw.usage.total_tokens == 20


async def test_structured_output_that_does_not_match_the_schema(provider: GeminiProvider) -> None:
    with respx.mock() as router:
        router.post(url__regex=INTERACTIONS).respond(200, json=interaction('{"name": "x"}'))
        with pytest.raises(LLMResponseError, match="does not match Topic"):
            await provider.generate_structured(LLMRequest(prompt="x"), Topic)


@pytest.mark.parametrize(
    "body",
    [interaction(status="failed", errors=[{"message": "blocked"}]), interaction(text="   ")],
    ids=["failed-status", "empty-text"],
)
async def test_unusable_responses_raise(provider: GeminiProvider, body: dict[str, object]) -> None:
    with respx.mock() as router:
        router.post(url__regex=INTERACTIONS).respond(200, json=body)
        with pytest.raises(LLMResponseError):
            await provider.generate(LLMRequest(prompt="x"))


async def test_rate_limits_are_retried_by_the_sdk(provider: GeminiProvider) -> None:
    with respx.mock() as router:
        route = router.post(url__regex=INTERACTIONS).mock(
            side_effect=[api_error(429), httpx.Response(200, json=interaction("ok"))]
        )
        response = await provider.generate(LLMRequest(prompt="x"))
    assert response.text == "ok"
    assert route.call_count == 2


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (429, LLMRateLimitError),
        (500, LLMUnavailableError),
        (503, LLMUnavailableError),
        (401, LLMAuthenticationError),
        (403, LLMAuthenticationError),
        (400, LLMInvalidRequestError),
        (404, LLMInvalidRequestError),
    ],
)
async def test_http_errors_map_to_provider_neutral_errors(
    provider: GeminiProvider, status: int, error: type[Exception]
) -> None:
    with respx.mock() as router:
        route = router.post(url__regex=INTERACTIONS).mock(return_value=api_error(status))
        with pytest.raises(error):
            await provider.generate(LLMRequest(prompt="x"))
    if status < 500 and status != 429:
        assert route.call_count == 1  # client errors are never retried


async def test_auth_errors_never_echo_the_api_key(provider: GeminiProvider) -> None:
    with respx.mock() as router:
        router.post(url__regex=INTERACTIONS).mock(return_value=api_error(401))
        with pytest.raises(LLMAuthenticationError) as caught:
            await provider.generate(LLMRequest(prompt="x"))
    assert FAKE_KEY not in str(caught.value)


async def test_timeouts_map_to_unavailable() -> None:
    provider = GeminiProvider(api_key=FAKE_KEY, model="m", timeout_seconds=1, max_retries=0)
    with respx.mock() as router:
        router.post(url__regex=INTERACTIONS).mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(LLMUnavailableError):
            await provider.generate(LLMRequest(prompt="x"))
    await provider.aclose()


def test_request_validation() -> None:
    with pytest.raises(ValueError, match="prompt"):
        LLMRequest(prompt="  ")
    with pytest.raises(ValueError, match="max_output_tokens"):
        LLMRequest(prompt="x", max_output_tokens=0)


class Subtopic(BaseModel):
    name: str
    weight: float = 0.5


class TopicTree(BaseModel):
    topics: list[Subtopic]
    primary: Subtopic | None = None


async def test_nested_schemas_are_sent_self_contained(provider: GeminiProvider) -> None:
    with respx.mock() as router:
        route = router.post(url__regex=INTERACTIONS).respond(
            200, json=interaction('{"topics": [{"name": "AI agents"}], "primary": null}')
        )
        result = await provider.generate_structured(LLMRequest(prompt="Topics?"), TopicTree)
    schema = json.loads(route.calls.last.request.content)["response_format"]["schema"]
    assert "$defs" not in schema
    assert "$ref" not in json.dumps(schema)
    assert schema["properties"]["topics"]["items"]["properties"]["name"] == {"title": "Name", "type": "string"}  # fmt: skip
    assert result.data.topics[0].weight == 0.5  # defaults are applied on validation


async def test_invalid_structured_output_still_reports_billed_usage(provider: GeminiProvider) -> None:  # fmt: skip
    with respx.mock() as router:
        router.post(url__regex=INTERACTIONS).respond(200, json=interaction('{"name": 1'))
        with pytest.raises(LLMResponseError) as caught:
            await provider.generate_structured(LLMRequest(prompt="x"), Topic)
    assert caught.value.usage is not None
    assert caught.value.usage.total_tokens == 20
