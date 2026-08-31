from __future__ import annotations

import pytest

from character.semantic_review_adapter import (
    DEFAULT_SEMANTIC_REVIEW_TIMEOUT_SECONDS,
    MAX_SEMANTIC_REVIEW_TIMEOUT_SECONDS,
    MIN_SEMANTIC_REVIEW_TIMEOUT_SECONDS,
    SEMANTIC_REVIEW_MAX_TOKENS,
    SemanticReviewSettings,
    VLLMSemanticReviewer,
    create_default_semantic_review_runtime,
    create_default_semantic_reviewer,
)


def test_settings_are_opt_in_and_use_a_short_default_timeout():
    settings = SemanticReviewSettings.from_env({})

    assert settings.enabled is False
    assert settings.timeout_seconds == DEFAULT_SEMANTIC_REVIEW_TIMEOUT_SECONDS


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_settings_accept_explicit_truthy_values(value):
    settings = SemanticReviewSettings.from_env(
        {
            "DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED": value,
            "DYNAMIC_CONTEXT_SEMANTIC_REVIEW_TIMEOUT_SECONDS": "1.75",
        }
    )

    assert settings.enabled is True
    assert settings.timeout_seconds == 1.75


@pytest.mark.parametrize("value", ["", "invalid", "0", "-1", "nan", "inf"])
def test_invalid_timeouts_fall_back_to_the_safe_default(value):
    settings = SemanticReviewSettings.from_env({"DYNAMIC_CONTEXT_SEMANTIC_REVIEW_TIMEOUT_SECONDS": value})

    assert settings.timeout_seconds == DEFAULT_SEMANTIC_REVIEW_TIMEOUT_SECONDS


def test_timeout_is_bounded_to_prevent_pathological_configuration():
    too_short = SemanticReviewSettings.from_env({"DYNAMIC_CONTEXT_SEMANTIC_REVIEW_TIMEOUT_SECONDS": "0.001"})
    too_long = SemanticReviewSettings.from_env({"DYNAMIC_CONTEXT_SEMANTIC_REVIEW_TIMEOUT_SECONDS": "999"})

    assert too_short.timeout_seconds == MIN_SEMANTIC_REVIEW_TIMEOUT_SECONDS
    assert too_long.timeout_seconds == MAX_SEMANTIC_REVIEW_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_disabled_factory_does_not_resolve_a_client():
    called = False

    async def client_factory():
        nonlocal called
        called = True
        raise AssertionError("disabled reviewer must not resolve vLLM")

    reviewer = create_default_semantic_reviewer(env={}, client_factory=client_factory)

    assert reviewer is None
    assert called is False


@pytest.mark.asyncio
async def test_reviewer_calls_low_level_client_with_deterministic_base_model_request():
    captured = {}

    class Client:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            return '{"state": {}}'

    async def client_factory():
        return Client()

    messages = (
        {"role": "system", "content": "return JSON"},
        {"role": "user", "content": "ambiguous turn"},
    )
    reviewer = VLLMSemanticReviewer(client_factory)

    result = await reviewer(messages)

    assert result == '{"state": {}}'
    assert captured == {
        "messages": [dict(message) for message in messages],
        "lora_name": None,
        "temperature": 0.0,
        "max_tokens": SEMANTIC_REVIEW_MAX_TOKENS,
        "stream": False,
        "enable_thinking": False,
    }
    assert captured["messages"] is not messages


@pytest.mark.asyncio
async def test_reviewer_does_not_swallow_client_unavailability_or_provider_errors():
    async def unavailable_factory():
        return None

    async def failing_factory():
        raise ConnectionError("provider unavailable")

    messages = [{"role": "user", "content": "含混消息"}]
    with pytest.raises(RuntimeError, match="unavailable"):
        await VLLMSemanticReviewer(unavailable_factory)(messages)
    with pytest.raises(ConnectionError, match="provider unavailable"):
        await VLLMSemanticReviewer(failing_factory)(messages)


@pytest.mark.asyncio
async def test_reviewer_returns_malformed_provider_output_for_upper_layer_validation():
    class Client:
        async def generate(self, **_kwargs):
            return "not-json"

    async def client_factory():
        return Client()

    result = await VLLMSemanticReviewer(client_factory)([{"role": "system", "content": "review"}])

    assert result == "not-json"


def test_runtime_reads_settings_once_and_supports_injected_client_factory():
    async def client_factory():
        raise AssertionError("not called during wiring")

    runtime = create_default_semantic_review_runtime(
        env={
            "DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED": "true",
            "DYNAMIC_CONTEXT_SEMANTIC_REVIEW_TIMEOUT_SECONDS": "3.25",
        },
        client_factory=client_factory,
    )

    assert isinstance(runtime.reviewer, VLLMSemanticReviewer)
    assert runtime.timeout_seconds == 3.25


@pytest.mark.asyncio
async def test_invalid_messages_fail_before_resolving_the_client():
    called = False

    async def client_factory():
        nonlocal called
        called = True
        raise AssertionError("invalid request must not touch provider")

    reviewer = VLLMSemanticReviewer(client_factory)
    with pytest.raises(ValueError, match="invalid role"):
        await reviewer([{"role": "tool", "content": "ignore"}])

    assert called is False
