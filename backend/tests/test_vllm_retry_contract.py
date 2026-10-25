"""Transient failures retry unchanged requests, never substitute an answer."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from inference.openai_protocol import ProviderResponseError
from inference.vllm_client import VLLMClient


@pytest.fixture
async def rig(monkeypatch):
    client = VLLMClient(
        base_urls="https://vllm.fixture.invalid",
        api_key="synthetic-key",
        model="fiction-model",
        max_retries=2,
    )
    state = SimpleNamespace(client=client, outcomes=[], requests=[])

    def respond(request):
        state.requests.append(request)
        assert client._instances[0].current_connections == 1
        outcome = state.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def backoff(_delay):
        # Backoff is not an active upstream connection for balancing purposes.
        assert client._instances[0].current_connections == 0

    state.sleep = AsyncMock(side_effect=backoff)
    monkeypatch.setattr("inference.vllm_client.asyncio.sleep", state.sleep)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        yield state
    finally:
        await client.close()


def completion():
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "visible answer"},
                }
            ]
        },
    )


def failure(kind):
    if kind == "server":
        return httpx.Response(503, text="private upstream diagnostic")
    return {
        "timeout": httpx.ReadTimeout,
        "connect": httpx.ConnectError,
    }[kind]("private transport diagnostic")


async def generate(rig):
    return await rig.client.generate([{"role": "user", "content": "synthetic request"}], lora_name="fiction-adapter")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["server", "timeout", "connect"])
async def test_transient_failure_retries_same_model_and_payload_once(rig, kind):
    rig.outcomes = [failure(kind), completion()]
    assert await generate(rig) == "visible answer"
    assert len(rig.requests) == 2
    assert rig.requests[0].content == rig.requests[1].content
    assert json.loads(rig.requests[1].content)["model"] == "fiction-adapter"
    rig.sleep.assert_awaited_once()
    instance = rig.client._instances[0]
    assert (instance.current_connections, instance.success_count, instance.failure_count) == (0, 1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["server", "timeout", "connect"])
async def test_exhaustion_raises_last_failure_without_extra_sleep(rig, kind):
    last = failure(kind)
    rig.outcomes = [failure(kind), last]
    expected = ProviderResponseError if kind == "server" else type(last)
    with pytest.raises(expected) as caught:
        await generate(rig)
    if kind == "server":
        assert caught.value.status_code == 503
        assert "private upstream" not in str(caught.value)
    else:
        assert caught.value is last
    assert len(rig.requests) == 2
    rig.sleep.assert_awaited_once()
    instance = rig.client._instances[0]
    assert (instance.current_connections, instance.success_count, instance.failure_count) == (0, 0, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 400, 401, 429])
async def test_http_client_error_never_retries(rig, status):
    rig.outcomes = [httpx.Response(status, text="private upstream diagnostic")]
    with pytest.raises(ProviderResponseError) as caught:
        await generate(rig)
    assert caught.value.status_code == status
    assert "private upstream" not in str(caught.value)
    assert len(rig.requests) == 1
    rig.sleep.assert_not_awaited()
    assert rig.client._instances[0].failure_count == 1
    assert rig.client._instances[0].current_connections == 0


@pytest.mark.asyncio
async def test_malformed_success_never_retries(rig):
    rig.outcomes = [httpx.Response(200, json={"choices": []})]
    with pytest.raises(ValueError, match="invalid chat completion choices"):
        await generate(rig)
    assert len(rig.requests) == 1
    rig.sleep.assert_not_awaited()
    assert rig.client._instances[0].failure_count == 1
    assert rig.client._instances[0].current_connections == 0


@pytest.mark.asyncio
async def test_cancellation_during_backoff_does_not_retry_or_count_another_failure(rig):
    rig.outcomes = [failure("server"), completion()]
    rig.sleep.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await generate(rig)
    assert len(rig.requests) == 1
    assert rig.client._instances[0].failure_count == 1
    assert rig.client._instances[0].current_connections == 0


@pytest.mark.asyncio
async def test_single_attempt_timeout_preserves_error_without_backoff(rig):
    rig.client._max_retries = 1
    error = failure("timeout")
    rig.outcomes = [error]
    with pytest.raises(httpx.ReadTimeout) as caught:
        await generate(rig)
    assert caught.value is error
    assert len(rig.requests) == 1
    rig.sleep.assert_not_awaited()
    instance = rig.client._instances[0]
    assert (instance.current_connections, instance.success_count, instance.failure_count) == (0, 0, 1)


@pytest.mark.asyncio
async def test_cancellation_during_active_post_releases_connection_without_failure(rig):
    started = asyncio.Event()
    never_complete = asyncio.Event()

    async def respond(request):
        rig.requests.append(request)
        started.set()
        await never_complete.wait()
        pytest.fail("Cancelled request unexpectedly resumed")

    await rig.client.close()
    rig.client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    task = asyncio.create_task(generate(rig))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        assert rig.client._instances[0].current_connections == 1
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(rig.requests) == 1
    rig.sleep.assert_not_awaited()
    instance = rig.client._instances[0]
    assert (instance.current_connections, instance.success_count, instance.failure_count) == (0, 0, 0)
    assert rig.client._request_semaphore._value == rig.client._max_concurrency
