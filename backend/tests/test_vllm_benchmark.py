"""Isolated transport checks for the direct vLLM load generator."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from benchmarks import vllm_benchmark as benchmark


def install_transport(monkeypatch, handler):
    real_client = httpx.AsyncClient
    clients = []

    def factory(*args, **kwargs):
        client = real_client(*args, **kwargs, transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    monkeypatch.setattr(benchmark.httpx, "AsyncClient", factory)
    monkeypatch.setattr(benchmark.asyncio, "sleep", AsyncMock())
    monkeypatch.setenv("VLLM_URL", "http://synthetic-vllm.test/v1")
    monkeypatch.setenv("VLLM_BENCHMARK_MODEL_ID", "synthetic-benchmark-model")
    return clients


def completion(content="完整回复", finish_reason="stop"):
    return {
        "id": "synthetic-response",
        "object": "chat.completion",
        "model": "synthetic-benchmark-model",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    }


@pytest.mark.asyncio
async def test_completed_load_reuses_connection_owner_and_varies_messages(monkeypatch):
    messages = []

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "synthetic-benchmark-model"
        messages.append(body["messages"][0]["content"])
        return httpx.Response(200, json=completion())

    clients = install_transport(monkeypatch, handler)
    assert await benchmark.main() == 0
    assert len(messages) == 81 and set(messages) == set(benchmark.MESSAGES)
    assert len(clients) == 1 and clients[0].is_closed


@pytest.mark.asyncio
async def test_transport_protocol_and_unfinished_replies_remain_failures(monkeypatch):
    calls = 0

    def handler(request):
        nonlocal calls
        index = calls % 5
        calls += 1
        if index == 4:
            raise httpx.ConnectTimeout("synthetic timeout", request=request)
        if index == 3:
            return httpx.Response(503, json={"error": "synthetic unavailable"})
        return httpx.Response(
            200, json=completion("" if index == 2 else "完整回复", "length" if index == 1 else "stop")
        )

    clients = install_transport(monkeypatch, handler)
    results = await benchmark.test_vllm_direct()
    outcomes = [outcome for result in results for outcome in result.outcomes]
    assert len(outcomes) == 81 and sum(outcome.success for outcome in outcomes) == 17
    assert sum(not outcome.success and outcome.status_code == 200 for outcome in outcomes) == 32
    assert sum(outcome.status_code == 503 for outcome in outcomes) == 16
    assert sum(outcome.status_code == 0 for outcome in outcomes) == 16
    assert all(outcome.error for outcome in outcomes if not outcome.success)
    assert len(clients) == 1 and clients[0].is_closed


@pytest.mark.asyncio
async def test_unexpected_error_propagates_and_closes_client(monkeypatch):
    def handler(request):
        raise RuntimeError("synthetic programming failure")

    clients = install_transport(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="synthetic programming failure"):
        await benchmark.main()
    assert len(clients) == 1 and clients[0].is_closed
