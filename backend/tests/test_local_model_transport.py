"""Local-only transports cannot reuse proxy-enabled clients or follow redirects."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from character.memory_llm import MemoryLlmConfig, OpenAICompatibleMemoryCompletion
from inference.model_manager import MockProvider
from inference.request_context import InferenceRequestContext, use_request_context
from inference.vllm_client import VLLMClient


@pytest.fixture
def transports(monkeypatch):
    original = httpx.AsyncClient
    created, sent = [], []

    def respond(request):
        sent.append(request)
        return httpx.Response(307, headers={'location': 'https://cloud.invalid/upload'})

    def factory(**kwargs):
        client = original(**kwargs, transport=httpx.MockTransport(respond))
        created.append((client, kwargs))
        return client

    monkeypatch.setattr(httpx, 'AsyncClient', factory)
    monkeypatch.setenv('HTTPS_PROXY', 'http://proxy.invalid:8080')
    return created, sent


@pytest.mark.asyncio
async def test_provider_bypasses_shared_pool_and_redirect(transports):
    created, sent = transports
    pool = SimpleNamespace(acquire=Mock(side_effect=AssertionError('shared proxy client must not be borrowed')))
    provider = MockProvider()
    context = InferenceRequestContext('openai_compat', provider, len, pool, local_only=True)
    with use_request_context(context):
        async with provider._acquire_http_client() as client:
            response = await client.post('http://localhost:8001/v1/chat/completions', json={'private': 'text'})
            assert response.status_code == 307
    assert created[0][1]['trust_env'] is False and created[0][1]['follow_redirects'] is False
    assert created[0][0].is_closed and len(sent) == 1
    pool.acquire.assert_not_called()


@pytest.mark.asyncio
async def test_auxiliary_local_client_rejects_redirect_and_closes(transports):
    created, sent = transports
    writer = OpenAICompatibleMemoryCompletion(MemoryLlmConfig(True, 'http://localhost:8001/v1', 'test'), local_only=True)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await writer.complete([{'role': 'user', 'content': 'private'}])
    finally:
        await writer.close()
    assert created[0][1]['trust_env'] is False and created[0][1]['follow_redirects'] is False
    assert created[0][0].is_closed and len(sent) == 1


@pytest.mark.asyncio
async def test_vllm_separates_clients_and_closes_both(transports):
    created, sent = transports
    owner = VLLMClient(base_urls='http://localhost:8001', model='test')
    normal = await owner._ensure_client()
    with use_request_context(InferenceRequestContext('vllm', owner, len, local_only=True)):
        direct = await owner._ensure_client()
        assert direct is not normal
        assert await owner._ensure_client() is direct
        response = await direct.post('http://localhost:8001/v1/chat/completions', json={'private': 'text'})
        assert response.status_code == 307
    assert await owner._ensure_client() is normal
    await owner.close()
    assert len(created) == 2 and all(client.is_closed for client, _ in created)
    assert created[1][1]['trust_env'] is False and created[1][1]['follow_redirects'] is False
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_vllm_close_attempts_local_client_after_normal_client_failure():
    owner = VLLMClient(base_urls='http://localhost:8001', model='test')
    error = RuntimeError('close failed')
    normal = SimpleNamespace(is_closed=False, aclose=AsyncMock(side_effect=error))
    direct = SimpleNamespace(is_closed=False, aclose=AsyncMock())
    owner._client, owner._local_client = normal, direct
    with pytest.raises(RuntimeError) as caught:
        await owner.close()
    assert caught.value is error
    normal.aclose.assert_awaited_once()
    direct.aclose.assert_awaited_once()
