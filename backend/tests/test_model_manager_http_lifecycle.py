from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from app import config as app_config
from inference.model_manager import BaseProvider


class DummyProvider(BaseProvider):
    def __init__(self):
        super().__init__("dummy")

    def generate(self, prompt, session_history=None, rag_docs=None, max_tokens_override=None):
        return prompt, 0.0


@pytest.mark.parametrize("fail", [False, True])
def test_standalone_http_client_closes_on_its_creating_loop(monkeypatch, fail):
    clients = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.closed = False
            clients.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            self.closed = True
            return False

    monkeypatch.setattr(app_config, "http_client_pool", None)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    provider = DummyProvider()

    async def acquire_once():
        with pytest.raises(RuntimeError, match="request failed") if fail else nullcontext():
            async with provider._acquire_http_client(timeout=1.0) as client:
                assert client.closed is False
                if fail:
                    raise RuntimeError("request failed")

    asyncio.run(acquire_once())
    asyncio.run(acquire_once())

    assert len(clients) == 2
    assert all(client.closed for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure_at', [None, 'pool', 'request'])
async def test_shared_pool_owns_client_and_failures_never_create_replacement(monkeypatch, failure_at):
    client = SimpleNamespace(aclose=AsyncMock())
    events = []
    failure = OSError('private-transport-detail')

    @asynccontextmanager
    async def acquire():
        events.append('acquire')
        if failure_at == 'pool':
            raise failure
        try:
            yield client
        finally:
            events.append('release')

    monkeypatch.setattr(app_config, 'http_client_pool', SimpleNamespace(acquire=acquire))
    replacement = Mock(side_effect=AssertionError('replacement client is forbidden'))
    monkeypatch.setattr(httpx, 'AsyncClient', replacement)
    provider = DummyProvider()
    with pytest.raises(OSError) if failure_at else nullcontext():
        async with provider._acquire_http_client() as actual:
            assert actual is client
            if failure_at == 'request':
                raise failure
    provider.close()
    replacement.assert_not_called()
    client.aclose.assert_not_awaited()
    assert events == (['acquire'] if failure_at == 'pool' else ['acquire', 'release'])


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['OpenAICompatProvider', 'OllamaProvider', 'LlamaCppProvider'])
@pytest.mark.parametrize('mode', ['success', 'http_error', 'transport_error'])
async def test_http_provider_errors_propagate_without_private_body_logging(monkeypatch, caplog, kind, mode):
    from inference import model_manager
    monkeypatch.setattr(model_manager, '_get_db_config', lambda: {})
    provider = getattr(model_manager, kind)()
    provider.api_key = 'synthetic-test-key'
    if mode == 'http_error':
        response = httpx.Response(502, text='private-response-body')
    else:
        response = httpx.Response(200, json={
            'choices': [{'finish_reason': 'stop', 'message': {'content': '有效回答'}}],
            'message': {'content': '有效回答'}, 'content': '有效回答'})
    failure = httpx.ConnectError('private-transport-detail')
    post = AsyncMock(side_effect=failure) if mode == 'transport_error' else AsyncMock(return_value=response)
    released = []

    @asynccontextmanager
    async def acquire():
        try:
            yield SimpleNamespace(post=post)
        finally:
            released.append(True)

    monkeypatch.setattr(app_config, 'http_client_pool', SimpleNamespace(acquire=acquire))
    if mode == 'success':
        reply, _ = await provider.async_generate('完整的合成问题')
        assert reply == '有效回答'
    else:
        with pytest.raises(RuntimeError if mode == 'http_error' else httpx.ConnectError) as caught:
            await provider.async_generate('完整的合成问题')
        if mode == 'http_error':
            assert '502' in str(caught.value) and 'private-response-body' not in str(caught.value)
        else:
            assert caught.value is failure
    assert released == [True]
    assert 'private-response-body' not in caplog.text and 'private-transport-detail' not in caplog.text
    assert post.await_count == 1
