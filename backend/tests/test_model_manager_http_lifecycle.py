from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from app import config as app_config
from inference.model_manager import BaseProvider
from provider_fixtures import completed_provider_response


class DummyProvider(BaseProvider):
    def __init__(self):
        super().__init__("dummy")

    def generate(self, prompt, session_history=None, max_tokens_override=None):
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
    monkeypatch.setattr(model_manager, '_get_db_config', lambda _owner_database=None: {'openaiCompatBaseUrl': 'https://fiction.example', 'openaiCompatApiKey': 'synthetic-test-key', 'openaiCompatModel': 'fiction-model'})
    provider = getattr(model_manager, kind)()
    if mode == 'http_error':
        response = httpx.Response(502, text='private-response-body')
    else:
        response = httpx.Response(200, json=completed_provider_response(kind, '有效回答'))
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


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['OllamaProvider', 'LlamaCppProvider'])
@pytest.mark.parametrize('fault', ['healthy', 'empty', 'partial', 'limit', 'missing_completion'])
async def test_local_provider_accepts_only_complete_text(monkeypatch, kind, fault):
    from inference import model_manager
    monkeypatch.setattr(model_manager, '_get_db_config', lambda _owner_database=None: {'maxTokens': 64, 'temperature': 0.2})
    provider = getattr(model_manager, kind)()
    payload = completed_provider_response(kind, '6元；尚未付款，下雨取消。')
    content = payload['message'] if kind == 'OllamaProvider' else payload['choices'][0]['message']
    completion = payload if kind == 'OllamaProvider' else payload['choices'][0]
    reason = 'done_reason' if kind == 'OllamaProvider' else 'finish_reason'
    if fault == 'empty':
        content['content'] = ' \n\t '
    elif fault == 'partial':
        if kind == 'OllamaProvider':
            payload['done'] = False
        else:
            completion[reason] = None
    elif fault == 'limit':
        completion[reason] = 'length'
    elif fault == 'missing_completion':
        del payload['done_reason' if kind == 'OllamaProvider' else '__verbose']
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=payload)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:

        @asynccontextmanager
        async def acquire(timeout):
            yield http
        monkeypatch.setattr(provider, '_acquire_http_client', acquire)
        prompt = '完整虚构题设：三张票每张2元，尚未付款，下雨取消。请计算总额并保留条件。'
        if fault == 'healthy':
            assert (await provider.async_generate(prompt))[0] == '6元；尚未付款，下雨取消。'
        else:
            with pytest.raises(KeyError if fault == 'missing_completion' else ValueError):
                await provider.async_generate(prompt)
    assert len(requests) == 1 and http.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize('metadata', [{'truncated': True}, {}])
async def test_llama_chat_rejects_lost_context(monkeypatch, metadata):
    from inference import model_manager
    monkeypatch.setattr(model_manager, '_get_db_config', lambda _owner_database=None: {'maxTokens': 64, 'temperature': 0.2})
    provider = model_manager.LlamaCppProvider()
    payload = completed_provider_response('LlamaCppProvider', '完整受控回答')
    payload['__verbose'] = metadata
    post = AsyncMock(return_value=httpx.Response(200, json=payload))

    @asynccontextmanager
    async def acquire(timeout):
        yield SimpleNamespace(post=post)
    monkeypatch.setattr(provider, '_acquire_http_client', acquire)
    with pytest.raises(ValueError if metadata else KeyError):
        await provider.async_generate('完整题设：预约尚未付款，改期需要书面同意。请核对付款及改期条件。')
    assert post.await_count == 1
