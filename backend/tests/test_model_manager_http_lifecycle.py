from __future__ import annotations

import asyncio

import httpx

from app import config as app_config
from inference.model_manager import BaseProvider


class DummyProvider(BaseProvider):
    def __init__(self):
        super().__init__("dummy")

    def generate(self, prompt, session_history=None, rag_docs=None, max_tokens_override=None):
        return prompt, 0.0


def test_fallback_http_client_closes_on_its_creating_loop(monkeypatch):
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
        async with provider._acquire_http_client(timeout=1.0) as client:
            assert client.closed is False

    asyncio.run(acquire_once())
    asyncio.run(acquire_once())

    assert len(clients) == 2
    assert all(client.closed for client in clients)
