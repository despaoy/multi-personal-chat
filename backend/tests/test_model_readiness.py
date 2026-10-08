"""Selected-provider readiness, using complete synthetic protocol responses."""

import asyncio
import inspect
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

from app.readiness import ReadinessProbe
from inference import model_manager
from inference.model_readiness import check_model_readiness


@pytest.fixture
def providers(monkeypatch):
    monkeypatch.setattr(model_manager, "_get_db_config", lambda: {})
    return {
        "openai_compat": model_manager.OpenAICompatProvider(),
        "ollama": model_manager.OllamaProvider(),
        "llama_cpp": model_manager.LlamaCppProvider(),
        "transformers_peft": model_manager.TransformersPeftProvider(),
        "mock": model_manager.MockProvider(),
        "vllm": model_manager.VLLMProvider(),
    }


def select(monkeypatch, provider):
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: SimpleNamespace(get_current_provider=lambda: provider))


def install_http(monkeypatch, provider, handler):
    @asynccontextmanager
    async def acquire(timeout):
        assert timeout == 3.0
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            yield client

    monkeypatch.setattr(provider, "_acquire_http_client", acquire)


@pytest.mark.asyncio
@pytest.mark.parametrize("base", ["https://fiction.example", "https://fiction.example/v1/"])
async def test_openai_readiness_uses_refreshed_generation_configuration(monkeypatch, providers, base):
    provider = providers["openai_compat"]
    monkeypatch.setattr(model_manager, "_get_db_config", lambda: {
        "openaiCompatBaseUrl": base, "openaiCompatApiKey": "fiction-readiness-key", "openaiCompatModel": "fiction-model",
    })
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == "https://fiction.example/v1/models"
        assert request.method == "GET" and request.content == b""
        assert request.headers["Authorization"] == "Bearer fiction-readiness-key"
        assert request.extensions["timeout"]["read"] == 3.0
        return httpx.Response(200, json={"object": "list", "data": [{"id": "fiction-model", "object": "model"}]})

    select(monkeypatch, provider)
    install_http(monkeypatch, provider, handler)
    assert await check_model_readiness() is True
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status,payload,expected", [
    (401, {"error": "fiction-private-diagnostic"}, "HTTPStatusError"),
    (503, {"error": "fiction-private-diagnostic"}, "HTTPStatusError"),
    (200, {"data": [{"id": "other-model"}]}, "unavailable"),
    (200, {"data": []}, "unavailable"),
    (200, {"data": [{"name": "fiction-model"}]}, "ValueError"),
    (200, {"data": [{"id": "fiction-model"}, None]}, "ValueError"),
    (200, {"error": "fiction-private-diagnostic"}, "ValueError"),
])
async def test_openai_failure_never_becomes_ready(monkeypatch, providers, status, payload, expected):
    provider = providers["openai_compat"]
    provider.base_url = "https://fiction.example"
    provider.api_key = "fiction-readiness-key"
    provider.model = "fiction-model"
    select(monkeypatch, provider)
    install_http(monkeypatch, provider, lambda request: httpx.Response(status, json=payload))
    probe = ReadinessProbe(database_check=lambda: True, model_check=check_model_readiness, model_required=True)
    try:
        result = await probe.get()
    finally:
        probe.shutdown()
    assert result["ready"] is False and result["deps"] == {"database": True, "model": False}
    assert result["details"]["model"] == expected
    assert "fiction-private" not in str(result) and "fiction-readiness-key" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault,expected", [("connect", "ConnectError"), ("json", "JSONDecodeError"), ("timeout", "TimeoutError")])
async def test_remote_failure_and_timeout_are_explicit_and_cancelled(monkeypatch, providers, fault, expected):
    provider = providers["openai_compat"]
    provider.api_key = "fiction-readiness-key"
    events = []

    async def handler(request):
        if fault == "connect":
            raise httpx.ConnectError("fiction-private-diagnostic", request=request)
        if fault == "json":
            return httpx.Response(200, text="not json")
        try:
            await asyncio.Event().wait()
        finally:
            events.append("cancelled")

    select(monkeypatch, provider)
    install_http(monkeypatch, provider, handler)
    probe = ReadinessProbe(database_check=lambda: True, model_check=check_model_readiness, model_required=True, check_timeout=.05, wait_timeout=.5)
    try:
        result = await probe.get()
    finally:
        probe.shutdown()
    assert result["ready"] is False and result["details"]["model"] == expected
    if fault == "timeout":
        assert events == ["cancelled"]


def test_missing_key_and_configuration_read_failure_propagate(monkeypatch, providers):
    provider = providers["openai_compat"]
    provider.api_key = ""
    select(monkeypatch, provider)
    with pytest.raises(ValueError, match="API key"):
        check_model_readiness()

    def failed_config():
        raise OSError("fiction-database-outage")

    monkeypatch.setattr(model_manager, "_get_db_config", failed_config)
    with pytest.raises(OSError, match="fiction-database-outage"):
        check_model_readiness()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["ollama", "llama_cpp"])
async def test_other_remote_providers_probe_their_protocol(monkeypatch, providers, name):
    provider = providers[name]
    provider.base_url = "http://fiction.example/"
    requests = []

    def handler(request):
        requests.append(request)
        if name == "ollama":
            assert request.method == "POST" and request.url.path == "/api/show"
            assert request.read() == b'{"model":"qwen2.5:7b"}'
            return httpx.Response(200, json={"details": {"family": "qwen2"}})
        assert request.method == "GET" and request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    select(monkeypatch, provider)
    install_http(monkeypatch, provider, handler)
    assert await check_model_readiness() is True
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("client_available,healthy", [(False, 0), (True, 0), (True, 1)])
async def test_vllm_uses_chat_initialization_gate(monkeypatch, providers, client_available, healthy):
    from api import generate

    async def health_check():
        return {"summary": {"healthy": healthy}}

    async def get_client():
        return SimpleNamespace(health_check=health_check) if client_available else None

    monkeypatch.setattr(generate, "get_vllm_client", get_client)
    select(monkeypatch, providers["vllm"])
    assert await check_model_readiness() is bool(client_available and healthy)


def test_check_tracks_runtime_selection_without_loading_local_weights(monkeypatch, providers):
    current = [providers["mock"]]
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: SimpleNamespace(get_current_provider=lambda: current[0]))
    assert check_model_readiness() is False
    current[0] = providers["transformers_peft"]
    assert check_model_readiness() is False
    current[0]._loaded = True
    assert check_model_readiness() is True
    current[0] = providers["mock"]
    assert check_model_readiness() is False
    assert not inspect.isawaitable(check_model_readiness())


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_default_app_checks_selected_provider_when_vllm_is_disabled(monkeypatch, providers, failure):
    from fastapi import HTTPException

    from app.main import create_app, readiness_check
    from app.runtime import RuntimeContainer

    provider = providers["openai_compat"]
    provider.api_key = "fiction-readiness-key"
    provider.model = "fiction-model"
    select(monkeypatch, provider)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(401 if failure else 200, json={"data": [{"id": "fiction-model"}]})

    install_http(monkeypatch, provider, handler)
    monkeypatch.setenv("VLLM_ENABLED", "false")
    app = create_app(RuntimeContainer(
        db=SimpleNamespace(execute_sql=lambda query: [{"ok": 1}]), is_pg_mode=lambda: False,
        startup_env={"SECURITY_MIDDLEWARE_ENABLED": "false", "MODEL_PROVIDER": "openai_compat", "VLLM_ENABLED": "false"},
    ))
    try:
        if failure:
            with pytest.raises(HTTPException) as caught:
                await readiness_check(SimpleNamespace(app=app))
            assert caught.value.status_code == 503
            assert caught.value.detail["details"]["model"] == "HTTPStatusError"
        else:
            result = await readiness_check(SimpleNamespace(app=app))
            assert result["status"] == "ready" and result["deps"] == {"database": True, "model": True}
        assert len(requests) == 1
    finally:
        app.state.readiness_probe.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["ollama", "llama_cpp"])
async def test_other_remote_failure_is_not_ready(monkeypatch, providers, name):
    provider = providers[name]
    select(monkeypatch, provider)
    install_http(monkeypatch, provider, lambda request: httpx.Response(503, json={"error": "unavailable"}))
    with pytest.raises(httpx.HTTPStatusError):
        await check_model_readiness()
