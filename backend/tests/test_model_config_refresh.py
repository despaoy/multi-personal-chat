"""Configuration authority and refresh with isolated persistence and transports."""

import json
import sys
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

from api import config as config_api
from db import adapter
from db.database import SQLiteDB
from inference import model_manager

STORED = {
    "openaiCompatBaseUrl": "https://stored.example/v1", "openaiCompatApiKey": "fiction-stored-key",
    "openaiCompatModel": "stored-model", "temperature": 0.37, "maxTokens": 733,
}


@pytest.fixture
def configured(monkeypatch, tmp_path):
    database = SQLiteDB(tmp_path / "config.db")
    database.update_config(STORED)
    monkeypatch.setattr(adapter, "db", database)
    monkeypatch.setattr(config_api, "db", database)
    # These provider tests isolate refresh/transport behavior from cache expiry.
    monkeypatch.setattr(model_manager, "_get_db_config", lambda: database.config)
    monkeypatch.setenv("MODEL_PROVIDER", "openai_compat")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://environment.example")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "fiction-environment-key")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "environment-model")
    manager = model_manager.ModelManager(base_dir=tmp_path)
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    try:
        yield database, manager
    finally:
        manager.shutdown()
        database.close_connection()


def request(body):
    return SimpleNamespace(json=AsyncMock(return_value=body))


@pytest.mark.asyncio
async def test_config_update_and_provider_switch_use_the_same_resolver(configured):
    database, manager = configured
    provider = manager.get_current_provider()
    result = await config_api.update_config(request({"openaiCompatModel": "new-model"}), current_user={"isAdmin": True})
    assert result["success"] and database.config["openaiCompatModel"] == "new-model"
    assert provider.model == provider.get_status()["modelName"] == "new-model"
    assert provider.api_key == STORED["openaiCompatApiKey"] and "fiction-stored-key" not in str(result)
    database.update_config({"openaiCompatModel": "switched-model"})
    switched = await config_api.set_model_provider(request({"provider": "openai_compat"}), current_user={"isAdmin": True})
    assert switched["success"] and manager.get_current_provider() is provider
    assert provider.model == "switched-model" and switched["status"]["providers"]["openai_compat"]["modelName"] == "switched-model"


@pytest.mark.asyncio
async def test_cleared_database_fields_re_resolve_environment_instead_of_old_memory(configured):
    _, manager = configured
    provider = manager.get_current_provider()
    body = {key: "" for key in ("openaiCompatBaseUrl", "openaiCompatApiKey", "openaiCompatModel")}
    result = await config_api.update_config(request(body), current_user={"isAdmin": True})
    assert result["success"]
    assert (provider.base_url, provider.api_key, provider.model) == (
        "https://environment.example", "fiction-environment-key", "environment-model")


@pytest.mark.asyncio
async def test_cleared_key_without_environment_cannot_use_the_previous_key(configured, monkeypatch):
    _, manager = configured
    provider = manager.get_current_provider()
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY")
    await config_api.update_config(request({"openaiCompatApiKey": ""}), current_user={"isAdmin": True})
    assert provider.api_key == ""
    with pytest.raises(RuntimeError, match="API Key"):
        await provider.async_generate("完整的合成问题")


@pytest.mark.asyncio
@pytest.mark.parametrize("switch", [False, True])
async def test_runtime_sync_failure_reports_persisted_partial_state(configured, monkeypatch, switch):
    database, manager = configured
    provider = manager.get_current_provider()
    failure = OSError("fiction-private-config-failure")

    def fail():
        raise failure

    monkeypatch.setattr(provider, "_refresh_db_config", fail)
    operation = config_api.set_model_provider if switch else config_api.update_config
    body = {"provider": "openai_compat"} if switch else {"openaiCompatModel": "persisted-model"}
    with pytest.raises(HTTPException) as error:
        await operation(request(body), current_user={"isAdmin": True})
    assert error.value.status_code == 503 and error.value.__cause__ is failure
    assert error.value.detail["configPersisted"] and error.value.detail["stage"] == "runtime_config"
    if switch:
        assert error.value.detail["providerSwitched"] and database.config["modelProvider"] == "openai_compat"
    else:
        assert database.config["openaiCompatModel"] == "persisted-model"
    assert "fiction-private" not in str(error.value.detail)


@pytest.mark.asyncio
async def test_request_keeps_its_connection_snapshot_during_runtime_refresh(configured, monkeypatch):
    database, manager = configured
    provider = manager.get_current_provider()
    calls = []

    def respond(raw):
        calls.append((str(raw.url), raw.headers["Authorization"], json.loads(raw.content)))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "完整回答"}}]})

    refreshed = False

    @asynccontextmanager
    async def acquire(timeout):
        nonlocal refreshed
        if not refreshed:
            database.update_config({"openaiCompatBaseUrl": "https://next.example", "openaiCompatApiKey": "fiction-next-key",
                                    "openaiCompatModel": "next-model", "temperature": 0.12})
            provider._refresh_db_config()
            refreshed = True
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            yield client

    monkeypatch.setattr(provider, "_acquire_http_client", acquire)
    for _ in range(2):
        assert (await provider.async_generate("完整的合成问题"))[0] == "完整回答"
    assert calls[0][0] == "https://stored.example/v1/chat/completions" and calls[0][1] == "Bearer fiction-stored-key"
    assert calls[0][2]["model"] == "stored-model" and calls[0][2]["temperature"] == 0.37
    assert calls[1][0] == "https://next.example/v1/chat/completions" and calls[1][1] == "Bearer fiction-next-key"
    assert calls[1][2]["model"] == "next-model" and calls[1][2]["temperature"] == 0.12


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["OpenAICompatProvider", "OllamaProvider", "LlamaCppProvider"])
async def test_http_generation_refreshes_parameters_and_propagates_config_read_failure(configured, monkeypatch, kind):
    database, _ = configured
    provider = getattr(model_manager, kind)()
    calls = []

    def respond(raw):
        calls.append(json.loads(raw.content))
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": "完整回答"}}],
            "message": {"content": "完整回答"}, "content": "完整回答",
        })

    @asynccontextmanager
    async def acquire(timeout):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            yield client

    monkeypatch.setattr(provider, "_acquire_http_client", acquire)
    for temperature, tokens in [(0.37, 733), (0.12, 377)]:
        database.update_config({"temperature": temperature, "maxTokens": tokens})
        assert (await provider.async_generate("完整的合成问题"))[0] == "完整回答"
        body = calls[-1]
        if kind == "OllamaProvider":
            assert body["options"]["temperature"] == temperature and body["options"]["num_predict"] == tokens
        else:
            assert body["temperature"] == temperature and body["max_tokens" if kind == "OpenAICompatProvider" else "n_predict"] == tokens
    failure = OSError("fiction-config-read-failure")

    def fail():
        raise failure

    monkeypatch.setattr(model_manager, "_get_db_config", fail)
    with pytest.raises(OSError) as error:
        await provider.async_generate("完整的合成问题")
    assert error.value is failure and len(calls) == 2


@pytest.mark.asyncio
async def test_vllm_generation_refreshes_parameters_without_connection_db_overrides(configured, monkeypatch):
    from inference import vllm_client

    database, _ = configured
    provider = model_manager.VLLMProvider()
    connection = (provider.base_url, provider.model, provider.timeout)
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return "完整回答"

    async def get_client():
        return SimpleNamespace(generate=generate)

    monkeypatch.setattr(vllm_client, "get_vllm_client", get_client)
    for temperature, tokens in [(0.37, 733), (0.12, 377)]:
        database.update_config({"temperature": temperature, "maxTokens": tokens, "vllmBaseUrl": "https://unused.example"})
        assert (await provider.async_generate("完整的合成问题"))[0] == "完整回答"
        assert calls[-1]["temperature"] == temperature and calls[-1]["max_tokens"] == tokens
    assert (provider.base_url, provider.model, provider.timeout) == connection
    failure = OSError("fiction-config-read-failure")

    def fail():
        raise failure

    monkeypatch.setattr(model_manager, "_get_db_config", fail)
    with pytest.raises(OSError) as error:
        await provider.async_generate("完整的合成问题")
    assert error.value is failure and len(calls) == 2


def test_loaded_local_generation_reads_current_parameters_without_loading_weights(configured, monkeypatch):
    database, _ = configured
    provider = model_manager.TransformersPeftProvider()
    calls = []
    tensor = SimpleNamespace(shape=(1, 2), to=lambda device: tensor)

    def generate(*args, **kwargs):
        calls.append(kwargs)
        return [[1, 2, 3]]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    monkeypatch.setattr(provider, "_ensure_loaded", lambda: None)
    provider._model = SimpleNamespace(device="cpu", generate=generate)
    provider._tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: SimpleNamespace(input_ids=tensor),
                                          decode=lambda *args, **kwargs: "完整回答", eos_token_id=0)
    try:
        for temperature, tokens in [(0.37, 733), (0.12, 377)]:
            database.update_config({"temperature": temperature, "maxTokens": tokens})
            assert provider.generate("完整的合成问题")[0] == "完整回答"
            assert calls[-1]["temperature"] == temperature and calls[-1]["max_new_tokens"] == tokens
        failure = OSError("fiction-config-read-failure")

        def fail():
            raise failure

        monkeypatch.setattr(model_manager, "_get_db_config", fail)
        with pytest.raises(OSError) as error:
            provider.generate("完整的合成问题")
        assert error.value is failure and len(calls) == 2
    finally:
        provider._model = None
