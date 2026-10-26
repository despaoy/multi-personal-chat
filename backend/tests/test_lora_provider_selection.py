"""Selected-provider ownership and explicit adapter failures; no real inference."""

from services import chat_answer

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from provider_fixtures import cloud_connection

from db.schemas import MessageRequest
from services.chat_answer import ChatAnswer

pytestmark = pytest.mark.usefixtures("no_live_context_review")


def test_selected_provider_only_configures_its_own_adapter():
    from inference.model_manager import ModelManager, ModelProvider, OpenAICompatProvider

    setter = Mock()
    manager = ModelManager.__new__(ModelManager)
    manager._providers = {
        ModelProvider.OPENAI_COMPAT: OpenAICompatProvider(dict(
            openaiCompatBaseUrl="https://isolated-provider.invalid/v1",
            openaiCompatApiKey="synthetic-unused-key", openaiCompatModel="synthetic-selected-model",
        )),
        ModelProvider.TRANSFORMERS_PEFT: SimpleNamespace(set_lora_adapter=setter),
    }
    manager._current_provider = ModelProvider.OPENAI_COMPAT
    manager.get_current_provider().set_lora_adapter(None)
    with pytest.raises(ValueError, match="does not support LoRA"):
        manager.get_current_provider().set_lora_adapter("controlled-adapter")
    setter.assert_not_called()
    assert manager._current_provider == ModelProvider.OPENAI_COMPAT
    manager._current_provider = ModelProvider.TRANSFORMERS_PEFT
    manager.get_current_provider().set_lora_adapter("controlled-adapter")
    setter.assert_called_once_with("controlled-adapter")


@pytest.mark.parametrize("stage", ["scan", "database"])
def test_adapter_lookup_propagates_the_original_read_failure(monkeypatch, stage):
    from db import database

    failure = OSError("Declared adapter storage read failure")

    def fail():
        raise failure

    class Database:
        @property
        def loras(self):
            raise failure

    monkeypatch.setattr(database, "refresh_lora_dir_map", fail if stage == "scan" else lambda: {})
    with pytest.raises(OSError) as caught:
        database.get_lora_path_by_id("registered-adapter", loras=[] if stage == "scan" else Database().loras)
    assert caught.value is failure


def install_api(monkeypatch, provider):
    from api import generate
    from inference import model_manager

    setter, inactive = Mock(), Mock()
    manager = SimpleNamespace(
        _current_provider=SimpleNamespace(value=provider), set_lora_adapter=setter,
        set_provider=Mock(side_effect=AssertionError("A request must not change the configured provider")),
        _providers={model_manager.ModelProvider.TRANSFORMERS_PEFT: SimpleNamespace(set_lora_adapter=inactive)},
        get_status=lambda: dict(currentProvider=provider, providers={provider: dict(modelName="controlled-selected-model")}),
    )
    row = dict(id="registered-adapter", name="controlled-adapter", status="active")
    connection = cloud_connection("controlled-selected-model")
    connection.name = provider
    connection.set_lora_adapter = setter
    manager.get_current_provider = lambda: connection
    manager._providers[model_manager.ModelProvider(provider)] = connection
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    monkeypatch.setattr(generate, "db", SimpleNamespace(config={}, loras=[row]))
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "circuit_breaker_registry", None)
    monkeypatch.setattr(generate, "get_lora_character_id", lambda _: None)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: asyncio.Semaphore(1))
    generation = AsyncMock(return_value=ChatAnswer(reply='根据完整当前输入回答。', used_rag=False, model_invoked=True, metadata={}))
    save = AsyncMock(return_value=True)
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', generation)
    monkeypatch.setattr(generate, "_save_message", save)
    return SimpleNamespace(api=generate, manager=manager, generation=generation, save=save, inactive=inactive)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai_compat", "ollama", "llama_cpp"])
@pytest.mark.parametrize("selection", ["explicit", "active"])
async def test_unsupported_selected_adapter_stops_before_lookup_or_model(monkeypatch, provider, selection):
    state = install_api(monkeypatch, provider)
    lookup = Mock(side_effect=AssertionError("Unsupported provider must be rejected before adapter storage lookup"))
    monkeypatch.setattr(state.api, "get_lora_path_by_id", lookup)
    request = MessageRequest(message="根据本题完整数据计算3张票、每张2元的总额。",
                             loraName="controlled-adapter" if selection == "explicit" else "")
    with pytest.raises(HTTPException) as caught:
        await state.api._generate_reply_impl(request, record_invocation=False, message_db=state.api.db)
    assert caught.value.status_code == 422 and "不支持 LoRA" in caught.value.detail
    assert state.manager._current_provider.value == provider
    lookup.assert_not_called()
    state.manager.set_provider.assert_not_called()
    state.manager.set_lora_adapter.assert_not_called()
    state.inactive.assert_not_called()
    state.generation.assert_not_awaited()
    state.save.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("availability", ["ready", "missing", "read_failure"])
async def test_explicit_local_provider_requires_its_registered_adapter(tmp_path, monkeypatch, availability):
    state = install_api(monkeypatch, "transformers_peft")
    adapter = tmp_path / "controlled-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"controlled-unit-weights")
    failure = OSError("Declared adapter directory failure")
    lookup = Mock(side_effect=failure) if availability == "read_failure" else Mock(return_value=str(adapter) if availability == "ready" else None)
    monkeypatch.setattr(state.api, "get_lora_path_by_id", lookup)
    request = MessageRequest(message="根据完整当前输入回答并保留全部条件。", loraName="controlled-adapter")
    if availability == "ready":
        response = await state.api._generate_reply_impl(request, record_invocation=False, message_db=state.api.db)
        assert response.reply == "根据完整当前输入回答。"
        state.manager.set_lora_adapter.assert_called_once_with(str(adapter))
        state.generation.assert_awaited_once()
        state.save.assert_awaited_once()
    else:
        with pytest.raises(HTTPException) as caught:
            await state.api._generate_reply_impl(request, record_invocation=False, message_db=state.api.db)
        assert caught.value.status_code == (503 if availability == "read_failure" else 409)
        if availability == "read_failure":
            assert caught.value.__cause__ is failure
        state.manager.set_lora_adapter.assert_not_called()
        state.generation.assert_not_awaited()
        state.save.assert_not_awaited()
    lookup.assert_called_once_with("registered-adapter", loras=[dict(id="registered-adapter", name="controlled-adapter", status="active")])
    assert state.manager._current_provider.value == "transformers_peft"
    state.manager.set_provider.assert_not_called()
    state.inactive.assert_not_called()
