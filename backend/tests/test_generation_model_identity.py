"""Reply identity depends on the requested provider, not every configured provider."""

from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from db.schemas import MessageRequest
from inference.model_manager import ModelManager, ModelProvider
from test_lora_provider_selection import install_api

pytestmark = pytest.mark.usefixtures("no_live_context_review")


def request():
    return MessageRequest(
        message="三张票每张两元，请计算总额。",
        loraName="default",
        platform="web",
        adapter="synthetic-identity",
        senderId="synthetic-owner",
        userId="synthetic-owner",
        sessionId="synthetic-identity-session",
        sourceMessageId="synthetic-identity-question",
    )


def rig(monkeypatch, provider="ollama"):
    state = install_api(monkeypatch, provider)
    selected = state.manager.get_current_provider()
    selected.model = "selected-model"
    selected.get_status = Mock(return_value={"modelName": "selected-model"})
    inactive = SimpleNamespace(get_status=Mock(side_effect=OSError("synthetic-private-inactive-status")))
    state.manager._providers = {
        ModelProvider(provider): selected,
        ModelProvider.LLAMA_CPP if provider != "llama_cpp" else ModelProvider.OLLAMA: inactive,
    }
    state.manager.get_status = MethodType(ModelManager.get_status, state.manager)
    state.selected, state.other = selected, inactive
    state.record = AsyncMock()
    monkeypatch.setattr(state.api, "_record_model_invocation", state.record)
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["ollama", "llama_cpp", "transformers_peft"])
async def test_inactive_status_fault_cannot_block_selected_provider_reply(monkeypatch, provider):
    state = rig(monkeypatch, provider)
    result = await state.api._generate_reply_impl(request())
    assert result.reply == "根据完整当前输入回答。" and result.model == f"selected-model ({provider})"
    state.selected.get_status.assert_called_once()
    state.other.get_status.assert_not_called()
    state.generation.assert_awaited_once()
    state.save.assert_awaited_once()
    assert state.save.await_args.args[2] == "selected-model"
    assert state.record.await_args.args[1] == "selected-model" and not state.record.await_args.kwargs["error_type"]
    state.manager.set_provider.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [{}, None, {"modelName": None}, {"modelName": ""}, {"modelName": "   "}, {"modelName": 123}]
)
async def test_missing_or_invalid_identity_never_becomes_a_successful_unknown_model(monkeypatch, caplog, status):
    state = rig(monkeypatch)
    state.selected.get_status.return_value = status
    with pytest.raises(HTTPException) as caught:
        await state.api._generate_reply_impl(request())
    assert caught.value.status_code == 503
    assert caught.value.detail["stage"] == "model_status"
    assert caught.value.detail["modelInvoked"] and not caught.value.detail["messageSaved"]
    assert caught.value.__cause__ is not None and caught.value.__cause__.__cause__ is not None
    assert "synthetic-private" not in str(caught.value.detail) + caplog.text
    state.generation.assert_awaited_once()
    state.save.assert_not_awaited()
    state.record.assert_awaited_once()
    assert state.record.await_args.kwargs["error_type"]
    state.other.get_status.assert_not_called()


@pytest.mark.asyncio
async def test_selected_status_read_failure_preserves_original_cause_and_failure_audit(monkeypatch, caplog):
    state = rig(monkeypatch)
    failure = OSError("synthetic-private-selected-status")
    state.selected.get_status.side_effect = failure
    with pytest.raises(HTTPException) as caught:
        await state.api._generate_reply_impl(request())
    assert caught.value.status_code == 503 and caught.value.__cause__.__cause__ is failure
    assert "synthetic-private" not in str(caught.value.detail) + caplog.text
    state.record.assert_awaited_once()
    state.save.assert_not_awaited()
    state.other.get_status.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["ollama", "openai_compat"])
async def test_result_without_model_invocation_does_not_need_model_status(monkeypatch, provider):
    state = rig(monkeypatch, provider)
    state.selected.get_status.side_effect = OSError("Status is irrelevant to a deterministic result")
    state.generation.return_value = (
        "已依据当前存储记录完成查询。",
        False,
        {"modelInvoked": False, "answerMode": "memory_lookup"},
    )
    result = await state.api._generate_reply_impl(request())
    assert result.model == "memory/lookup" and not result.modelInvoked
    state.selected.get_status.assert_not_called()
    state.other.get_status.assert_not_called()
    state.record.assert_not_awaited()
    assert state.save.await_args.args[2] == "memory/lookup" and state.save.await_args.kwargs["model_invoked"] is False


@pytest.mark.asyncio
async def test_local_identity_uses_request_provider_even_if_selection_changes_after_generation(monkeypatch):
    state = rig(monkeypatch)

    async def generated(*args, **kwargs):
        state.manager._current_provider = SimpleNamespace(value="llama_cpp")
        return "根据完整当前输入回答。", False, {}

    state.generation.side_effect = generated
    result = await state.api._generate_reply_impl(request())
    assert result.model == "selected-model (ollama)"
    state.other.get_status.assert_not_called()


@pytest.mark.asyncio
async def test_cloud_identity_uses_bound_snapshot_after_runtime_selection_changes(monkeypatch):
    state = rig(monkeypatch, "openai_compat")

    async def generated(*args, **kwargs):
        state.manager._current_provider = SimpleNamespace(value="llama_cpp")
        return "根据完整当前输入回答。", False, {}

    state.generation.side_effect = generated
    result = await state.api._generate_reply_impl(request())
    assert result.model == "selected-model (openai_compat)"
    state.other.get_status.assert_not_called()
    state.selected.get_status.assert_called_once()
