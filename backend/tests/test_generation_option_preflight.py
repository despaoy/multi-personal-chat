"""Unsupported cloud settings must fail before paid semantic preparation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from db.schemas import GenerateResponse, MessageRequest
from test_lora_provider_selection import install_api

pytestmark = pytest.mark.usefixtures("no_live_context_review")
ROLE = "tsukiyashiro_kisaki"
MESSAGE = "请根据完整题目信息回答：三张票，每张两元，共多少钱？"


@pytest.fixture
def rig(monkeypatch):
    state = install_api(monkeypatch, "openai_compat")
    state.manager.get_current_provider().base_url = "https://api.deepseek.com"
    # Another database has valid settings. The request must still use its own.
    monkeypatch.setattr(state.api, "db", SimpleNamespace(config={"topP": 1.0}, loras=[]))
    state.prepare = AsyncMock(side_effect=AssertionError("invalid configuration reached semantic preparation"))
    monkeypatch.setattr(state.api, "_prepare_character_turn", state.prepare)
    return state


def message(*, character=True):
    return MessageRequest(
        message=MESSAGE,
        characterId=ROLE if character else None,
        loraName="default",
        platform="web",
        adapter="web-character",
        senderId="synthetic-reader",
        userId="synthetic-reader",
        sessionId="synthetic-option-session",
        sourceMessageId="synthetic-option-question",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", ["https://api.deepseek.com", "https://api.deepseek.com/v1"])
@pytest.mark.parametrize("value", [0.9, float("nan"), None, "synthetic-private-setting"])
async def test_unsupported_owned_setting_fails_before_semantic_work_or_completion_slot(
    rig, monkeypatch, caplog, base_url, value
):
    from services import turn_completion

    rig.manager.get_current_provider().base_url = base_url
    owned = SimpleNamespace(config={"topP": value, "useKnowledgeBase": True}, loras=[])
    reserve = Mock(side_effect=AssertionError("invalid configuration reserved a completion slot"))
    monkeypatch.setattr(turn_completion, "get_turn_completion_runtime", reserve)
    with pytest.raises(HTTPException) as caught:
        await rig.api._generate_reply_impl(
            message(), record_invocation=False, character_service=SimpleNamespace(), message_db=owned
        )
    assert caught.value.status_code == 503
    assert "topP" in caught.value.detail and "synthetic-private" not in caught.value.detail + caplog.text
    assert isinstance(caught.value.__cause__, (TypeError, ValueError))
    rig.prepare.assert_not_awaited()
    rig.generation.assert_not_awaited()
    rig.manager.get_current_provider().async_complete.assert_not_awaited()
    rig.manager.set_provider.assert_not_called()
    rig.save.assert_not_awaited()
    reserve.assert_not_called()
    assert owned.config["topP"] is value


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", [{}, {"topP": 1.0}, {"topP": "1.0"}])
async def test_missing_or_valid_cloud_setting_preserves_success_and_default(rig, settings):
    config = {**settings, "useKnowledgeBase": False}
    original = dict(config)
    owned = SimpleNamespace(config=config, loras=[])
    response = await rig.api._generate_reply_impl(
        message(character=False), persist_message=False, record_invocation=False, message_db=owned
    )
    assert response.reply == "根据完整当前输入回答。"
    rig.generation.assert_awaited_once()
    passed = rig.generation.await_args.kwargs["runtime_config"]
    assert float(passed["topP"]) == 1.0
    assert config == original
    rig.manager.set_provider.assert_not_called()


@pytest.mark.asyncio
async def test_other_compatible_service_keeps_its_supported_top_p(rig):
    rig.manager.get_current_provider().base_url = "https://other-provider.invalid/v1"
    owned = SimpleNamespace(config={"topP": 0.9, "useKnowledgeBase": False}, loras=[])
    response = await rig.api._generate_reply_impl(
        message(character=False), persist_message=False, record_invocation=False, message_db=owned
    )
    assert response.reply == "根据完整当前输入回答。"
    assert rig.generation.await_args.kwargs["runtime_config"]["topP"] == 0.9


@pytest.mark.asyncio
async def test_invalid_setting_cannot_be_hidden_by_a_cached_reply(rig, monkeypatch):
    cached = GenerateResponse(reply="合成历史回复", model="openai_compat/controlled-selected-model", costTime=0.0)
    cache = SimpleNamespace(get=AsyncMock(return_value=cached.model_dump()))
    monkeypatch.setattr(rig.api, "response_cache", cache)
    owned = SimpleNamespace(config={"topP": 0.9, "useKnowledgeBase": False}, loras=[])
    with pytest.raises(HTTPException) as caught:
        await rig.api._generate_reply_impl(
            message(character=False), persist_message=False, record_invocation=False, message_db=owned
        )
    assert caught.value.status_code == 503
    cache.get.assert_not_awaited()
    rig.generation.assert_not_awaited()
