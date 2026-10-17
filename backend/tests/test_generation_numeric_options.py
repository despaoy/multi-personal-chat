"""Invalid stored generation settings must fail before semantic work or cache."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from db.schemas import GenerateResponse, MessageRequest
from inference.generation_options import read_generation_options
from test_lora_provider_selection import install_api
from test_generation_boolean_config import control as control

pytestmark = pytest.mark.usefixtures("no_live_context_review")
ROLE = "tsukiyashiro_kisaki"


def request(*, character=True):
    return MessageRequest(
        message="三张票，每张两元，请计算总额并说明计算过程。",
        characterId=ROLE if character else None,
        loraName="default",
        platform="web",
        adapter="web-character",
        senderId="synthetic-reader",
        userId="synthetic-reader",
        sessionId="synthetic-parameters",
        sourceMessageId="synthetic-parameter-question",
    )


@pytest.fixture
def rig(monkeypatch):
    state = install_api(monkeypatch, "openai_compat")
    state.manager.get_current_provider().base_url = "https://api.deepseek.com"
    state.prepare = AsyncMock(side_effect=AssertionError("Invalid settings reached semantic preparation"))
    monkeypatch.setattr(state.api, "_prepare_character_turn", state.prepare)
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", "synthetic-private-invalid"),
        ("temperature", -0.1),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("temperature", None),
        ("temperature", True),
        ("maxTokens", "synthetic-private-invalid"),
        ("maxTokens", 64.5),
        ("maxTokens", 0),
        ("maxTokens", -1),
        ("maxTokens", None),
        ("maxTokens", True),
        ("topP", 0),
        ("topP", 1.1),
        ("topP", float("nan")),
        ("topP", False),
    ],
)
async def test_invalid_owned_number_stops_before_semantic_work_or_slot(rig, monkeypatch, caplog, field, value):
    from services import turn_completion

    owned = SimpleNamespace(config={"topP": 1.0, "useKnowledgeBase": True, field: value})
    monkeypatch.setattr(rig.api, "db", SimpleNamespace(config={"temperature": 0.7, "maxTokens": 2048, "topP": 1.0}))
    reserve = Mock(side_effect=AssertionError("Invalid settings reserved a completion slot"))
    monkeypatch.setattr(turn_completion, "get_turn_completion_runtime", reserve)
    with pytest.raises(HTTPException) as caught:
        await rig.api._generate_reply_impl(
            request(), message_db=owned, character_service=SimpleNamespace(), record_invocation=False
        )
    assert caught.value.status_code == 503 and isinstance(caught.value.__cause__, ValidationError)
    assert field in caught.value.detail
    assert "synthetic-private" not in caught.value.detail + caplog.text + str(caught.value.__cause__)
    rig.prepare.assert_not_awaited()
    rig.generation.assert_not_awaited()
    rig.manager.get_current_provider().async_complete.assert_not_awaited()
    rig.manager.set_provider.assert_not_called()
    rig.save.assert_not_awaited()
    reserve.assert_not_called()
    assert owned.config[field] is value


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("temperature", -1), ("maxTokens", 0), ("topP", 0)])
async def test_invalid_number_cannot_use_an_old_cached_answer(rig, monkeypatch, field, value):
    cached = GenerateResponse(reply="这是旧缓存回答。", model="synthetic", costTime=0).model_dump()
    cache = SimpleNamespace(get=AsyncMock(return_value=cached), set=AsyncMock())
    monkeypatch.setattr(rig.api, "response_cache", cache)
    owned = SimpleNamespace(config={"topP": 1.0, "useKnowledgeBase": False, field: value})
    with pytest.raises(HTTPException) as caught:
        await rig.api._generate_reply_impl(
            request(character=False), message_db=owned, persist_message=False, record_invocation=False
        )
    assert caught.value.status_code == 503
    cache.get.assert_not_awaited()
    cache.set.assert_not_awaited()
    rig.generation.assert_not_awaited()


@pytest.mark.asyncio
async def test_valid_owned_values_override_environment_without_mutating_storage(rig, monkeypatch):
    monkeypatch.setenv("VLLM_TEMPERATURE", "synthetic-invalid-environment")
    monkeypatch.setenv("VLLM_MAX_TOKENS", "synthetic-invalid-environment")
    monkeypatch.setenv("VLLM_TOP_P", "synthetic-invalid-environment")
    original = {"temperature": "0", "maxTokens": "1536", "topP": "1.0", "useKnowledgeBase": False}
    owned = SimpleNamespace(config=dict(original))
    response = await rig.api._generate_reply_impl(
        request(character=False), message_db=owned, persist_message=False, record_invocation=False
    )
    assert response.reply == "根据完整当前输入回答。"
    passed = rig.generation.await_args.kwargs["runtime_config"]
    assert {key: passed[key] for key in ["temperature", "maxTokens", "topP"]} == {
        "temperature": 0.0,
        "maxTokens": 1536,
        "topP": 1.0,
    }
    assert owned.config == original
    rig.generation.assert_awaited_once()


def test_one_numeric_contract_keeps_environment_defaults_and_cache_identity(rig, monkeypatch):
    monkeypatch.setenv("VLLM_TEMPERATURE", "0.4")
    monkeypatch.setenv("VLLM_MAX_TOKENS", "3072")
    monkeypatch.setenv("VLLM_TOP_P", "0.8")
    resolved = read_generation_options({})
    assert resolved == {"temperature": 0.4, "maxTokens": 3072, "topP": 0.8}
    arguments = dict(model_identity={"provider": "synthetic"}, enable_rag=False)
    assert rig.api._response_cache_keys(request(character=False), "default", {}, **arguments) == (
        rig.api._response_cache_keys(request(character=False), "default", resolved, **arguments)
    )


@pytest.mark.asyncio
async def test_cloud_absent_top_p_keeps_one_while_other_missing_values_use_environment(rig, monkeypatch):
    monkeypatch.setenv("VLLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("VLLM_MAX_TOKENS", "1024")
    monkeypatch.setenv("VLLM_TOP_P", "0.8")
    owned = SimpleNamespace(config={"useKnowledgeBase": False})
    await rig.api._generate_reply_impl(
        request(character=False), message_db=owned, persist_message=False, record_invocation=False
    )
    passed = rig.generation.await_args.kwargs["runtime_config"]
    assert {key: passed[key] for key in ["temperature", "maxTokens", "topP"]} == {
        "temperature": 0.2,
        "maxTokens": 1024,
        "topP": 1.0,
    }
    assert owned.config == {"useKnowledgeBase": False}


@pytest.mark.asyncio
async def test_shared_options_reach_the_actual_generation_request(control):
    from api import generate

    reply, used, _ = await generate._generate_with_retrieval(
        control.request,
        None,
        runtime_config={"temperature": "0.25", "maxTokens": "384", "topP": "0.9", "useKnowledgeBase": False},
        model_generate=control.model,
    )
    assert reply == "你好，很高兴见到你。" and not used
    control.model.assert_awaited_once()
    passed = control.model.await_args.kwargs
    assert passed["temperature"] == 0.25 and passed["max_tokens"] == 384 and passed["top_p"] == 0.9
