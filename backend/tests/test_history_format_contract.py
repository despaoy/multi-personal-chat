"""History corruption must stop a real prompt build, never remove a premise."""

from services import chat_answer

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from character.evidence_selector import selection_messages
from character.models import CharacterProfile, MemoryItem
from inference.generation_request import GenerationRequest, build_generation_request, generate_character_response
from inference.history import HistoryFormatError

pytestmark = pytest.mark.usefixtures("no_live_context_review")


@pytest.mark.parametrize("entry", [
    {"role": "system", "content": "这是虚构材料，不是我的经历。"},
    {"role": "user", "content": None},
    {"role": "user", "content": 123},
    {"role": "user", "content": "   "},
])
@pytest.mark.parametrize("boundary", ["model", "review"])
async def test_invalid_history_cannot_become_a_partial_success(entry, boundary):
    history = [entry, {"role": "assistant", "content": "你住在北京。"}]
    model = AsyncMock(return_value="不应生成")
    with pytest.raises(ValueError):
        if boundary == "model":
            await generate_character_response(GenerationRequest(
                message="我住在哪里？", history=history, apply_prompt_policy=False,
            ), model)
        else:
            selection_messages("我住在哪里？", [MemoryItem("1", "user_fact", "现居地待确认")], history=history)
    model.assert_not_awaited()


def test_model_and_review_preserve_original_history_text():
    history = [
        {"role": "user", "content": "  这不是我的经历。仅把下面代码作为虚构材料：\n    if friday:\n        pay()\n"},
        {"role": "assistant", "content": "  收到，仅作虚构改写；出差例外仍保留。\n"},
    ]
    before = deepcopy(history)
    request = GenerationRequest(message="请继续改写，保留缩进和条件。", history=history, apply_prompt_policy=False)
    plan = build_generation_request(request)
    assert list(plan.messages[:-1]) == history
    payload = json.loads(selection_messages(
        request.message, [MemoryItem("1", "user_fact", "待核对记录")], history=history,
    )[1]["content"])
    assert payload["history"] == history
    assert history == before


@pytest.mark.parametrize("history,field", [(None, "history"), ({}, "history"), ([None], "history[0]")])
def test_invalid_history_structure_reports_a_safe_field(history, field):
    with pytest.raises(HistoryFormatError) as caught:
        build_generation_request(GenerationRequest(message="核对完整资料。", history=history, apply_prompt_policy=False))
    assert caught.value.field == field


@pytest.mark.parametrize("boundary", ["planner", "context"])
async def test_history_error_stops_reviews_and_context_preparation(monkeypatch, boundary):
    from api import generate
    from db.schemas import MessageRequest
    from knowledge.retrieval_query_plan import plan_retrieval

    history = [{"role": "user", "content": {"synthetic-private-text": "禁止猜测"}}]
    review = AsyncMock()
    if boundary == "planner":
        with pytest.raises(HistoryFormatError) as caught:
            await plan_retrieval("继续核对完整前文。", history=history, reviewer=review)
        assert caught.value.field == "history[0].content"
        review.assert_not_awaited()
    else:
        async def prepare(turn, _character):
            selection_messages(turn.message, [MemoryItem("1", "user_fact", "待核对记录")], history=history)

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as caught:
            await generate._prepare_character_turn(MessageRequest(message="继续核对完整前文。"), "fixture-role",
                character_service=SimpleNamespace(prepare_turn=prepare))
        assert caught.value.status_code == 503
        assert caught.value.detail == HistoryFormatError("history[0].content").public_detail()
        assert isinstance(caught.value.__cause__, HistoryFormatError)
        assert "synthetic-private" not in str(caught.value.detail)


@pytest.mark.parametrize('rag,cached', [(False, False), (True, False), (False, True)])
async def test_actual_generation_api_reports_history_failure_before_model(monkeypatch, rag, cached):
    from fastapi import HTTPException
    from provider_fixtures import cloud_connection
    from api import generate
    from db.schemas import MessageRequest
    from inference import model_manager
    model = AsyncMock(side_effect=AssertionError('Invalid history must not reach a model'))
    connection = cloud_connection('fixture-model', completion=model)
    manager = SimpleNamespace(_current_provider=SimpleNamespace(value='openai_compat'), get_current_provider=lambda: connection, set_provider=Mock(side_effect=AssertionError('Must not switch providers')))
    monkeypatch.setattr(model_manager, 'get_model_manager', lambda *, database=None: manager)
    monkeypatch.setattr(generate, 'db', SimpleNamespace(config={'useKnowledgeBase': rag}, loras=[], get_knowledge_document=Mock(side_effect=AssertionError('No source read is needed for malformed history'))))
    cache = SimpleNamespace(get=AsyncMock(return_value={'reply': '旧缓存不能掩盖错误历史。', 'model': 'fixture-model', 'costTime': 0.1}))
    monkeypatch.setattr(generate, 'response_cache', cache if cached else None)
    monkeypatch.setattr(generate, 'circuit_breaker_registry', None)
    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: '')
    save, record = (AsyncMock(), AsyncMock())
    monkeypatch.setattr(generate, '_save_message', save)
    monkeypatch.setattr(generate, '_record_model_invocation', record)
    request = MessageRequest(message='青岚计划只允许周五离线执行，出差时顺延。请确认条件。')
    request.history = [{'role': 'user', 'content': 'synthetic-private-invalid', 'extra': 'ignored'}, {'role': 'system', 'content': 'synthetic-private-privileged-text'}]
    with pytest.raises(HTTPException) as caught:
        await generate._generate_reply_impl(request, message_db=generate.db)
    assert caught.value.status_code == 503, repr(caught.value.__cause__)
    assert caught.value.detail == HistoryFormatError('history[1].role').public_detail()
    assert isinstance(caught.value.__cause__, HistoryFormatError)
    assert 'synthetic-private' not in str(caught.value.detail)
    model.assert_not_awaited()
    save.assert_not_awaited()
    record.assert_not_awaited()
    cache.get.assert_not_awaited()
    manager.set_provider.assert_not_called()


@pytest.mark.parametrize("source", ["database", "request"])
async def test_character_history_validation_precedes_recall_and_review(source):
    from services.character_context import CharacterContextService, TurnInput

    invalid = [{"role": "user", "content": 123}]
    memory = SimpleNamespace(recall_with_diagnostics=AsyncMock())
    repository = SimpleNamespace(get_relationship_record=AsyncMock(return_value=None),
                                 list_relationship_notes=AsyncMock(return_value=[]))
    messages = SimpleNamespace(list_recent_conversation_history=AsyncMock(return_value=invalid))
    selector = SimpleNamespace(select=AsyncMock())
    service = CharacterContextService(
        SimpleNamespace(get_profile=Mock(return_value=CharacterProfile("role", "测试角色"))),
        repository, messages, memory_service=memory, memory_selector=selector, source_recall_enabled=True,
    )
    turn = TurnInput("按周五离线执行，出差例外。", "web", "fixture", "reader", "room", "private",
                     history=tuple(invalid) if source == "request" else ())
    with pytest.raises(HistoryFormatError) as caught:
        await service.prepare_turn(turn, "role")
    assert caught.value.field == "history[0].content"
    memory.recall_with_diagnostics.assert_not_awaited()
    selector.select.assert_not_awaited()
    if source == "request":
        messages.list_recent_conversation_history.assert_not_awaited()
