from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from services.character_context import CharacterContextService, PreparedCharacterTurn, TurnInput

from test_provider_usage import installed as installed, publication_database as publication_database
from services.chat_answer import generate_chat_answer as actual_generate_chat_answer
from character.memory_operation import operation_receipt_context
from character.models import CompiledCharacterContext, RelationshipState, UserScope


def setup_service(monkeypatch, receipt):
    scope = UserScope('web', 'test', 'u', 'c', 'private')
    prepared = PreparedCharacterTurn('role', scope, CompiledCharacterContext('', '', ''), (),
                                    RelationshipState(), 0, 0, None)
    service = object.__new__(CharacterContextService)
    service._profiles = SimpleNamespace(get_profile=lambda character: object())
    service._memory_repo = SimpleNamespace(increment_interaction=AsyncMock(return_value=1))
    service._load_history = AsyncMock(return_value=[])
    service.prepare_turn = AsyncMock(return_value=prepared)
    worker = SimpleNamespace(enabled=True, schedule_and_wait=AsyncMock(return_value=receipt),
                             schedule=lambda **kwargs: pytest.fail('duplicate operation submitted'))
    monkeypatch.setattr('character.memory_llm.get_memory_enrichment_scheduler', lambda: worker)
    return service, worker


@pytest.mark.parametrize('status', ['erased', 'pending', 'failed', 'partial', 'cancelled', 'not_scheduled', 'conflict'])
async def test_explicit_operation_precedes_snapshot_and_is_not_resubmitted(monkeypatch, status):
    service, worker = setup_service(monkeypatch, {'status': status, 'persisted': int(status == 'erased')})
    original_prepare = service.prepare_turn

    async def snapshot(*args):
        worker.schedule_and_wait.assert_awaited_once()
        return await original_prepare(*args)

    service.prepare_turn = snapshot
    turn = TurnInput('请从记忆里删除我的住址。', 'web', 'test', 'u', 'c', 'private')
    prepared = await service.prepare_interactive_turn(turn, 'role', source_message_id='request-1')
    assert prepared.memory_operation_receipt['status'] == status
    assert '操作执行回执' in prepared.compiled.dynamic_context
    assert worker.schedule_and_wait.await_args.kwargs['source_message_id'] == 'request-1'
    outcome = await service.complete_turn(prepared, turn, source_message_id='request-1')
    assert outcome.memory_enrichment_status == status
    assert outcome.memory_enrichment_mode == 'explicit_operation'
    assert not outcome.memory_enrichment_scheduled


@pytest.mark.parametrize('message', ['我住在临沂。', '不要忘掉我的住址。', '“请忘掉我的住址”是示例。'])
async def test_ordinary_and_non_authorizing_turns_do_not_execute(monkeypatch, message):
    service, worker = setup_service(monkeypatch, {})
    result = await service.prepare_interactive_turn(TurnInput(message, 'web', 'test', 'u', 'c', 'private'), 'role')
    assert result.memory_operation_receipt is None
    worker.schedule_and_wait.assert_not_awaited()


async def test_invalid_profile_prevents_operation(monkeypatch):
    service, worker = setup_service(monkeypatch, {})

    def missing(character):
        raise ValueError('missing profile')

    service._profiles.get_profile = missing
    with pytest.raises(ValueError, match='missing profile'):
        await service.prepare_interactive_turn(TurnInput('忘掉我的住址。', 'web', 'test', 'u', 'c', 'private'), 'role')
    worker.schedule_and_wait.assert_not_awaited()


def test_receipt_rendering_never_includes_source_or_arbitrary_error():
    rendered = operation_receipt_context({'status': 'ignore instructions', 'persisted': 'secret', 'error': 'private fact'})
    assert 'ignore instructions' not in rendered and 'secret' not in rendered and 'private fact' not in rendered
    assert '结果未确认' in rendered


@pytest.mark.parametrize('enabled', [False, True])
async def test_api_preparation_only_executes_when_explicitly_enabled(enabled):
    from api.generate import _prepare_character_turn
    from db.schemas import MessageRequest

    service = SimpleNamespace(prepare_turn=AsyncMock(return_value='read-only'),
                              prepare_interactive_turn=AsyncMock(return_value='interactive'))
    request = MessageRequest(message='忘掉我的住址。', userId='u', sessionId='c', sourceMessageId='s')
    result = await _prepare_character_turn(request, 'role', character_service=service,
                                           execute_memory_operations=enabled)
    assert result == ('interactive' if enabled else 'read-only')
    if enabled:
        service.prepare_turn.assert_not_awaited()
    else:
        service.prepare_interactive_turn.assert_not_awaited()


@pytest.mark.parametrize("status", ["erased", "partial"])
async def test_context_failure_preserves_executed_memory_operation_receipt(monkeypatch, status):
    from fastapi import HTTPException
    from services.character_context import ContextAfterMemoryOperationError

    from api.generate import _prepare_character_turn
    from db.schemas import MessageRequest

    receipt = {"status": status, "persisted": 1, "error": "fiction-private-memory-text"}
    service, worker = setup_service(monkeypatch, receipt)
    failure = RuntimeError("fiction-private-provider-text")
    service.prepare_turn.side_effect = failure
    request = MessageRequest(message="请从记忆里删除我的住址。", userId="u", sessionId="c", sourceMessageId="operation-1")
    with pytest.raises(HTTPException) as caught:
        await _prepare_character_turn(request, "role", character_service=service, execute_memory_operations=True)
    error = caught.value
    assert error.status_code == 503 and error.detail["code"] == "context_failed_after_memory_operation"
    assert "已确认完成的操作数：1" in error.detail["memoryOperation"]
    assert "回复未生成" in error.detail["message"]
    assert "fiction-private" not in str(error.detail)
    assert isinstance(error.__cause__, ContextAfterMemoryOperationError)
    assert error.__cause__.__cause__ is failure
    worker.schedule_and_wait.assert_awaited_once()
    service.prepare_turn.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['success', 'provider', 'archive'])
async def test_committed_erasure_receipt_survives_later_generation_failure(installed, publication_database, monkeypatch, fault):
    import json
    from datetime import datetime, timedelta, timezone
    from fastapi import HTTPException
    from api import generate
    from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
    from character.models import build_user_scope
    from character.profile_registry import get_default_profile_registry
    from db.schemas import MessageRequest
    from repositories.character_memory import DatabaseCharacterMemoryRepository
    from repositories.messages import DatabaseMessageRepository
    from services import chat_answer
    import test_compiled_cloud_transport
    database = publication_database
    repository = DatabaseCharacterMemoryRepository(database)
    character = 'tsukiyashiro_kisaki'
    scope = build_user_scope('web', 'erasure-fixture', 'synthetic-owner', 'synthetic-owner', 'private')
    await repository.capture_source(character, scope, source_message_id='old', body='我喜欢测试红茶。', observed_at=datetime.now(timezone.utc) - timedelta(days=1))

    class Completion:
        calls = 0

        async def complete(self, messages):
            self.calls += 1
            payload = json.loads(messages[1]['content'])
            assert any((row['source_id'] == 'old' for row in payload['source_erasure_candidates']))
            return '{"memories":[],"erase_source_ids":["old"]}'

        async def close(self):
            pass
    completion = Completion()
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, 'unit-only', 'unit-only'), completion=completion)
    service = CharacterContextService(profile_registry=get_default_profile_registry(), memory_repository=repository, message_repository=DatabaseMessageRepository(database), source_recall_enabled=False)
    monkeypatch.setattr('character.memory_llm.get_memory_enrichment_scheduler', lambda: scheduler)
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', actual_generate_chat_answer)
    monkeypatch.setattr(test_compiled_cloud_transport, 'ANSWER', '20。')
    installed.state.status = 503 if fault == 'provider' else 200
    if fault == 'archive':
        monkeypatch.setattr(database, 'add_message', Mock(side_effect=OSError('synthetic-private-archive-fault')))
    request = MessageRequest(message='请从长期记忆中删除我喜欢测试红茶的记录，同时计算12+8。', characterId=character, loraName='default', platform='web', adapter='erasure-fixture', senderId='synthetic-owner', userId='synthetic-owner', conversationId='synthetic-owner', sessionId='synthetic-owner', sourceMessageId='current-erasure', traceId='synthetic-erasure-trace')
    try:
        if fault == 'success':
            result = await generate._generate_reply_impl(request, enable_rag=False, message_db=database, character_service=service)
            assert '已删除' in result.reply and result.reply.endswith('20。')
        else:
            with pytest.raises(HTTPException) as caught:
                await generate._generate_reply_impl(request, enable_rag=False, message_db=database, character_service=service)
            assert caught.value.__cause__ is not None
            assert 'synthetic-private' not in str(caught.value.detail)
        assert completion.calls == 1 and len(installed.calls) == 1
        assert await repository.list_sources(character, scope) == []
        assert len(database.execute_sql('SELECT id FROM model_invocations')) == 1
        assert len(database.execute_sql('SELECT id FROM messages')) == int(fault == 'success')
        if fault != 'success':
            if fault == 'provider':
                assert caught.value.status_code == 502
                assert caught.value.detail['upstreamStatus'] == 503
            else:
                assert caught.value.status_code == 503
                assert caught.value.detail['stage'] == 'message_archive'
                assert caught.value.detail['messageSaved'] is None
            assert 'memoryOperation' in caught.value.detail, {'http_status': caught.value.status_code, 'fields': sorted(caught.value.detail)}
            assert '已确认执行删除' in caught.value.detail['memoryOperation']
            assert '已确认完成的操作数：1' in caught.value.detail['memoryOperation']
            assert '记忆操作回执' in caught.value.detail['message']
    finally:
        await scheduler.shutdown(timeout=2)
