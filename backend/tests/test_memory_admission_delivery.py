"""Refused semantic work stays pending; accepted work owns its exact receipt."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services import delivery_memory as dm
from services.character_context import CharacterContextService
from services.turn_completion import TurnCompletionRuntime
from test_delivery_memory import deliver, generated

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import UserScope
from db.database import SQLiteDB
from db.schemas import MessageRequest


@pytest.mark.parametrize('rejection', ['capacity', 'closed'])
async def test_refused_memory_is_pending_then_recovers_without_repeating_turn(tmp_path, monkeypatch, rejection):
    database = SQLiteDB(tmp_path / 'delivery.sqlite')
    repo = DatabaseCharacterMemoryRepository(database)
    key, owner, stored = generated(database)
    deliver(database, key, owner, stored)
    context = stored['context']
    snapshot, _ = dm.thaw_completion(context)
    body = context['request']['message']
    proposal = json.dumps({'memories': [dict(attributed_to='user', operation='ADD', kind='like',
        value='深蓝色油墨', evidence=body, confidence=.99)]}, ensure_ascii=False)
    completion = SimpleNamespace(complete=AsyncMock(return_value=proposal), close=AsyncMock())

    def scheduler():
        return MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, 'unused', 'fixture',
            queue_size=1, idle_seconds=3600), completion=completion)

    current = {'scheduler': scheduler()}
    schedulers = [current['scheduler']]
    monkeypatch.setattr('character.memory_llm.get_memory_enrichment_scheduler', lambda: current['scheduler'])
    runtime = TurnCompletionRuntime(1)
    monkeypatch.setattr(dm, 'get_turn_completion_runtime', lambda: runtime)
    monkeypatch.setattr(dm, 'POLL_SECONDS', 3600)
    slot = runtime.reserve()  # Keep the automatic loop parked during explicit job steps.
    worker = dm.DeliveryMemoryWorker(database, CharacterContextService(object(), repo, object()))
    try:
        if rejection == 'capacity':
            assert current['scheduler'].schedule(repository=repo, character_id=snapshot.character_id,
                user_scope=UserScope('web', 'test', 'other', 'other', 'private'), message=body,
                rule_hints=[], source_message_id='occupying', immediate=False)
        else:
            await current['scheduler'].shutdown(timeout=1)
        await worker._process(database.integration_receipt('get', key=key), runtime, slot)
        row = database.integration_receipt('get', key=key)
        response = json.loads(row['response'])
        marker = response['context']['memory_completion']
        assert row['status'] == 'delivered' and response['memory_completion_state'] == 'pending'
        assert marker['turn_finished'] and marker['semantic_receipt']['status'] == rejection
        assert marker['attempts'] == 0 and marker['execution_started'] is False
        assert current['scheduler'].status.last_outcome == rejection
        assert current['scheduler'].status.recent_results[-1] == {'status': rejection, 'reason': rejection}
        completion.complete.assert_not_awaited()
        assert not await repo.list_memory_records(snapshot.character_id, snapshot.user_scope)
        sources = await repo.list_sources(snapshot.character_id, snapshot.user_scope)
        assert sources == [dict(source_message_id=context['request']['sourceMessageId'], body=body,
                                observed_at=snapshot.received_at.isoformat(timespec='microseconds'))]
        assert (await repo.get_relationship_record(snapshot.character_id, snapshot.user_scope))['interaction_count'] == 1
        assert any(item['receipt_key'] == key for item in database.integration_receipt('memory_pending', now=float('inf'), limit=10))

        if rejection == 'capacity':
            assert await current['scheduler'].flush_memory(timeout=2)
        else:
            current['scheduler'] = scheduler()
            schedulers.append(current['scheduler'])
        current['scheduler']._idle_seconds = 0
        # Make the existing five-second admission backoff eligible without sleeping.
        assert database.integration_receipt('memory_update', key=key, owner=owner,
            expected=row['response'], response=row['response'], expires_at=0)
        row = database.integration_receipt('get', key=key)
        retry_slot = runtime.reserve()
        try:
            await worker._process(row, runtime, retry_slot)
        finally:
            retry_slot.release()
        final = json.loads(database.integration_receipt('get', key=key)['response'])
        marker = final['context']['memory_completion']
        assert final['memory_completion_state'] == 'completed'
        assert marker['attempts'] == 1
        assert marker['semantic_receipt']['status'] == 'saved' and marker['semantic_receipt']['persisted'] == 1
        records = await repo.list_memory_records(snapshot.character_id, snapshot.user_scope)
        assert len(records) == 1 and '深蓝色油墨' in records[0]['content']
        assert records[0]['source_message_id'] == context['request']['sourceMessageId']
        assert await repo.list_sources(snapshot.character_id, snapshot.user_scope) == sources
        assert (await repo.get_relationship_record(snapshot.character_id, snapshot.user_scope))['interaction_count'] == 1
        assert final['reply'] == stored['reply']
        assert completion.complete.await_count == (2 if rejection == 'capacity' else 1)
        assert not any(item['receipt_key'] == key for item in database.integration_receipt('memory_pending', now=float('inf'), limit=10))
    finally:
        slot.release()
        await worker.shutdown(timeout=1)
        await runtime.shutdown(timeout=1)
        for item in schedulers:
            if not item.status.closed:
                await item.shutdown(timeout=2)
        database.close_connection()


@pytest.mark.parametrize('status', ['capacity', 'closed'])
@pytest.mark.parametrize('capture', ['recorded', ''])
async def test_chat_feedback_reports_unaccepted_memory_without_claiming_raw_storage(status, capture):
    from api import generate

    service = SimpleNamespace(complete_turn=AsyncMock(return_value=SimpleNamespace(
        memory_enrichment_status=status, source_capture=capture)))
    warning = await generate._complete_character_turn(SimpleNamespace(character_id='tsukiyashiro_kisaki'),
        MessageRequest(message='我本人长期喜欢深蓝色油墨，这是明确的个人偏好。', sourceMessageId='synthetic-source'),
        '合成回复', character_service=service)
    assert '尚未受理' in warning and '不代表结构化记忆已写入' in warning
    assert ('原文已保存' in warning) is (capture == 'recorded')
