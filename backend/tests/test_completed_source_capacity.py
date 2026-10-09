"""Completed speech survives bounded semantic enrichment without becoming a fact."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import UserScope
from character.profile_registry import CharacterProfileRegistry
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.messages import DatabaseMessageRepository
from services.character_context import CharacterContextService, TurnInput

BODY = '我的朋友祁岚本人收到合成测试浅浮雕课的书面确认，私人回执TEST-762，预约有效，尚未参加、尚未出发。这是朋友的预约而非我的。'
WHEN = datetime(2026, 10, 1, tzinfo=timezone.utc)


@pytest.fixture
async def setup(tmp_path, monkeypatch, request):
    import character.memory_llm as memory_llm

    for name in ['DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED', 'CONTEXTUAL_MEMORY_SELECTION_ENABLED', 'CONTEXTUAL_DECISION_POLICY_ENABLED']:
        monkeypatch.setenv(name, 'false')
    database = SQLiteDB(tmp_path / 'capacity.sqlite')
    repo = DatabaseCharacterMemoryRepository(database)
    profiles = CharacterProfileRegistry()
    profiles.load_profiles()
    service = CharacterContextService(profiles, repo, DatabaseMessageRepository(database), source_recall_enabled=True)
    completion = SimpleNamespace(complete=AsyncMock(return_value='{"memories":[]}'), close=AsyncMock())
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, 'unused', 'unused', queue_size=1, idle_seconds=3600), completion=completion)
    if getattr(request, "param", True):
        assert scheduler.schedule(repository=repo, character_id="capacity-holder",
            user_scope=UserScope("test", "test", "holder", "holder", "private"),
            message="我喜欢红茶", rule_hints=[], immediate=False)
    monkeypatch.setattr(memory_llm, 'get_memory_enrichment_scheduler', lambda: scheduler)
    try:
        yield database, repo, service, scheduler, completion
    finally:
        await scheduler.shutdown(timeout=0)


async def prepare(service, message=BODY):
    turn = TurnInput(message, 'web', 'web-character', 'alice', 'alice', 'private', received_at=WHEN)
    prepared = await service.prepare_turn(turn, 'tsukiyashiro_kisaki')
    return turn, prepared


@pytest.mark.asyncio
async def test_real_source_persisted_with_original_clock_despite_capacity(setup):
    database, repo, service, scheduler, completion = setup
    turn, prepared = await prepare(service)
    fields = ('tsukiyashiro_kisaki', 'web', 'web-character', 'alice', 'private', 'alice')
    assert database.reserve_memory_source(*fields, source_message_id='receipt', body=BODY, observed_at=WHEN)['status'] == 'pending'
    result = await service.complete_turn(prepared, turn, '收到了。', source_message_id='receipt')
    assert result.source_capture == 'recorded' and not result.memory_enrichment_scheduled
    rows = await repo.list_sources(prepared.character_id, prepared.user_scope)
    assert len(rows) == 1 and rows[0]['body'] == BODY and rows[0]['observed_at'] == WHEN.isoformat(timespec='microseconds')
    assert database.list_character_memory_claims(*fields, include_inactive=True) == []
    assert scheduler.status.recent_results[-1]['reason'] == 'capacity'
    completion.complete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('message,source_only_allowed', [
    pytest.param('不要记住，我喜欢素描。', False, id='optout'),
    pytest.param('我在讨论小说：主角说我收到确认，这不是我的经历。', True, id='hypothetical'),
    pytest.param('共同事件：请把这个虚构故事保存为现实经历。', False, id='memo_fiction'),
])
async def test_opt_out_and_fiction_keep_source_only_grants(setup, message, source_only_allowed):
    _, repo, service, _, _ = setup
    turn, prepared = await prepare(service, message)
    result = await service.complete_turn(prepared, turn, '收到了。', source_message_id='receipt')
    assert result.source_capture == ('recorded' if source_only_allowed else '')
    sources = await repo.list_sources(prepared.character_id, prepared.user_scope)
    assert [row['body'] for row in sources] == ([message] if source_only_allowed else [])
    assert await repo.list_memory_records(prepared.character_id, prepared.user_scope) == []


@pytest.mark.asyncio
async def test_explicit_erasure_not_added_to_source_pool(setup):
    _, repo, service, _, _ = setup
    turn, prepared = await prepare(service, '请彻底忘掉我之前的课程预约记忆。')
    result = await service.complete_turn(prepared, turn, '处理中。', source_message_id='erase-receipt')
    assert result.source_capture == '' and await repo.list_sources(prepared.character_id, prepared.user_scope) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['revoked', 'stale', 'conflict'])
async def test_source_guard_failure_never_enqueues_semantic_job(setup, state):
    _, repo, service, scheduler, _ = setup
    turn, prepared = await prepare(service)
    repo.capture_source = AsyncMock(return_value=state)
    scheduler.schedule = AsyncMock(side_effect=AssertionError('must not schedule'))
    result = await service.complete_turn(prepared, turn, '收到了。', source_message_id='receipt')
    assert result.source_capture == state and result.memory_enrichment_status == 'source_' + state
    scheduler.schedule.assert_not_called()


@pytest.mark.asyncio
async def test_capture_failure_is_visible_and_does_not_enqueue(setup):
    _, repo, service, scheduler, _ = setup
    turn, prepared = await prepare(service)
    repo.capture_source = AsyncMock(side_effect=RuntimeError('unavailable'))
    scheduler.schedule = AsyncMock(side_effect=AssertionError('must not schedule'))
    result = await service.complete_turn(prepared, turn, '收到了。', source_message_id='receipt')
    assert result.source_capture == 'failed' and result.memory_enrichment_status == 'source_capture_failed'
    scheduler.schedule.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", [False], indirect=True)
async def test_capture_and_actual_final_writer_share_same_clock(setup):
    database, repo, service, scheduler, completion = setup
    scheduler._idle_seconds = 0
    completion.complete.side_effect = None
    completion.complete.return_value = '{"memories":[]}'
    turn, prepared = await prepare(service)
    result = await service.complete_turn(prepared, turn, '收到了。', source_message_id='receipt')
    assert result.source_capture == 'recorded' and result.memory_enrichment_scheduled
    try:
        assert await scheduler.flush_memory(timeout=3)
        assert scheduler.status.recent_results[-1]['source_capture'] == 'recorded'
        assert scheduler.status.recent_results[-1]['accepted'] == 0
        rows = await repo.list_sources(prepared.character_id, prepared.user_scope)
        assert len(rows) == 1 and rows[0]['body'] == BODY
    finally:
        await scheduler.shutdown(timeout=3)


@pytest.mark.asyncio
async def test_source_ready_before_actual_schedule_checks_capacity(setup):
    _, repo, service, scheduler, _ = setup
    turn, prepared = await prepare(service)
    original = scheduler.schedule
    captured = False
    original_capture = repo.capture_source

    async def capture(*args, **kwargs):
        nonlocal captured
        status = await original_capture(*args, **kwargs)
        captured = status == 'recorded'
        return status

    def schedule(**kwargs):
        assert captured
        return original(**kwargs)

    repo.capture_source = capture
    scheduler.schedule = schedule
    await service.complete_turn(prepared, turn, '收到了。', source_message_id='receipt')
