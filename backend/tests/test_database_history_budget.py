"""Real SQLite and mocked-PG assembly must not manufacture orphan replies."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from db.database import SQLiteDB


async def read_history(tmp_path, monkeypatch, backend, turns, budget, limit=50, conversation_type='private'):
    if backend == 'sqlite':
        db = SQLiteDB(tmp_path / 'history.db')
        for message, reply in turns:
            db.add_message(dict(sessionId='room', sessionType=conversation_type, conversationId='room',
                conversationType=conversation_type, platform='web', adapter='audit',
                senderId='user', characterId='role', message=message, reply=reply,
                createdAt='2026-09-26T12:00:00'))
        return db.list_conversation_history('web', 'audit', 'user', conversation_type, 'room',
                                             limit=limit, max_chars=budget, character_id='role')
    pytest.importorskip('asyncpg')
    monkeypatch.setenv('DATABASE_URL', 'postgresql+asyncpg://test:test@localhost/test')
    from db.pg_database import PgDatabase
    rows = [SimpleNamespace(_mapping={'message': u, 'reply': a}) for u, a in reversed(turns)]
    session = MagicMock()
    async def execute(statement, params):
        return SimpleNamespace(fetchall=lambda: rows[:params['limit']])
    session.execute = AsyncMock(side_effect=execute)
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=session)
    manager.__aexit__ = AsyncMock(return_value=False)
    return await PgDatabase.list_conversation_history(SimpleNamespace(async_session=lambda: manager),
        'web', 'audit', 'user', conversation_type, 'room', limit=limit, max_chars=budget, character_id='role')


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['sqlite', 'postgres'])
@pytest.mark.parametrize('conversation_type', ['private', 'group'])
@pytest.mark.parametrize('limit', [8, 128, 1000])
async def test_cloud_history_limit_reaches_database(tmp_path, monkeypatch, backend, conversation_type, limit):
    turns = [(f'question-{i:04}', f'answer-{i:04}') for i in range(140)]
    result = await read_history(tmp_path, monkeypatch, backend, turns, 65536,
                                limit=limit, conversation_type=conversation_type)
    expected = turns[-min(limit, 500):]
    assert [item['content'] for item in result] == [value for pair in expected for value in pair]


@pytest.mark.parametrize('value, expected', [(-1, 1), (0, 1), (8, 8), (128, 128), (501, 500)])
def test_history_loading_has_bounded_explicit_limit(value, expected):
    from db.history_budget import history_turn_limit
    assert history_turn_limit(value) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['sqlite', 'postgres'])
@pytest.mark.parametrize('budget', [1, 6, 20, 106])
async def test_latest_oversize_turn_never_leaves_its_reply(tmp_path, monkeypatch, backend, budget):
    history = await read_history(tmp_path, monkeypatch, backend, [('前提' * 50, '回复')], budget)
    expected = [] if budget < 102 else [{'role': 'user', 'content': '前提' * 50}, {'role': 'assistant', 'content': '回复'}]
    assert history == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['sqlite', 'postgres'])
async def test_keep_contiguous_whole_turn_suffix(tmp_path, monkeypatch, backend):
    turns = [('更早', '旧'), ('长前提' * 30, '旧回复'), ('最新问题', '新回复')]
    assert await read_history(tmp_path, monkeypatch, backend, turns, 12) == [
        {'role': 'user', 'content': '最新问题'}, {'role': 'assistant', 'content': '新回复'}]


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['sqlite', 'postgres'])
@pytest.mark.parametrize('budget', [0, -1, 100])
async def test_single_sided_legacy_and_order_remain(tmp_path, monkeypatch, backend, budget):
    assert await read_history(tmp_path, monkeypatch, backend, [(' 用户 ', ''), ('', ' 助手 ')], budget) == [
        {'role': 'user', 'content': '用户'}, {'role': 'assistant', 'content': '助手'}]


def test_shared_assembly_respects_every_budget_and_never_mutates_source():
    from db.history_budget import assemble_history

    turns = [('最新' * 9, '答' * 7), ('中间' * 20, '答' * 3), ('更早', '旧回复')]
    original = list(turns)
    for budget in range(1, 120):
        result = assemble_history(iter(turns), budget)
        assert sum(len(item['content']) for item in result) <= budget
        assert len(result) % 2 == 0
        count = len(result) // 2
        assert [m['content'] for m in result] == [text for pair in reversed(turns[:count]) for text in pair]
    assert turns == original


@pytest.mark.asyncio
async def test_real_history_budget_has_no_orphan_and_does_not_delete_storage(tmp_path, monkeypatch):
    user = '请改写虚构故事。' + '背景' * 7990 + '不是我的经历。'
    result = await read_history(tmp_path, monkeypatch, 'sqlite', [(user, '你住在北京。')], 16000)
    assert result == []
    db = SQLiteDB(tmp_path / 'history.db')
    stored = db.list_conversation_history('web', 'audit', 'user', 'private', 'room',
                                           max_chars=0, character_id='role')
    assert stored == [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': '你住在北京。'}]
