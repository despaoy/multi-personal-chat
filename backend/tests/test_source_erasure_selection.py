import json
from datetime import datetime, timezone

import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import UserScope
from character.source_erasure_selection import candidates, selected_ids
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

SCOPE = UserScope('web', 'test', 'user', 'user', 'private')


@pytest.mark.parametrize('payload,authorized', [({'erase_source_ids': ['foreign']}, True),
    ({'erase_source_ids': ['known']}, False), ({'erase_source_ids': 'known'}, True)])
def test_invalid_source_targets_never_authorized(payload, authorized):
    with pytest.raises(ValueError):
        selected_ids(payload, {'known'}, authorized=authorized)


async def test_source_only_target_uses_same_writer_call_and_committed_receipt(tmp_path):
    db = SQLiteDB(tmp_path / 'source.sqlite')
    repo = DatabaseCharacterMemoryRepository(db)
    for source, body in [('pet', '我的狗叫松糕。'), ('other', '我周末喜欢参观美术馆。')]:
        await repo.capture_source('role', SCOPE, source_message_id=source, body=body,
                                  observed_at=datetime.now(timezone.utc))

    class Completion:
        calls = 0

        async def complete(self, messages):
            self.calls += 1
            data = json.loads(messages[1]['content'])
            assert data['existing_memories'] == []
            assert any(r['source_id'] == 'pet' and r['text'] == '我的狗叫松糕。'
                       for r in data['source_erasure_candidates'])
            return '{"memories":[],"erase_source_ids":["pet"]}'

        async def close(self):
            pass

    completion = Completion()
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(enabled=True, base_url='http://unused',
        model='test', context_window_tokens=65536), completion=completion)
    try:
        receipt = await scheduler.schedule_and_wait(repository=repo, character_id='role', user_scope=SCOPE,
            message='请把我的狗的名字从记忆里删除。', rule_hints=(), history=(), source_message_id='delete')
        assert receipt['status'] == 'erased' and receipt['source_erased'] == 1
        from character.memory_operation import render_operation_response

        reply = render_operation_response('请把我的狗的名字从记忆里删除。', receipt)
        assert '已删除本次匹配的记录' in reply and '未逐一核对全部历史' in reply
        assert completion.calls == 1
        assert [r['source_message_id'] for r in await repo.list_sources('role', SCOPE)] == ['other']
        assert await repo.list_memory_records('role', SCOPE) == []
    finally:
        await scheduler.shutdown(timeout=2)


async def test_candidate_budget_keeps_full_sources_and_reports_omissions():
    class Repo:
        async def search_sources(self, *args, **kwargs):
            return [dict(source_message_id='long', body='甲' * 5000)]

        async def list_sources(self, *args, **kwargs):
            return [dict(source_message_id='short', body='完整短原话')]

    rows, coverage = await candidates(Repo(), 'role', SCOPE, '删除', (), context_window_tokens=8192)
    assert rows == (dict(source_id='short', text='完整短原话', observed_at=None),)
    assert coverage['omitted'] == 1 and coverage['complete'] is False
