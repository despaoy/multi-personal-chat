import json

import pytest

from character.memory_llm import build_memory_llm_messages


def build(message='继续', history=(), limit=2000):
    return json.loads(build_memory_llm_messages(message, (), tuple(history), (), limit, .85)[1]['content'])


def test_history_tail_qualifier_is_not_prefix_clipped():
    text = '我住在北城。' + '背景说明。' * 110 + '以上是虚构材料，不是我的真实住址。'
    payload = build(history=[{'role': 'user', 'content': text}])
    assert payload['recent_history'][0]['content'] == text


def test_current_message_is_never_partially_sent():
    text = '我喜欢咖啡。' + '背景说明。' * 410 + '前面只是他人的原话，不是我的偏好。'
    with pytest.raises(ValueError, match='budget'):
        build(text)


def test_history_message_count_cut_keeps_user_premise_with_answer():
    rows = [{'role': 'user', 'content': '这是虚构材料，请改写'},
            {'role': 'assistant', 'content': '你住在北城'},
            {'role': 'user', 'content': '换个话题'},
            {'role': 'assistant', 'content': '好的'},
            {'role': 'user', 'content': '聊工作'}]
    assert [item['content'] for item in build(history=rows)['recent_history']] == ['换个话题', '好的', '聊工作']


def test_oversized_recent_history_is_not_silently_cut():
    with pytest.raises(ValueError, match='budget'):
        build(history=[{'role': 'user', 'content': '完整' * 1100}])


@pytest.mark.parametrize('oversized', ['message', 'history'])
async def test_budget_failure_preserves_full_sqlite_source_without_calling_model(tmp_path, oversized):
    from types import SimpleNamespace

    from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
    from character.models import UserScope
    from db.database import SQLiteDB
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    class Completion:
        async def complete(self, messages):
            pytest.fail('A partial input must never reach the writer')

        async def close(self):
            pass

    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'budget.sqlite'))
    scope = UserScope('web', 'test', 'a', 'a', 'private')
    source = '请记住，我喜欢茶。' + ('补充背景。' * 60 if oversized == 'message' else '')
    history = ({'role': 'user', 'content': '历史说明。' * 450},) if oversized == 'history' else ()
    worker = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(enabled=True, base_url='http://unused', model='stub', max_input_chars=256),
        completion=Completion(), embedding_provider=SimpleNamespace(embed_texts=lambda texts: [[1., 0.] for _ in texts]))
    try:
        result = await worker.schedule_and_wait(repository=repo, character_id='role', user_scope=scope,
            message=source, history=history, rule_hints=[], source_message_id='budget-source')
        assert result['status'] == 'failed' and result['reason'] == 'input_budget'
        assert result['persisted'] == 0 and result['source_capture'] == 'recorded'
        assert await repo.list_memory_records('role', scope) == []
        assert (await repo.list_sources('role', scope))[0]['body'] == source
    finally:
        await worker.shutdown(timeout=1)
