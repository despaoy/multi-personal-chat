from dataclasses import replace

import pytest

from character.memory_service import CharacterMemoryService
from character.models import CompiledCharacterContext, UserScope
from inference.generation_request import GenerationRequest, generate_character_response
from inference.memory_response import render_memory_response, storage_fields


class Repo:
    def __init__(self, rows=(), error=False):
        self.rows = rows
        self.error = error
        self.scope = None

    async def list_memory_records(self, character, scope, **kwargs):
        self.scope = (character, scope)
        if self.error:
            raise RuntimeError('unavailable')
        return self.rows


async def status(rows=(), *, limit=None, error=False, query='你保存了我的姓名吗？'):
    repo = Repo(rows, error=error)
    service = CharacterMemoryService(repo, candidate_limit=limit, semantic_enabled=False)
    scope = UserScope('web', 'test', 'user-a', 'chat-a', 'private')
    items, _, trace = await service.recall_with_diagnostics('character-a', scope, query)
    assert repo.scope == ('character-a', scope)
    return items, trace


@pytest.mark.asyncio
@pytest.mark.parametrize('rows,expected', [
    ([], False),
    ([{'id': '1', 'memory_key': 'user_name', 'content': '用户叫温宁'}], True),
    ([{'id': '1', 'memory_key': 'user_name', 'content': '用户叫温宁', 'status': 'retracted'}], False),
    ([{'id': '1', 'memory_key': 'user_name', 'content': '用户叫温宁', 'status': 'pending'}], False),
    ([{'id': '1', 'memory_key': 'user_name', 'content': '用户叫温宁', 'valid_to': '2000-01-01'}], False),
    ([{'id': '1', 'memory_key': 'preference_x', 'content': '用户喜欢读书'}], False),
])
async def test_scoped_current_presence_is_not_retrieval_relevance(rows, expected):
    _, trace = await status(rows)
    assert trace['field_presence']['name'] is expected


@pytest.mark.asyncio
async def test_partial_read_and_error_never_prove_absence():
    _, partial = await status(limit=1)
    _, failed = await status(error=True)
    assert 'field_presence' not in partial and 'field_presence' not in failed
    assert failed['status'] == 'retrieval_error'


@pytest.mark.asyncio
async def test_presence_survives_irrelevant_query():
    row = {'id': '1', 'memory_key': 'user_name', 'content': '用户说自己叫温宁'}
    _, trace = await status([row], query='我的专业是什么？')
    assert trace['field_presence']['name'] is True
    assert trace['field_presence']['major'] is False


@pytest.mark.asyncio
async def test_completed_storage_read_survives_later_ranking_failure(monkeypatch):
    import character.memory_service as memory_service

    def broken(*args):
        raise RuntimeError('ranking failed')

    monkeypatch.setattr(memory_service, '_retrieval_text', broken)
    _, trace = await status([{'id': '1', 'memory_key': 'user_name', 'content': '用户说自己叫温宁'}])
    assert trace['status'] == 'retrieval_error'
    assert trace['field_presence']['name'] is True


@pytest.mark.parametrize('message', ['你保存了我的姓名吗？', '你记住我的姓名了吗？',
                                  '你有没有记录我的姓名？', '请问你是否保存我的姓名？'])
def test_storage_query(message):
    assert storage_fields(message) == ('name',)


@pytest.mark.parametrize('message', ['请保存我的姓名。', '你能记住我的姓名吗？',
    '你保存了我朋友的姓名吗？', '你以前保存过我的姓名吗？', '你保存了我的姓名吗，顺便给我建议。',
    '我叫温宁，你保存了吗？', '你保存了我的姓名。', '不要保存我的姓名。'])
def test_non_status_tasks_are_not_swallowed(message):
    assert storage_fields(message) == ()


@pytest.mark.asyncio
async def test_empty_authoritative_snapshot_beats_third_party_history_without_model():
    _, trace = await status()
    context = CompiledCharacterContext('', '', '', memory_status='no_match',
                                      memory_field_presence=tuple(trace['field_presence'].items()))

    async def model(**kwargs):
        pytest.fail('Stored state must not be guessed from conversation names')

    result = await generate_character_response(GenerationRequest(message='你记住我的姓名了吗？',
        history=({'role': 'user', 'content': '我朋友叫温宁。'},), character_context=context), model)
    assert '没有你的姓名记录' in result.reply and '温宁' not in result.reply
    assert result.response_mode == 'memory_storage_status' and not result.model_invoked


def test_snapshot_is_independent_of_injection_budget_and_not_inferred_from_no_match():
    base = CompiledCharacterContext('', '', '', memory_status='no_match')
    assert '暂时不能确认' in render_memory_response('你保存了我的名字吗？', base)
    present = replace(base, memory_field_presence=(('name', True), ('major', False)))
    reply = render_memory_response('你保存了我的姓名和专业吗？', present)
    assert '有你的姓名记录' in reply and '没有你的专业记录' in reply
    assert present.used_memory_ids == ()


@pytest.mark.asyncio
@pytest.mark.parametrize('presence', [(), (('name', False),), (('name', True),)])
async def test_api_status_task_needs_neither_rag_nor_model(monkeypatch, presence):
    from types import SimpleNamespace

    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector

    monkeypatch.setattr(generate, '_get_system_prompt', lambda _: 'persona')
    monkeypatch.setattr(intent_detector, 'needs_rag', lambda _: pytest.fail('Storage status is not external knowledge'))

    async def model(**kwargs):
        pytest.fail('Storage status needs no model')

    prepared = SimpleNamespace(history=(), compiled=CompiledCharacterContext('', '', '',
                                memory_field_presence=presence))
    _, used_rag, meta = await generate._generate_with_vllm(MessageRequest(message='你保存了我的姓名吗？'),
        None, prepared_character_turn=prepared, runtime_config={'useKnowledgeBase': True}, model_generate=model)
    assert not used_rag and meta['answerMode'] == 'memory_storage_status'
    assert meta['modelInvoked'] is False
