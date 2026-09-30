import pytest

from character.context_builder import build_user_scope
from character.memory_extractor import extract_memories
from character.memory_mentions import mention_query, review_mentions
from character.memory_service import CharacterMemoryService
from character.models import CompiledCharacterContext
from character.rule_memory_writer import write_rule_memory
from db.database import SQLiteDB
from inference.memory_response import render_memory_response
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import TurnInput, build_character_context_service


@pytest.mark.parametrize('message', ['我提过哪些游泳限制？', '我说过什么专业？',
                                    '请列出我之前说过哪些工作地点。'])
def test_complete_mention_tasks(message):
    assert mention_query(message) is not None


@pytest.mark.parametrize('message', ['我的专业是什么？', '我提过哪些限制，顺便给我建议。',
    '我朋友提过哪些限制？', '我提过哪些朋友的限制？', '删除我说过哪些名字。'])
def test_other_tasks_are_not_executed_as_record_review(message):
    assert mention_query(message) is None


def row(status='pending', text='我只有放假才去游泳。'):
    item, = extract_memories(text)
    return dict(id='m1', memory_key=item.memory_key, status=status, evidence=[text],
        source_message_ids=['source1'], metadata={'origin': 'rule_candidate', 'qualifiers': dict(item.qualifiers)})


@pytest.mark.parametrize('status', ['retracted', 'erased', 'unknown'])
def test_deleted_or_unsupported_states_are_not_resurrected(status):
    answer = review_mentions('我提过哪些游泳限制？', [row(status)], complete_read=True)
    assert '我只有放假才去游泳' not in answer
    assert '不代表你从未说过' in answer


def test_pending_quote_carries_status_without_becoming_current_fact():
    answer = review_mentions('我提过哪些游泳限制？', [row()], complete_read=True)
    assert '待确认，不能当作当前事实' in answer
    assert '我只有放假才去游泳。' in answer
    assert '当前记录]' not in answer


@pytest.mark.parametrize('change', [dict(source_message_ids=[]), dict(evidence=['模型猜测用户放假游泳']),
                                  dict(metadata={'origin': 'llm'}), dict(relation_type='ERASE')])
def test_unverifiable_or_erased_provenance_is_not_quoted(change):
    answer = review_mentions('我提过哪些游泳限制？', [{**row(), **change}], complete_read=True)
    assert '我只有放假才去游泳' not in answer


def test_topic_does_not_return_unrelated_constraints():
    answer = review_mentions('我提过哪些驾驶限制？', [row()], complete_read=True)
    assert '游泳' not in answer


def test_bounded_review_labels_partial_results_and_keeps_quotes_whole():
    rows = [row(text=f'我只有条件{i}才去游泳。') for i in range(12)]
    answer = review_mentions('我提过哪些游泳限制？', rows, complete_read=True)
    assert answer.count('- [') == 8 and '部分可核对记录' in answer
    assert all(f'我只有条件{i}才去游泳。' in answer for i in range(8))
    assert '条件8' not in answer


def test_review_from_a_different_query_is_not_reused():
    context = CompiledCharacterContext('', '', '', memory_review_query='我说过哪些专业？',
                                      memory_review_text='private old review')
    assert 'private old review' not in render_memory_response('我提过哪些限制？', context)


def test_unknown_read_is_not_reported_as_empty_history():
    result = render_memory_response('我提过哪些限制？', CompiledCharacterContext('', '', ''))
    assert '没有取得' in result and '从未' not in result


@pytest.mark.asyncio
@pytest.mark.parametrize('use_kb', [False, True])
async def test_provenance_display_uses_no_external_rag_or_model(monkeypatch, use_kb):
    from types import SimpleNamespace

    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector

    message = '我提过哪些游泳限制？'
    context = CompiledCharacterContext('', '', '', memory_review_query=message,
        memory_review_text=review_mentions(message, [row()], complete_read=True))
    async def model(**kwargs):
        pytest.fail('Provenance display must not invent or rewrite statements')

    monkeypatch.setattr(generate, '_get_system_prompt', lambda _: 'persona')
    monkeypatch.setattr(intent_detector, 'needs_rag', lambda _: pytest.fail('Not external knowledge'))
    reply, used_rag, meta = await generate._generate_with_vllm(MessageRequest(message=message), None,
        runtime_config={'useKnowledgeBase': use_kb}, prepared_character_turn=SimpleNamespace(history=(), compiled=context),
        model_generate=model)
    assert '待确认' in reply and not used_rag
    assert meta['modelInvoked'] is False and meta['answerMode'] == 'memory_mentions'


@pytest.mark.asyncio
async def test_real_storage_review_is_separate_from_fact_recall_and_user_scope(tmp_path, monkeypatch):
    monkeypatch.setenv('CONTEXTUAL_MEMORY_SELECTION_ENABLED', 'false')
    database = SQLiteDB(tmp_path / 'mentions.sqlite')
    repo = DatabaseCharacterMemoryRepository(database)
    scope = build_user_scope('web', 'test', 'u1', 'session-a', 'private')
    for i, text in enumerate(['我只有周末才去游泳。', '我只有放假才去游泳。']):
        item, = extract_memories(text)
        await write_rule_memory(repo, 'tsukiyashiro_kisaki', scope, item, f'source{i}')
    service = build_character_context_service(database)
    service._memory_service = CharacterMemoryService(repo, semantic_enabled=True)
    monkeypatch.setattr(service._memory_service, '_semantic_similarities',
                        lambda *args: pytest.fail('A record review requires no embedding/ranking'))
    message = '我提过哪些游泳限制？'
    prepared = await service.prepare_turn(TurnInput(message, 'web', 'test', 'u1', 'session-b', 'private'),
                                          'tsukiyashiro_kisaki')
    answer = render_memory_response(message, prepared.compiled)
    assert '周末' in answer and '放假' in answer and '待确认' in answer
    assert prepared.history == ()
    assert all(item.status != 'pending' for item in prepared.compiled.memory_packets)
    assert '放假' not in prepared.compiled.reference_context
    other = await service.prepare_turn(TurnInput(message, 'web', 'test', 'u2', 'session-b', 'private'),
                                      'tsukiyashiro_kisaki')
    assert '放假' not in render_memory_response(message, other.compiled)
    service._memory_service = CharacterMemoryService(repo, semantic_enabled=False)
    ordinary = await service.prepare_turn(TurnInput('我的限制有哪些？', 'web', 'test', 'u1', 'session-b', 'private'),
                                         'tsukiyashiro_kisaki')
    assert ordinary.compiled.memory_review_text == ''
    assert all(item.status != 'pending' for item in ordinary.compiled.memory_packets)
