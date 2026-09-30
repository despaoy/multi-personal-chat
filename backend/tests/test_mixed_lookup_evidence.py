"""A closed trailing lookup cannot erase a separately supported update."""
import json

import pytest

from character.memory_extractor import assertion_before_lookup, extract_memories
from character.memory_llm import parse_llm_proposals


def candidate(message, kind, value, **kwargs):
    return json.dumps({'memories': [dict(kind=kind, value=value, evidence=message,
        content='用户提到' + value, confidence=.96, operation='ADD', **kwargs)]}, ensure_ascii=False)


@pytest.mark.parametrize('statement,kind,value,key', [
    ('我搬到晋城了', 'location', '晋城', 'user_residence'),
    ('我已经搬家到江门了', 'location', '江门', 'user_residence'),
    ('我刚搬到衡阳了', 'location', '衡阳', 'user_residence'),
    ('我来自延吉', 'location', '延吉', 'user_origin'),
    ('我的专业是考古学', 'major', '考古学', 'user_major'),
    ('我叫林夏', 'name', '林夏', 'user_name'),
    ('我喜欢木工', 'like', '木工', 'preference_木工'),
])
@pytest.mark.parametrize('separator', ['，', '。', '；', '\n'])
def test_supported_statement_survives_closed_question(statement, kind, value, key, separator):
    message = statement + separator + '你还记得我叫什么吗？'
    assert assertion_before_lookup(message) == statement
    proposal, = parse_llm_proposals(candidate(message, kind, value), source_message=message)
    assert proposal.memory.memory_key == key
    assert proposal.evidence == statement
    assert '？' not in proposal.evidence
    assert any(item.memory_key == key for item in extract_memories(message))


@pytest.mark.parametrize('message', [
    '我明天搬到晋城了，你还记得我叫什么吗？',
    '我准备搬到晋城了，你还记得我叫什么吗？',
    '我可能搬到晋城了，你还记得我叫什么吗？',
    '如果我搬到晋城了，你还记得我叫什么吗？',
    '我如果搬到晋城了，你还记得我叫什么吗？',
    '我搬到晋城了吗，你还记得我叫什么吗？',
    '我朋友搬到晋城了，你还记得我叫什么吗？',
    '我搬到晋城了，你觉得这是真的吗？',
    '我搬到晋城了，你还记得我叫什么吗？其实是编的。',
    '我搬到晋城了，你还记得我叫什么吗？不要保存。',
    '“我搬到晋城了”，你还记得我叫什么吗？',
    '我搬到晋城了，但只住周末，你还记得我叫什么吗？',
])
def test_question_split_does_not_certify_uncertain_or_qualified_residence(message):
    assert not parse_llm_proposals(candidate(message, 'location', '晋城'), source_message=message)


@pytest.mark.parametrize('kind', ['location', 'user_fact'])
def test_existing_residence_can_be_superseded_with_scoped_evidence(kind):
    message = '我搬到宣城了，你还记得我的专业吗？'
    raw = dict(kind=kind, value='宣城', evidence=message, confidence=.96,
               operation='SUPERSEDE', target_memory_id='7', target_memory_key='user_residence')
    result, = parse_llm_proposals(json.dumps({'memories': [raw]}), source_message=message,
        existing_memories=({'id':'7', 'memory_key':'user_residence',
                           'content':'用户说自己居住在衡水', 'status':'active'},))
    assert result.target_memory_id == '7'
    assert result.memory.memory_key == 'user_residence'
    assert result.memory.content == '用户说自己居住在宣城'
    assert result.evidence == '我搬到宣城了'


def test_value_only_in_question_is_not_a_new_assertion():
    message = '我喜欢木工，我叫什么名字？'
    assert not parse_llm_proposals(candidate(message, 'name', '什么名字'), source_message=message)


@pytest.mark.parametrize('denial', [
    '前面搬家那句是编的。', '刚才那句不是真的。', '以上都是开玩笑。',
    '之前那个说法不属实。', '那句是假的。',
])
def test_clipped_model_evidence_cannot_omit_backward_denial(denial):
    evidence = '我搬到晋城了'
    message = evidence + '，你还记得我叫什么吗？' + denial
    assert not parse_llm_proposals(candidate(evidence, 'location', '晋城'), source_message=message)
    assert not extract_memories(message)


def test_independent_assertion_after_explicit_denial_is_still_available():
    message = '我搬到晋城了。前面那句是编的。我现在住在丹东。'
    items = extract_memories(message)
    assert len(items) == 1 and items[0].content == '用户说自己居住在丹东'
    proposal, = parse_llm_proposals(candidate('我现在住在丹东。', 'location', '丹东'),
                                  source_message=message)
    assert proposal.memory.content == '用户提到丹东'


@pytest.mark.parametrize('message', [
    '小说台词：“前面那句是编的。我现在住在丹东。”',
    '“我搬到晋城了。前面那句是编的。我现在住在丹东。”',
    '假设我搬到晋城了。前面那句是编的。我现在住在丹东。',
])
def test_retraction_does_not_close_an_existing_fiction_scope(message):
    assert not extract_memories(message)


@pytest.mark.asyncio
async def test_scoped_update_persists_versions_and_keeps_full_source(tmp_path):
    from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
    from character.memory_service import CharacterMemoryService
    from character.models import CompiledCharacterContext, UserScope
    from db.database import SQLiteDB
    from inference.memory_response import render_memory_response
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    statements = ['我现在住在衡水。', '我搬到宣城了，你还记得我的专业吗？']

    class Completion:
        async def complete(self, messages):
            payload = json.loads(messages[-1]['content'])
            source = payload['current_user_message']
            update = source == statements[1]
            proposal = dict(kind='location', value='宣城' if update else '衡水',
                evidence=source, confidence=.96, operation='SUPERSEDE' if update else 'ADD')
            if update:
                old, = payload['existing_memories']
                proposal.update(target_memory_id=old['memory_id'], target_memory_key=old['memory_key'])
            return json.dumps({'memories': [proposal]})

        async def close(self):
            pass

    class Embedding:
        def embed_texts(self, texts):
            return [[1.0, 0.0] for _ in texts]

    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'update.sqlite'))
    scope = UserScope('test', 'test', 'user', 'user', 'private')
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, 'unused', 'stub'),
                                          completion=Completion(), embedding_provider=Embedding())
    try:
        for index, message in enumerate(statements):
            assert scheduler.schedule(repository=repo, character_id='role', user_scope=scope,
                                      message=message, rule_hints=extract_memories(message),
                                      source_message_id=str(index))
            assert await scheduler.flush_memory(timeout=3)
        assert scheduler.status.saved == 2 and scheduler.status.failed == 0
        rows = await repo.list_memory_records('role', scope, include_inactive=True)
        active = [row for row in rows if row['status'] == 'active']
        assert len(active) == 1 and active[0]['content'] == '用户说自己居住在宣城'
        assert active[0]['evidence'] == ['我搬到宣城了']
        assert len(rows) == 2
        sources = await repo.list_sources('role', scope)
        assert {row['body'] for row in sources} == set(statements)
        service = CharacterMemoryService(repo, semantic_enabled=False)
        items, _, _ = await service.recall_with_diagnostics('role', scope, '我现在住哪里？')
        context = CompiledCharacterContext('', '', '', tuple(item.memory_id for item in items),
                                          memory_packets=items, memory_status='available')
        assert render_memory_response('我现在住哪里？', context) == '我这里记着的是：现在住在宣城。'
    finally:
        await scheduler.shutdown(timeout=3)
