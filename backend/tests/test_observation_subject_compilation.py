import json
from dataclasses import replace
from html import unescape

import pytest

from character.context_builder import MEMORY_REFERENCE_DISCLAIMER, compile_reference_context
from character.memory_service import CharacterMemoryService
from character.memory_subject import is_source_observation
from character.models import CompiledCharacterContext, MemoryItem, UserScope
from db.database import SQLiteDB
from inference.generation_request import GenerationRequest, build_generation_request
from repositories.character_memory import DatabaseCharacterMemoryRepository


def observation():
    return MemoryItem('1', 'shared_event', '原话：我妹妹在学陶艺。',
                      evidence=('我妹妹在学陶艺。',), source_observation=True)


@pytest.mark.parametrize('complete', [False, True])
def test_observation_never_inherits_personal_subject_disclaimer(complete):
    text, ids = compile_reference_context((observation(),), complete_evidence=complete, observation_semantics=True)
    assert ids == ('1',)
    assert MEMORY_REFERENCE_DISCLAIMER not in text
    if complete:
        packet = json.loads(text.splitlines()[1][2:])
        assert packet['subject_scope'] == 'not_resolved'
        assert packet['speaker_role'] == 'user'
    else:
        assert '描述主体=未解析' in text
    plan = build_generation_request(GenerationRequest(message='陶艺是谁学的？',
        character_context=CompiledCharacterContext('', '', text)))
    assert text in unescape(plan.messages[-1]['content'])
    assert not any(text in m['content'] for m in plan.messages if m['role'] == 'system')


def test_mixed_packets_keep_individual_subjects_and_all_ids():
    fact = MemoryItem('2', 'user_fact', '用户喜欢红茶')
    text, ids = compile_reference_context((observation(), fact), complete_evidence=True, observation_semantics=True)
    rows = [json.loads(line[2:]) for line in text.splitlines()[1:]]
    assert ids == ('1', '2')
    assert [r['subject_scope'] for r in rows] == ['not_resolved', 'current_user_not_character']
    ordinary, _ = compile_reference_context((fact,))
    assert ordinary.startswith(MEMORY_REFERENCE_DISCLAIMER)


def test_filtered_observation_does_not_change_other_reference_contract():
    fact = MemoryItem('2', 'user_fact', '用户喜欢红茶')
    text, ids = compile_reference_context((replace(observation(), status='erased'), fact))
    assert ids == ('2',) and text.startswith(MEMORY_REFERENCE_DISCLAIMER)


def test_model_actor_label_does_not_create_observation_envelope():
    assert not is_source_observation(dict(metadata=dict(attributed_to='user')))
    assert not is_source_observation(dict(metadata='quoted_source'))


@pytest.mark.asyncio
async def test_real_repository_and_retrieval_preserve_observation_semantics(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'observations.db'))
    scope = UserScope('web', 'web', 'u', 'u', 'private')
    await repo.append_claim('c', scope, observation(), memory_key='event_陶艺',
        metadata=dict(content_semantics='quoted_source', speaker_role='user', described_subject='not_resolved'),
        evidence=('我妹妹在学陶艺。',))
    service = CharacterMemoryService(repo, semantic_enabled=False)
    direct = await repo.list_memories('c', scope)
    assert direct[0].source_observation
    assert direct[0].evidence == ('我妹妹在学陶艺。',)
    items, _ = await service.load_relevant_memories('c', scope, '陶艺')
    assert len(items) == 1 and items[0].source_observation
    text, _ = compile_reference_context(tuple(items), complete_evidence=True, observation_semantics=True)
    assert json.loads(text.splitlines()[1][2:])['subject_scope'] == 'not_resolved'


def test_new_metadata_does_not_enable_rejected_rendering_by_default():
    for complete in (False, True):
        assert compile_reference_context((observation(),), complete_evidence=complete) == compile_reference_context(
            (replace(observation(), source_observation=False),), complete_evidence=complete)
