"""Real rejected event proposal and source-bound observation safety."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from character.memory_llm import parse_llm_proposals


def fixture():
    return json.loads((Path(__file__).parent / 'fixtures/deepseek_memory_correction_proposal.json').read_text())


def parse(data, **overrides):
    options = dict(source_message=data['source_message'], history=tuple(data['history']),
                   existing_memories=tuple(data['existing_memories']),
                   feedback_target_ids=tuple(data['feedback_target_ids']), confidence_threshold=data['confidence_threshold'])
    options.update(overrides)
    return parse_llm_proposals(json.dumps(data['model_output'], ensure_ascii=False), **options)


def test_actual_rejected_correction_keeps_complete_source_and_prior_observation():
    data = fixture()
    raw = data['model_output']['memories'][0]
    assert raw['evidence'] in data['source_message'] and raw['value'] not in raw['evidence']
    proposals = parse(data)
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.operation == 'COEXIST'
    assert proposal.target_memory_id == '1'
    assert proposal.evidence == data['source_message']
    assert proposal.source_observation
    assert proposal.memory.memory_type == 'shared_event'
    assert raw['content'] not in proposal.memory.content
    assert not proposal.valid_from and not proposal.valid_to


def test_unproved_new_event_title_never_becomes_asserted_index_value():
    data = fixture()
    raw = data['model_output']['memories'][0]
    raw.update(operation='ADD', target_memory_id='', target_memory_key='', value='并不存在的已完成出发标题')
    proposals = parse(data, existing_memories=(), feedback_target_ids=())
    assert len(proposals) == 1
    proposal = proposals[0]
    digest = hashlib.sha256(data['source_message'].encode()).hexdigest()[:24]
    assert proposal.memory.memory_key == 'event_source_' + digest
    assert raw['value'] not in proposal.memory.memory_key
    assert raw['value'] not in proposal.memory.content
    assert proposal.evidence == data['source_message'] and proposal.source_observation


@pytest.mark.parametrize('change', [
    {'evidence': '这条内容没有出现在用户原话中'},
    {'confidence': 0.1},
    {'target_memory_id': 'unknown', 'target_memory_key': 'event_unknown'},
    {'qualifiers': {'condition': '原话没有这项条件'}},
    {'evidence': 'x' * 121},
    {'operation': 'ERASE'},
])
def test_observation_recovery_preserves_existing_admission_guards(change):
    data = fixture()
    data['model_output']['memories'][0].update(change)
    assert parse(data) == []


def test_generic_fact_cannot_use_event_observation_exception():
    data = fixture()
    data['model_output']['memories'][0].update(kind='name', operation='ADD',
        value='不存在的姓名', target_memory_id='', target_memory_key='')
    assert parse(data, existing_memories=(), feedback_target_ids=()) == []


def test_old_history_cannot_replace_current_correction_evidence():
    data = fixture()
    data['model_output']['memories'][0]['evidence'] = data['history'][0]['content']
    assert parse(data) == []


def test_non_user_source_is_never_promoted_by_event_topic_recovery():
    assert parse(copy.deepcopy(fixture()), source_type='assistant') == []
