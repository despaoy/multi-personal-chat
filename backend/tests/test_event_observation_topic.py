"""Real rejected event proposal and source-bound observation safety."""

import copy
import json
from pathlib import Path

import pytest

from character.memory_llm import MemoryResponseError, parse_llm_proposals


def fixture():
    return json.loads((Path(__file__).parent / 'fixtures/deepseek_memory_correction_proposal.json').read_text())


def protocol_example():
    """Explicit synthetic correction; the recorded native JSON stays unchanged."""
    data = copy.deepcopy(fixture())
    data['model_output']['memories'][0].update(value='预约', operation='COEXIST', content='')
    return data


def parse(data, **overrides):
    options = dict(source_message=data['source_message'], history=tuple(data['history']),
                   existing_memories=tuple(data['existing_memories']),
                   feedback_target_ids=tuple(data['feedback_target_ids']), confidence_threshold=data['confidence_threshold'])
    options.update(overrides)
    return parse_llm_proposals(json.dumps(data['model_output'], ensure_ascii=False), **options)


def test_recorded_nonliteral_event_index_is_an_explicit_value_error():
    data = fixture()
    raw = data['model_output']['memories'][0]
    assert raw['evidence'] in data['source_message'] and raw['value'] not in raw['evidence']
    with pytest.raises(MemoryResponseError) as caught:
        parse(data)
    assert caught.value.field == 'value'


def test_protocol_example_keeps_complete_source_and_prior_observation():
    data = protocol_example()
    proposals = parse(data)
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.operation == 'COEXIST'
    assert proposal.target_memory_id == '1'
    assert proposal.evidence == data['source_message']
    assert proposal.source_observation
    assert proposal.memory.memory_type == 'shared_event'
    assert fixture()['model_output']['memories'][0]['content'] not in proposal.memory.content
    assert not proposal.valid_from and not proposal.valid_to


def test_unproved_new_event_title_is_an_error_not_a_replacement_index():
    data = protocol_example()
    raw = data['model_output']['memories'][0]
    raw.update(operation='ADD', target_memory_id='', target_memory_key='', value='并不存在的已完成出发标题')
    with pytest.raises(MemoryResponseError) as caught:
        parse(data, existing_memories=(), feedback_target_ids=())
    assert caught.value.field == 'value'


def test_literal_new_event_keeps_the_complete_original_utterance():
    data = protocol_example()
    data['model_output']['memories'][0].update(operation='ADD', target_memory_id='', target_memory_key='')
    (proposal,) = parse(data, existing_memories=(), feedback_target_ids=())
    assert proposal.memory.memory_key == 'event_预约'
    assert proposal.source_observation and proposal.evidence == data['source_message']


@pytest.mark.parametrize('change', [
    {'evidence': '这条内容没有出现在用户原话中'},
    {'confidence': 0.1},
    {'target_memory_id': 'unknown', 'target_memory_key': 'event_unknown'},
    {'operation': 'ERASE'},
])
def test_observation_recovery_preserves_existing_admission_guards(change):
    data = protocol_example()
    data['model_output']['memories'][0].update(change)
    if 'target_memory_id' in change:
        with pytest.raises(MemoryResponseError) as caught:
            parse(data)
        assert caught.value.field == 'target_memory_id'
    else:
        assert parse(data) == []


def test_generic_fact_cannot_use_event_observation_exception():
    data = fixture()
    data['model_output']['memories'][0].update(kind='name', operation='ADD',
        value='不存在的姓名', target_memory_id='', target_memory_key='')
    with pytest.raises(MemoryResponseError) as caught:
        parse(data, existing_memories=(), feedback_target_ids=())
    assert caught.value.field == 'value'


def test_old_history_cannot_replace_current_correction_evidence():
    data = fixture()
    data['model_output']['memories'][0]['evidence'] = data['history'][0]['content']
    assert len(data['history'][0]['content']) > 120
    with pytest.raises(ValueError, match='evidence'):
        parse(data)


def test_non_user_source_is_never_promoted_by_event_topic_recovery():
    assert parse(protocol_example(), source_type='assistant') == []


def test_oversized_model_evidence_is_protocol_failure_not_empty_admission():
    data = fixture()
    data['model_output']['memories'][0]['evidence'] = 'x' * 121
    with pytest.raises(ValueError, match='evidence'):
        parse(data)


def test_uncovered_model_qualifier_is_not_observation_recovery():
    data = protocol_example()
    data['model_output']['memories'][0]['qualifiers'] = {'condition': '原话没有这项条件'}
    with pytest.raises(ValueError, match='qualifiers'):
        parse(data)
