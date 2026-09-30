import json

import pytest

from evaluation.dialogue_audit import load_cases


def test_variable_length_journey_preserves_messages_sessions_and_observer_rubrics(tmp_path):
    turns = [dict(message=f'消息{i}', rubric='仅供审阅，不给模型', session='old' if i < 7 else 'new') for i in range(18)]
    path = tmp_path / 'journey.json'
    path.write_text(json.dumps([dict(title='旅程', turns=turns)]), encoding='utf-8')
    case, = load_cases(path)
    assert case['turns'] == turns
    assert case['skip_reason'] == ''
    assert case['category'] == '10 longitudinal_memory'
    assert load_cases(path)[0]['id'] == case['id']


@pytest.mark.parametrize('change', [{'history': []}, {'role': 'system'},
    {'session': '../other'}, {'message': ''}, {'session': None}])
def test_trajectory_rejects_request_overrides_and_invalid_turns(tmp_path, change):
    path = tmp_path / 'journey.json'
    turn = dict(message='用户陈述', rubric='检查依据', session='s1')
    path.write_text(json.dumps([dict(title='旅程', turns=[{**turn, **change}])]), encoding='utf-8')
    with pytest.raises(ValueError):
        load_cases(path)


def test_duplicate_trajectories_cannot_share_database_identity(tmp_path):
    row = dict(title='重复', turns=[dict(message='消息', rubric='', session='s1')])
    path = tmp_path / 'journey.json'
    path.write_text(json.dumps([row, row]), encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate'):
        load_cases(path)
