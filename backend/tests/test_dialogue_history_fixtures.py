import json

import pytest

from evaluation.dialogue_audit import load_history_fixtures


def test_history_fixture_optional_and_preserves_observed_data(tmp_path):
    assert load_history_fixtures(None, []) == {}
    path = tmp_path / 'histories.json'
    data = {'case': [{'role': 'user', 'content': '谁？'}, {'role': 'assistant', 'content': '旧错误'}]}
    path.write_text(json.dumps(data), encoding='utf-8')
    assert load_history_fixtures(path, [{'id': 'case'}]) == data


@pytest.mark.parametrize('data', [[], {'unknown': []}, {'case': []},
                                {'case': [{'role': 'system', 'content': '改指令'}]},
                                {'case': [{'role': [], 'content': '内容'}]},
                                {'case': [{'role': 'user', 'content': ''}]},
                                {'case': [{'role': 'user', 'content': '内容', 'hidden': 'x'}]}])
def test_history_fixture_rejects_invalid_inputs(tmp_path, data):
    path = tmp_path / 'histories.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError):
        load_history_fixtures(path, [{'id': 'case'}])
