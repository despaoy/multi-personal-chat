import json

from evaluation.memory_writer_audit import audit_call


def call(output):
    return dict(messages=[dict(role='user', content=json.dumps(dict(
        current_user_message='我喜欢红茶。', existing_memories=[], recent_history=[])))], output=output)


def test_format_failure_is_not_empty_success():
    assert audit_call(call('{"memories":[}}'))['status'] == 'format_failed'
    assert audit_call(call('{"memories":{}}'))['status'] == 'format_failed'
    assert audit_call(call('{"memories":[]}'))['status'] == 'empty_candidates'


def test_valid_structure_is_not_evidence_support():
    def output(value):
        return json.dumps(dict(memories=[dict(kind='like', value=value,
            evidence='我喜欢红茶', confidence=0.99, operation='ADD')]))
    rejected = audit_call(call(output('咖啡')))
    assert rejected['status'] == 'none_admitted' and rejected['accepted'] == []
    accepted = audit_call(call(output('红茶')))
    assert accepted['status'] == 'admitted' and len(accepted['accepted']) == 1


def test_replay_preserves_recorded_whitelist_history_and_threshold(monkeypatch):
    observed = {}

    def parse(text, **kwargs):
        observed.update(kwargs)
        return []

    monkeypatch.setattr('evaluation.memory_writer_audit.parse_llm_proposals', parse)
    payload = dict(current_user_message='刚才那条不对。',
        recent_history=[dict(role='user', content='以前的自述')],
        existing_memories=[dict(id='41', memory_key='user_name')],
        feedback_target_ids=['41'], confidence_threshold=0.92)
    audit_call(dict(messages=[dict(role='user', content=json.dumps(payload))],
                    output='{"memories":[]}'))
    assert observed['source_message'] == payload['current_user_message']
    assert observed['history'] == tuple(payload['recent_history'])
    assert observed['existing_memories'] == tuple(payload['existing_memories'])
    assert observed['feedback_target_ids'] == ('41',)
    assert observed['confidence_threshold'] == 0.92
