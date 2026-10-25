import json

import pytest

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
        return json.dumps(dict(memories=[dict(attributed_to='user', kind='like', value=value,
            evidence='我喜欢红茶', confidence=0.99, operation='ADD')]))
    rejected = audit_call(call(output('咖啡')))
    assert rejected['status'] == 'proposal_failed' and rejected['field'] == 'value' and rejected['accepted'] == []
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


def test_audit_reports_invalid_call_and_continues_to_valid_call(tmp_path, monkeypatch, capsys):
    from evaluation.memory_writer_audit import main

    def output(value):
        return json.dumps(dict(memories=[dict(kind='like', value=value, evidence='我喜欢红茶',
            confidence=.99, operation='ADD', attributed_to='user')]))

    trace = tmp_path / 'trace.jsonl'
    target = tmp_path / 'audit.jsonl'
    trace.write_text(json.dumps(dict(case='synthetic', index=0,
        writer_calls=[call(output('咖啡')), call(output('红茶'))]), ensure_ascii=False) + '\n', encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['memory-writer-audit', '--trace', str(trace), '--output', str(target)])
    main()
    rows = [json.loads(line) for line in target.read_text(encoding='utf-8').splitlines()]
    assert [row['status'] for row in rows] == ['proposal_failed', 'admitted']
    assert rows[0]['field'] == 'value' and not rows[0]['accepted']
    assert len(rows[1]['accepted']) == 1 and rows[1]['accepted'][0]['memory']['memory_key'] == 'preference_红茶'
    assert [row['call'] for row in rows] == [0, 1]
    summary = json.loads(capsys.readouterr().out)
    assert summary['writer_calls'] == 2 and summary['statuses'] == {'proposal_failed': 1, 'admitted': 1}
    assert summary['fresh_model_calls'] == summary['database_writes'] == 0


def test_audit_does_not_hide_unexpected_parser_failures(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError('internal parser failure')

    monkeypatch.setattr('evaluation.memory_writer_audit.parse_llm_proposals', broken)
    with pytest.raises(RuntimeError, match='internal parser failure'):
        audit_call(call('{"memories":[]}'))
