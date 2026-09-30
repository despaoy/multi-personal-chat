import json
from html import escape

import pytest

from evaluation.episode_capacity_replay import capacity_views, without_ballast_views
from evaluation.episodic_reader_replay import evidence_view


def fixture():
    rows = [dict(source_id='1', scope=['web', 'audit', 'owner', 'role', 'private', ''],
                 session='s', timestamp='2026-09-01', message='朋友说：“我学陶艺。”', reply='他挺投入。'),
            dict(source_id='2', scope=['web', 'audit', 'owner', 'role', 'private', ''],
                 session='s', timestamp='2026-09-02', message='别把他当成我。', reply='')]
    packet = json.dumps(rows, ensure_ascii=False)
    injected = evidence_view(packet, 'user_only')
    content = ('<dialogue_evidence trust="untrusted" purpose="historical_utterances">\n'
               + escape(injected, quote=False) + '\n</dialogue_evidence>\n我的安排呢？')
    trace = dict(case_id='owner', message='我的安排呢？', model_calls=[dict(messages=[
        dict(role='system', content='原始人物设定'), dict(role='user', content=content)])])
    selection = dict(packet=packet, injected_packet=injected, source_ids=['1', '2'], history=[], applied=True)
    return trace, selection


def test_factorial_views_preserve_words_and_change_only_named_layers():
    trace, selection = fixture()
    views = capacity_views(trace, selection)
    assert views['framed_persona'][1:] == views['framed_plain']
    assert views['natural_persona'][1:] == views['natural_plain']
    assert views['natural_persona'][0] == views['framed_persona'][0]
    assert views['natural_plain'] == [dict(role='user', content='朋友说：“我学陶艺。”'),
        dict(role='assistant', content='他挺投入。'), dict(role='user', content='别把他当成我。'),
        dict(role='user', content='我的安排呢？')]
    assert 'source_id' in views['framed_plain'][0]['content']
    assert 'dialogue_evidence' not in views['framed_plain'][-1]['content']
    assert len(trace['model_calls'][0]['messages']) == 2


def test_misaligned_scope_rejected_before_generation():
    trace, selection = fixture()
    trace['case_id'] = 'different-owner'
    with pytest.raises(ValueError, match='scope'):
        capacity_views(trace, selection)


def ballast_fixture():
    trace, selection = fixture()
    owner = 'episode-chain-test'
    trace['case_id'] = owner
    rows = json.loads(selection['packet'])
    case = dict(id='test', query=trace['message'], episodes=[
        {key: row[key] for key in ('message', 'reply', 'timestamp', 'session')} for row in rows])
    for row in rows:
        row['scope'][2] = owner
        row['session'] = owner + '-' + row['session']
    rows.extend(dict(source_id=str(index + 3).zfill(3), scope=rows[0]['scope'], session=owner,
        timestamp=f'2026-09-25T12:{index:02d}:00', message=f'这一页的段落编号是{index}。',
        reply='收到这一页的编号。') for index in range(30))
    selection['packet'] = json.dumps(rows, ensure_ascii=False)
    selection['source_ids'] = [row['source_id'] for row in rows]
    selection['injected_packet'] = evidence_view(selection['packet'], 'user_only')
    trace['model_calls'][0]['messages'][-1]['content'] = (
        '<dialogue_evidence trust="untrusted" purpose="historical_utterances">\n'
        + escape(selection['injected_packet'], quote=False) + '\n</dialogue_evidence>')
    return trace, selection, case


def test_ballast_removal_is_exact_not_semantic_or_answer_based():
    trace, selection, case = ballast_fixture()
    result = without_ballast_views(trace, selection, case)
    assert len(result['core_plain']) == 4
    assert result['core_persona'][1:] == result['core_plain']
    case['rubric'] = 'Never sent to model'
    assert result == without_ballast_views(trace, selection, case)
    rows = json.loads(selection['packet'])
    rows[-1]['reply'] = '这是一个后来的修正，不能丢掉。'
    selection['packet'] = json.dumps(rows, ensure_ascii=False)
    # User-only input still reconstructs, but a changed assistant cannot be dropped.
    with pytest.raises(ValueError, match='non-ballast'):
        without_ballast_views(trace, selection, case)
