import json
from copy import deepcopy
from html import escape

import pytest

from evaluation.episode_role_replay import role_views
from evaluation.episodic_reader_replay import evidence_view


def fixture():
    rows = [dict(source_id=str(i), scope=['web', 'audit', 'case', 'char', 'private', ''],
        session='s' + str(i), timestamp=f'2026-09-2{i}T10:00:00',
        message='小说里他说：“我来自海边。” <原话>', reply='历史助手可能说错。') for i in range(2)]
    packet = json.dumps(rows, ensure_ascii=False, separators=(',', ':'))
    user_packet = evidence_view(packet, 'user_only')
    selection = dict(packet=packet, injected_packet=user_packet, source_ids=['0', '1'],
                     applied=True, history=[])
    messages = [dict(role='system', content='人物设定不变'), dict(role='user',
        content='前文\n<dialogue_evidence trust="untrusted" purpose="historical_utterances">\n'
        + escape(user_packet, quote=False) + '\n</dialogue_evidence>\n<user_query>我的家乡？</user_query>')]
    return messages, selection


def test_roles_preserve_system_query_original_utterance_and_source_order():
    messages, selection = fixture()
    snapshot = deepcopy(messages)
    views = role_views(messages, selection, 'case')
    assert messages == snapshot
    for view in views.values():
        assert view[0] == messages[0]
        assert view[-1]['content'].endswith('<user_query>我的家乡？</user_query>')
    pairs = views['history_pairs']
    assert [row['role'] for row in pairs] == ['system', 'user', 'assistant', 'user', 'assistant', 'user']
    assert pairs[1]['content'].endswith('小说里他说：“我来自海边。” <原话>')
    assert json.loads(pairs[1]['content'].split('\n')[0])['source_id'] == '0'
    assert '历史助手可能说错。' not in str(views['history_users'])
    assert '历史助手可能说错。' in str(views['flat_pairs'])
    assert 'dialogue_evidence' not in pairs[-1]['content']


@pytest.mark.parametrize('change', ['history', 'scope', 'order', 'ids', 'injection', 'empty_user'])
def test_ambiguous_replay_inputs_fail_closed(change):
    messages, selection = fixture()
    rows = json.loads(selection['packet'])
    if change == 'history':
        selection['history'] = [dict(role='user', content='possibly overlapping')]
    elif change == 'scope':
        rows[0]['scope'][2] = 'other'
    elif change == 'order':
        rows.reverse()
        selection['source_ids'].reverse()
    elif change == 'ids':
        selection['source_ids'] = ['0']
    elif change == 'injection':
        selection['injected_packet'] = '[]'
    else:
        rows[0]['message'] = ''
    selection['packet'] = json.dumps(rows, ensure_ascii=False, separators=(',', ':'))
    with pytest.raises(ValueError):
        role_views(messages, selection, 'case')


def test_does_not_invent_assistant_turn_when_reply_is_empty():
    messages, selection = fixture()
    rows = json.loads(selection['packet'])
    rows[0]['reply'] = ''
    selection['packet'] = json.dumps(rows, ensure_ascii=False, separators=(',', ':'))
    views = role_views(messages, selection, 'case')
    assert len(views['history_pairs']) == 5
