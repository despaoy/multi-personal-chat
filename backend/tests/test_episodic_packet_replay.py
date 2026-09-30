import json
from html import escape

import pytest

from evaluation.episodic_packet_replay import compact_packet, expand_packet, provenance_packet, replace_packet


def packet():
    return json.dumps([dict(source_id=str(i), scope=['web', 'owner'], session='room' + str(i % 2),
        timestamp='2026-09-26T00:00:00', message='原话<&>\n取消的是“那个”？' + str(i))
        for i in range(30)], ensure_ascii=False, separators=(',', ':'))


def test_roundtrip_preserves_order_scopes_sources_times_and_exact_text():
    raw = packet()
    compact = compact_packet(raw)
    assert expand_packet(compact) == json.loads(raw)
    assert len(compact) < len(raw)


def test_replacement_changes_only_packet_not_system_history_or_query():
    raw = packet()
    messages = [dict(role='system', content='系统不变'), dict(role='user', content='旧问题'),
                dict(role='assistant', content='旧回复'),
                dict(role='user', content='资料\n' + escape(raw, quote=False) + '\n问题不变')]
    packed = compact_packet(raw)
    result = replace_packet(messages, raw, packed)
    assert result[:-1] == messages[:-1]
    assert result[-1]['content'] == '资料\n' + escape(packed, quote=False) + '\n问题不变'
    assert raw not in result[-1]['content']
    assert escape(raw, quote=False) in messages[-1]['content']


@pytest.mark.parametrize('change', ['scope', 'extra', 'empty'])
def test_invalid_or_mixed_sources_rejected(change):
    rows = json.loads(packet())
    if change == 'scope':
        rows[1]['scope'] = ['other']
    elif change == 'extra':
        rows[1]['reply'] = 'do not silently lose fields'
    else:
        rows = []
    with pytest.raises(ValueError):
        compact_packet(json.dumps(rows))


def test_missing_or_repeated_packet_rejected():
    raw = packet()
    for content in ['none', escape(raw, quote=False) * 2]:
        with pytest.raises(ValueError):
            replace_packet([dict(role='user', content=content)], raw, compact_packet(raw))


def test_provenance_marks_speaker_not_fact_subject_or_current_state():
    rows = json.loads(packet())
    for row in rows:
        row['scope'] = ['web', 'audit', 'alice', 'character', 'private', '']
    rows[0]['message'] = '朋友说：“我在剧本里改了安排。”'
    rendered = provenance_packet(json.dumps(rows, ensure_ascii=False))
    assert expand_packet(rendered) == rows
    provenance = json.loads(rendered)['provenance']
    assert provenance['speaker_role'] == 'user'
    assert provenance['speaker_id'] == 'alice'
    assert provenance['described_subject'] == 'not_resolved'
    assert provenance['current_validity'] == 'not_resolved'
    with pytest.raises(ValueError):
        provenance_packet(packet())
