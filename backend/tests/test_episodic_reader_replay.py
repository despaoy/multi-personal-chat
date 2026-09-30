import json

import pytest

from evaluation.episodic_reader_replay import evidence_view


@pytest.mark.parametrize('text', ['我周六看展。', '修改安排。\n明天不去。', '“引文”里面也有回复这个词。'])
def test_views_preserve_source_and_user_content(text):
    row = dict(source_id='1', scope=['user', 'role'], session='s', timestamp='2026-09-26',
               message=text, reply='旧回复，不是用户事实。')
    packet = json.dumps([row], ensure_ascii=False)
    assert evidence_view(packet, 'raw') == packet
    only = json.loads(evidence_view(packet, 'user_only'))[0]
    assert only == {k: v for k, v in row.items() if k != 'reply'}
    roles = json.loads(evidence_view(packet, 'roles'))[0]
    assert roles.pop('utterances') == [{'role': 'user', 'content': text},
                                      {'role': 'assistant', 'content': row['reply']}]
    assert roles == {k: v for k, v in row.items() if k not in {'message', 'reply'}}
    assert json.loads(packet) == [row]


def test_unknown_view_is_not_silently_accepted():
    with pytest.raises(ValueError):
        evidence_view('[]', 'summarize')
