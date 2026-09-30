import copy
import json
from html import escape

import pytest

from evaluation.source_role_ablation import request_variants


def fixture():
    raw = json.dumps(dict(source_kind='historical_user_utterances', speaker_role='user', records=[
        dict(source_id='s1', observed_at='2026-09-27T01:00:00+00:00', text='如果录取了，我才会搬家。'),
        dict(source_id='s2', observed_at='2026-09-27T02:00:00+00:00', text='我现在还没收到通知。')]), ensure_ascii=False)
    packet = ('<dialogue_evidence trust="untrusted" purpose="historical_utterances">\n'
              + escape(raw, quote=False) + '\n</dialogue_evidence>')
    return dict(prepared=dict(compiled=dict(episodic_reference_context=raw)), model_calls=[dict(request=dict(messages=[
        dict(role='system', content='原有人物设定'),
        dict(role='user', content=packet+'\n<user_query>我已经搬家了吗？</user_query>')]))])


def test_moves_sources_without_changing_system_query_or_input():
    row = fixture()
    before = copy.deepcopy(row)
    variants = request_variants(row)
    assert row == before
    original = row['model_calls'][0]['request']['messages']
    assert variants['full'] == original
    for mode, messages in variants.items():
        assert messages[0] == original[0]
        assert messages[-1]['content'].endswith('<user_query>我已经搬家了吗？</user_query>')
        assert all(m['role'] == 'user' for m in messages[1:])
        if mode != 'full':
            assert 'dialogue_evidence' not in messages[-1]['content']
    raw = json.loads(row['prepared']['compiled']['episodic_reference_context'])
    assert [m['content'] for m in variants['speech_history'][1:-1]] == [r['text'] for r in raw['records']]
    assert variants['separate_packet'][1]['content'] in original[-1]['content']


@pytest.mark.parametrize('role', ['assistant', 'user', 'tool'])
def test_rejects_non_cold_requests(role):
    row = fixture()
    row['model_calls'][0]['request']['messages'].insert(1, dict(role=role, content='history'))
    with pytest.raises(ValueError, match='Cold reads'):
        request_variants(row)


@pytest.mark.parametrize('mutation', ['missing_packet', 'duplicate_packet', 'assistant_source', 'missing_source_id'])
def test_does_not_reconstruct_missing_or_unverified_source(mutation):
    row = fixture()
    messages = row['model_calls'][0]['request']['messages']
    if mutation == 'missing_packet':
        messages[-1]['content'] = 'question only'
    elif mutation == 'duplicate_packet':
        messages[-1]['content'] *= 2
    else:
        compiled = row['prepared']['compiled']
        raw = json.loads(compiled['episodic_reference_context'])
        if mutation == 'assistant_source':
            raw['speaker_role'] = 'assistant'
        else:
            raw['records'][0].pop('source_id')
        compiled['episodic_reference_context'] = json.dumps(raw)
    with pytest.raises(ValueError):
        request_variants(row)
