import json
from copy import deepcopy

import pytest

from evaluation.memory_context_ablation import request_variants


def trace():
    quote = '我现在不喝含糖饮料。'
    reference = '观测原话：' + quote
    raw = json.dumps(dict(records=[dict(text=quote)]), ensure_ascii=False)
    messages = [{'role': 'system', 'content': 'unchanged persona'},
                {'role': 'user', 'content': quote},
                {'role': 'assistant', 'content': '这是之前的回答。'},
                {'role': 'user', 'content':
                    '<character_memory trust="untrusted" purpose="historical_reference">\n'
                    + reference + '\n</character_memory>\n'
                    + '<dialogue_evidence trust="untrusted" purpose="historical_utterances">\n'
                    + raw + '\n</dialogue_evidence>\n'
                    + '<user_query>你会怎么推荐？</user_query>'}]
    return dict(model_calls=[dict(request=dict(messages=messages))], prepared=dict(compiled=dict(
        memory_packets=[dict(temporal_mode='observation', evidence=[quote])],
        reference_context=reference, episodic_reference_context=raw)))


def test_factorial_ablation_preserves_originals_system_question_and_user_history():
    row = trace()
    original = deepcopy(row)
    variants = request_variants(row)
    assert row == original
    assert len(variants) == 4
    for messages in variants.values():
        assert messages[0] == original['model_calls'][0]['request']['messages'][0]
        assert messages[1] == original['model_calls'][0]['request']['messages'][1]
        assert messages[-1]['content'].endswith('<user_query>你会怎么推荐？</user_query>')
        assert original['prepared']['compiled']['episodic_reference_context'] in messages[-1]['content']
    assert len(variants['full']) == len(variants['without_observation_copy']) == 4
    assert len(variants['without_assistant_history']) == len(variants['without_both']) == 3
    assert '<character_memory' not in variants['without_observation_copy'][-1]['content']
    assert '<character_memory' in variants['without_assistant_history'][-1]['content']


@pytest.mark.parametrize('change', ['current_fact', 'missing_quote', 'changed_request'])
def test_ablation_rejects_lost_evidence_or_nonobservation_packet(change):
    row = trace()
    if change == 'current_fact':
        row['prepared']['compiled']['memory_packets'][0]['temporal_mode'] = 'fact'
    elif change == 'missing_quote':
        row['prepared']['compiled']['episodic_reference_context'] = '{"records":[]}'
    else:
        row['model_calls'][0]['request']['messages'][-1]['content'] = 'different content'
    with pytest.raises(ValueError):
        request_variants(row)
