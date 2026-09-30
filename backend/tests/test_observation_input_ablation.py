import json
from dataclasses import asdict

import pytest

from character.context_builder import compile_reference_context
from character.models import CompiledCharacterContext, MemoryItem
from evaluation.observation_input_ablation import variants
from inference.generation_request import GenerationRequest, build_generation_request


def row():
    item = MemoryItem('1', 'shared_event', '原话', evidence=('我同学在学木工。',),
        observed_at='2026-01-01T00:00:00+00:00', source_message_ids=('s',), source_observation=True)
    reference, _ = compile_reference_context((item,), observation_semantics=True)
    source = json.dumps(dict(records=[dict(source_id='s', observed_at=item.observed_at, text=item.evidence[0])]),
                        ensure_ascii=False)
    context = CompiledCharacterContext('', '', reference, memory_packets=(item,), episodic_reference_context=source)
    plan = build_generation_request(GenerationRequest(message='谁在学木工？', character_context=context))
    return dict(prepared=dict(compiled=asdict(context)), model_calls=[dict(request=dict(messages=list(plan.messages)))])


def test_only_observation_reference_changes_and_original_is_immutable():
    data = row()
    before = json.dumps(data)
    results = variants(data)
    assert json.dumps(data) == before
    assert len(results) == 3
    for messages in results.values():
        assert messages[0] == data['model_calls'][0]['request']['messages'][0]
        assert '谁在学木工？' in messages[-1]['content']
        assert '我同学在学木工。' in messages[-1]['content']


def test_no_deletion_when_independent_source_has_different_receipt():
    data = row()
    source = json.loads(data['prepared']['compiled']['episodic_reference_context'])
    source['records'][0]['observed_at'] = '2025-01-01T00:00:00+00:00'
    data['prepared']['compiled']['episodic_reference_context'] = json.dumps(source)
    with pytest.raises(ValueError, match='same complete observation'):
        variants(data)
