from dataclasses import replace
from functools import partial

import pytest

from character.context_builder import compile_reference_context
from character.models import CompiledCharacterContext, MemoryItem
from character.source_memory import SourceRecall, compile_sources
from character.source_memory import attach_sources as default_attach_sources
from inference.generation_request import GenerationRequest, build_generation_request

STAMP = '2026-01-01T00:00:00+00:00'
TEXT = '我同事参加陶艺课，我自己没有参加。'
attach_sources = partial(default_attach_sources, share_observations=True)


def context(*, complete=False, address='', **kwargs):
    item = MemoryItem('1', 'shared_event', TEXT, evidence=(TEXT,), source_message_ids=('s',),
        observed_at=STAMP, temporal_mode='observation', source_observation=True)
    item = replace(item, **kwargs)
    reference, ids = compile_reference_context((item,), preferred_address=address, complete_evidence=complete)
    return CompiledCharacterContext('', '', reference, used_memory_ids=ids, memory_packets=(item,))


def sources(**kwargs):
    row = dict(source_message_id='s', observed_at=STAMP, body=TEXT)
    row.update(kwargs)
    return compile_sources([row])


def test_runtime_default_keeps_reference_and_only_attaches_original_sources():
    before = context()
    after = default_attach_sources(before, sources())
    assert after.reference_context == before.reference_context
    assert after.used_memory_ids == before.used_memory_ids
    assert after.episodic_reference_context == sources().context
    assert not after.source_shared_memory_ids and not after.source_reference_backup


def test_disabling_candidate_restores_its_previous_reference():
    before = context()
    after = default_attach_sources(attach_sources(before, sources()), sources())
    assert after.reference_context == before.reference_context
    assert not after.source_shared_memory_ids


@pytest.mark.parametrize('complete', [False, True])
@pytest.mark.parametrize('address', ['', '小舟'])
def test_exact_observation_is_shared_without_losing_ids_or_address(complete, address):
    before = context(complete=complete, address=address)
    after = attach_sources(before, sources(), preferred_address=address, complete_evidence=complete)
    assert TEXT not in after.reference_context and TEXT in after.episodic_reference_context
    assert after.used_memory_ids == before.used_memory_ids == ('1',)
    assert after.memory_packets == before.memory_packets
    assert after.source_shared_memory_ids == ('1',)
    assert after.source_reference_backup == before.reference_context
    assert bool(after.reference_context) == bool(address)
    assert attach_sources(after, sources(), preferred_address=address, complete_evidence=complete) == after


@pytest.mark.parametrize('change', [dict(body=TEXT+'补充'), dict(source_message_id='other'),
                                  dict(observed_at='2025-01-01T00:00:00+00:00')])
def test_same_words_or_receipt_alone_are_not_enough(change):
    before = context()
    after = attach_sources(before, sources(**change))
    assert after.reference_context == before.reference_context
    assert after.source_shared_memory_ids == ()


@pytest.mark.parametrize('change', [dict(source_observation=False), dict(temporal_mode='fact'),
    dict(historical=True), dict(qualifiers=(('condition', '周末'),)), dict(relation_type='COEXIST'),
    dict(evidence=(TEXT, '其他证据')), dict(source_message_ids=('s', 'other'))])
def test_distinct_claim_semantics_are_not_erased(change):
    before = context(**change)
    assert attach_sources(before, sources()).reference_context == before.reference_context


def test_changed_source_packet_restores_reference_before_reattaching():
    before = context()
    attached = attach_sources(before, sources())
    other = attach_sources(attached, sources(source_message_id='new'))
    assert other.reference_context == before.reference_context
    assert other.source_shared_memory_ids == ()
    empty = attach_sources(attached, SourceRecall())
    assert empty.reference_context == before.reference_context and not empty.episodic_reference_context


def test_extra_reference_or_omitted_source_never_drops_observation():
    before = replace(context(), reference_context=context().reference_context+'\n其他资料')
    assert attach_sources(before, sources()).reference_context == before.reference_context
    before = context()
    assert attach_sources(before, SourceRecall()) == before


def test_shared_text_reaches_model_once_and_backup_never_enters_system():
    attached = attach_sources(context(), sources())
    plan = build_generation_request(GenerationRequest(message='谁参加了陶艺课？', character_context=attached))
    assert sum(message['content'].count(TEXT) for message in plan.messages) == 1
    assert all(TEXT not in message['content'] for message in plan.messages if message['role'] == 'system')
    assert attached.source_reference_backup not in str(plan.messages)


def test_source_sharing_preserves_system_policy_without_promoting_source_only():
    before = context()
    after = attach_sources(before, sources())
    original = build_generation_request(GenerationRequest(message='谁参加了陶艺课？', character_context=before))
    shared = build_generation_request(GenerationRequest(message='谁参加了陶艺课？', character_context=after))
    assert original.messages[0] == shared.messages[0]
    source_only = attach_sources(CompiledCharacterContext('', '', ''), sources())
    unadmitted = build_generation_request(GenerationRequest(message='谁参加了陶艺课？', character_context=source_only))
    assert unadmitted.messages[0] != shared.messages[0]


def test_mixed_fact_and_observation_preserves_fact_content_and_usage():
    before = context()
    fact = MemoryItem('2', 'user_fact', '用户喜欢红茶', evidence=('我喜欢红茶。',))
    packets = (*before.memory_packets, fact)
    reference, ids = compile_reference_context(packets)
    before = replace(before, reference_context=reference, used_memory_ids=ids, memory_packets=packets)
    after = attach_sources(before, sources())
    assert '用户喜欢红茶' in after.reference_context and TEXT not in after.reference_context
    assert after.used_memory_ids == ('1', '2') and after.memory_packets == packets
