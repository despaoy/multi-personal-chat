"""Mechanism-level transfer checks, independent of persona and dialogue IDs."""
import pytest

from character.context_builder import MAX_MEMORY_TOTAL_CHARS, compile_reference_context
from character.memory_extractor import extract_memories
from character.models import MemoryItem


@pytest.mark.parametrize(('text', 'expected'), [
    ('我来自泉州，现在住在合肥。', {'user_origin': '泉州', 'user_residence': '合肥'}),
    ('我叫林溪，来自洛阳，现在住在贵阳。', {'user_name': '林溪', 'user_origin': '洛阳', 'user_residence': '贵阳'}),
    ('我喜欢素描，不喜欢攀岩。', {'preference_素描': '喜欢素描', 'preference_攀岩': '不喜欢攀岩'}),
    ('我学的是物理，现在在研究所工作。', {'user_major': '物理', 'user_workplace': '研究所'}),
])
def test_coordinated_self_claims_preserve_original_evidence(text, expected):
    memories = {item.memory_key: item for item in extract_memories(text)}
    for key, value in expected.items():
        assert key in memories
        assert value in memories[key].content
        assert memories[key].evidence in text
        assert value.removeprefix('不喜欢').removeprefix('喜欢') in memories[key].evidence


@pytest.mark.parametrize('text', [
    '我喜欢素描，小周不喜欢攀岩。',
    '我喜欢素描，小周来了，不喜欢攀岩。',
    '我喜欢素描。不喜欢攀岩。',
    '如果我喜欢素描，不喜欢攀岩。',
    '我喜欢素描，不喜欢攀岩吗？',
    '我叫你别走，不喜欢攀岩。',
])
def test_subject_inheritance_does_not_cross_unknown_subject_or_assertion_boundary(text):
    assert 'preference_攀岩' not in {item.memory_key for item in extract_memories(text)}


def test_omitted_subject_correction_keeps_latest_clause():
    memories = {item.memory_key: item for item in extract_memories('我喜欢徒步，不喜欢徒步。')}
    assert memories['preference_徒步'].polarity == 'dislike'


@pytest.mark.parametrize(('text', 'key', 'unrelated'), [
    ('我喜欢陶艺，小林来了，不喜欢游泳。', 'preference_陶艺', '游泳'),
    ('我来自宁波，小陈住在昆明。', 'user_origin', '昆明'),
    ('我学的是历史，我目前在档案馆工作。', 'user_workplace', '历史'),
    ('我喜欢弓箭，不喜欢射击。', 'preference_弓箭', '射击'),
])
def test_claim_evidence_excludes_unrelated_adjacent_predicates(text, key, unrelated):
    item = next(item for item in extract_memories(text) if item.memory_key == key)
    assert item.evidence in text
    assert unrelated not in item.evidence


@pytest.mark.parametrize(('text', 'key', 'value'), [
    ('我学的是历史，目前在档案馆工作。', 'user_workplace', '档案馆'),
    ('我来自湖州，现在住在保定。', 'user_residence', '保定'),
    ('我喜欢皮划艇，不喜欢滑冰。', 'preference_滑冰', '滑冰'),
])
def test_inherited_owner_evidence_preserves_original_subject(text, key, value):
    # Evidence is source provenance, not the normalized claim: retaining the
    # owner-bearing clause must not promote that clause into this claim's value.
    item = next(item for item in extract_memories(text) if item.memory_key == key)
    assert item.evidence == text
    assert value in item.content
    reparsed = extract_memories(item.evidence)
    assert any(other.memory_key == key and other.content == item.content for other in reparsed)


def test_scoped_evidence_preserves_attached_condition():
    text = '我喜欢徒步，但下雨不参加。'
    item = extract_memories(text)[0]
    assert '下雨不参加' in item.evidence
    assert '下雨不参加' in item.content


def test_lightweight_evidence_retains_late_qualification_and_all_sources():
    source = '补充背景' * 35 + '但是雨天不参加，且仅限室内。'
    item = MemoryItem('a', 'user_fact', '用户喜欢参加活动', evidence=(source, '依据二', '依据三'),
                      source_message_ids=('source-1', 'source-2', 'source-3'))
    context, ids = compile_reference_context((item,))
    assert ids == ('a',)
    assert source in context
    assert '依据三' in context and 'source-3' in context


def test_oversize_packet_is_skipped_not_partially_injected_and_next_fits():
    large = MemoryItem('large', 'user_fact', '背景' * MAX_MEMORY_TOTAL_CHARS + '仅限周末')
    small = MemoryItem('small', 'user_fact', '用户来自北方')
    diagnostics = {}
    context, ids = compile_reference_context((large, small), diagnostics=diagnostics)
    assert ids == ('small',)
    assert '背景' not in context
    assert '用户来自北方' in context
    assert diagnostics['budget_skipped'] == 1
    assert diagnostics['selected_count'] == 1
    assert diagnostics['input_count'] == 2


@pytest.mark.asyncio
async def test_default_recall_retains_late_evidence():
    from character.context_builder import build_user_scope
    from character.memory_service import CharacterMemoryService

    evidence = ['依据一', '依据二', '依据三', '依据四', '最后补充：下雨不参加']

    class Repo:
        async def list_memory_records(self, *args, **kwargs):
            return [dict(id='a', memory_type='user_fact', memory_key='preference_徒步',
                         content='用户喜欢徒步', evidence=evidence)]

    scope = build_user_scope(platform='web', adapter='test', sender_id='u',
                             conversation_id='c', conversation_type='private')
    service = CharacterMemoryService(Repo(), semantic_enabled=False)
    items, _ = await service.load_relevant_memories('role', scope, '我喜欢什么？')
    assert len(items) == 1
    assert items[0].evidence == tuple(evidence)
    context, _ = compile_reference_context(items)
    assert evidence[-1] in context


def test_missing_latency_is_not_zero_and_measurements_keep_sample_count():
    from evaluation.dialogue_audit_report import latency_summary
    assert latency_summary([{}], 'seconds')['p50_seconds'] is None
    result = latency_summary([{}, {'seconds': 1}, {'seconds': 2}, {'seconds': 3}], 'seconds')
    assert result == dict(samples=3, p50_seconds=2., p95_seconds=3.)


def test_model_latency_residual_is_not_reported_as_retrieval_measurement():
    from evaluation.dialogue_audit_report import summarize

    result = summarize([
        {'case_id': 'a', 'seconds': 10, 'model_calls': [{'seconds': 3}, {'seconds': 2}]},
        {'case_id': 'b', 'seconds': 9},
    ])
    assert result['generation_latency']['model_seconds'] == {
        'samples': 1, 'p50_seconds': 5., 'p95_seconds': 5.}
    assert result['generation_latency']['non_model_seconds']['p50_seconds'] == 5.
    assert result['memory_recall_observations'] == 0
    assert result['memory_recall_status'] == {}


def test_confirmed_no_model_response_is_zero_not_missing_measurement():
    from evaluation.dialogue_audit_report import summarize

    result = summarize([
        {'case_id': 'source', 'seconds': .2, 'model_calls': [],
         'generation': {'model_invoked': False, 'response_mode': 'source_excerpt'}},
        {'case_id': 'unknown', 'seconds': 1, 'model_calls': []},
    ])
    assert result['generation_latency']['model_seconds'] == {
        'samples': 1, 'p50_seconds': 0., 'p95_seconds': 0.}
    assert result['generation_latency']['non_model_seconds']['p50_seconds'] == .2
    assert result['response_modes'] == {'source_excerpt': 1, 'unobserved': 1}
