"""Unquoted identifiers retain complete originals without guessing data truth."""
import json
from datetime import datetime, timezone

import pytest

from character.models import UserScope
from character.source_memory import SourceMemoryService
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

SCOPE = UserScope('web', 'identifier-read', 'alice', 'room', 'private')
FIELDS = dict(character_id='role', platform='web', adapter='identifier-read', sender_id='alice', conversation_type='private', conversation_id='room')
ONE = '第三方资料编号QZ11-A01，费用12元，时限3小时，容量4，未核验。完整前提：登记页和封签页齐全且核验通过才受理；缺封签或未核验须暂停，付款不豁免。不是本人经历，实际办理未知。'
TWO = '第三方资料编号QZ11-B01，费用13元，时限4小时，容量5，已核验。完整前提：登记页和封签页齐全才受理；缺封签暂停，付款不豁免。实际办理未知。'


def put(db, identity, body, **extra):
    db.capture_memory_source(**(FIELDS | extra), source_message_id=identity, body=body, observed_at=datetime.now(timezone.utc))


@pytest.mark.asyncio
@pytest.mark.parametrize('declaration,expected,titles', [
    ('读取包含「QZ11-A01」的完整原话记录。读取《合成加速受理》的完整原始正文。', {ONE}, ('合成加速受理',)),
    ('读取《合成寄件受理》的完整原始正文。核对编号QZ11-B01的完整原始资料。', {TWO}, ('合成寄件受理',)),
    ('读取包含「QZ11-A01」的完整原话记录。读取《合成加速受理》的完整原始正文。核对编号QZ11-B01的完整原始资料。读取《合成寄件受理》的完整原始正文。', {ONE, TWO}, ('合成加速受理', '合成寄件受理')),
])
async def test_closed_cross_domain_reads_keep_raw_and_knowledge_dependencies(tmp_path, declaration, expected, titles):
    from knowledge.source_expansion import requested_document_titles
    db = SQLiteDB(tmp_path / 'causal.sqlite')
    put(db, 'one', ONE)
    put(db, 'two', TWO)
    noise = '其他完整第三方原始资料，费用时限容量核验及全部前提完整提供。' + '原始资料的完整前提与例外，付款不豁免。' * 500
    for i in range(4):
        put(db, 'noise-' + str(i), noise + str(i))
    put(db, 'foreign', ONE + TWO, sender_id='bob')
    query = declaration + '分别列出所问条目全部字段、原始前提和例外；明确区分第三方目录、公共说明和本人经历，实际办理未知，不新增或删除记忆。'
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), defer_budget=True).recall('role', SCOPE, query, retrieval_context=noise)
    actual = {r['text'] for r in json.loads(result.context)['records']} if result.context else set()
    assert actual == expected, result.diagnostics['status']
    assert not result.diagnostics['contextual_search_enabled']
    assert requested_document_titles(query) == titles
    assert await DatabaseCharacterMemoryRepository(db).list_memory_records('role', SCOPE) == []


@pytest.mark.parametrize('query', [
    '读取包含「QZ11-A01」的原话。读取《合成加速受理》的完整原始正文。不要读取《合成加速受理》。',
    '读取包含「QZ11-A01」的原话。读取《合成加速受理》的完整原始正文。仅核对《合成加速受理》。',
    '读取包含「QZ11-A01」的原话。读取《合成加速受理》的条件。',
    '读取包含「QZ11-A01」的原话。读取《合成加速受理》的完整原始正文。再读取《另一份》的条件。',
    '读取《合成加速受理》的完整原始正文。读取包含「未闭合',
    '读取《合成加速受理》的完整原始正文。核对编号QZ11-A01。',
    '读取《合成加速受理》和《合成寄件受理》的完整原始正文。读取包含「QZ11-A01」或「登记页」和「封签页」的原话。',
    '朋友说：“读取《合成加速受理》的完整原始正文。读取包含「QZ11-A01」的原话。”',
    '读取包含「QZ11-A01」的原话。读取《合成加速受理》的完整原始正文。代码示例`读取《另一份》`。',
    '读取包含「QZ11-A01」的原话。读取《合成加速受理》的完整原始正文。再读取《不完整',
])
def test_cross_domain_unfinished_negative_quoted_or_excluded_reads_keep_full_unresolved_scope(query):
    from db.memory_source_search import resolve_source_read_plan
    assert resolve_source_read_plan(query).groups == ()


@pytest.mark.parametrize('query,titles', [
    ('读取《合成加速受理》和《合成寄件受理》的完整原始正文。读取包含「QZ11-A01」的原话。', ('合成加速受理', '合成寄件受理')),
    ('读取《合成加速受理》的完整原始正文。核对编号QZ11-A01的原始资料。再读取《合成加速受理》的完整原始正文。', ('合成加速受理',)),
    ('核对编号QZ11-A01的原始资料。读取《QZ11-B01》的完整原始正文。公共标题QZ11-B01不证明本人经历。', ('QZ11-B01',)),
])
def test_document_roots_keep_their_namespace_and_stable_order(query, titles):
    from db.memory_source_search import resolve_source_read_plan
    from knowledge.source_expansion import requested_document_titles
    plan = resolve_source_read_plan(query)
    assert plan.groups == (('QZ11-A01',),) and plan.document_titles == titles
    assert requested_document_titles(query) == titles
    assert plan.matches(ONE) and not plan.matches(TWO)


def test_two_separately_closed_public_reads_keep_distinct_originals():
    from db.memory_source_search import resolve_source_read_plan
    from knowledge.source_expansion import requested_document_titles
    query = '读取《合成加速受理》的完整原始正文。读取《合成寄件受理》的完整原始正文。分别核对全部条件及例外。'
    plan = resolve_source_read_plan(query)
    assert plan.groups == () and plan.document_titles == ('合成加速受理', '合成寄件受理')
    assert requested_document_titles(query) == plan.document_titles


def cross_public_bundle(filters=None, titles=('绿泽加速受理', '绿泽寄件受理')):
    from pathlib import Path
    from threading import RLock

    from knowledge.source_expansion import expand_source_context
    from knowledge.vector_db import VectorDatabase
    case = json.loads((Path(__file__).parent / 'fixtures/deepseek_cross_kb_named_source_case.json').read_text())
    index = VectorDatabase.__new__(VectorDatabase)
    index._lock = RLock()
    index._cache_generation = 111
    index.snapshot_validated = True
    index.metadata = case['frozen_indexed_records']
    query = '读取包含「QZ11-A01」的完整原话记录。' + ''.join('读取《' + title + '》的完整原始正文。' for title in titles)
    result = expand_source_context(dict(results=[], confidence=0.0, abstained=True), index,
                                   expected_generation=111, source_budget_tokens=65536, filters=filters, query=query)
    return query, result, case


@pytest.mark.parametrize('kb', [None, 5])
def test_cross_domain_title_targets_do_not_override_the_original_knowledge_scope(kb):
    query, result, _ = cross_public_bundle(None if kb is None else {'knowledge_base_id': kb})
    assert {r['knowledge_base_id'] for r in result['results']} == ({5, 6} if kb is None else {5})
    assert result['requested_source_titles'] == ['绿泽加速受理', '绿泽寄件受理']
    assert result['unresolved_requested_titles'] == ([] if kb is None else ['绿泽寄件受理'])
    assert all(r['retrieval_role'] == 'requested_source' and r['score'] == 0 for r in result['results'])
    assert query.startswith('读取包含')


def test_missing_public_root_does_not_manufacture_an_original_or_private_claim():
    _, result, _ = cross_public_bundle(titles=('绿泽加速受理', '不存在的完整说明'))
    assert len(result['results']) == 1 and result['unresolved_requested_titles'] == ['不存在的完整说明']


def test_cross_domain_complete_private_and_public_bodies_reach_the_same_actual_request():
    from character.models import CompiledCharacterContext
    from character.source_memory import compile_sources
    from evaluation.source_transport import complete_source_in_transport
    from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
    from knowledge.evidence_packets import document_evidence_packets
    from knowledge.original_sources import attach_original_sources
    query, result, case = cross_public_bundle()
    original = {r['document_id']: dict(id=r['document_id'], title=r['title'], knowledge_base_id=r['knowledge_base_id'], category=r['category'], content=r['content']) for r in case['frozen_indexed_records']}
    result = attach_original_sources(result, original.get, source_budget_tokens=65536, authority_revision=111)
    packets = (*document_evidence_packets(result['results']), *result['original_source_packets'])
    private = compile_sources([dict(source_message_id='one', body=ONE, observed_at='2026-10-04T01:00:00+00:00')]).context
    context = CompiledCharacterContext('', '', '', episodic_reference_context=private, memory_source_status='available')
    wire = build_generation_request(GenerationRequest(message=query, context_window_tokens=65536, evidence_max_chars=0,
                                    character_context=context, retrieval=RetrievalResult(status='ok', evidence='\n'.join(p['text'] for p in packets),
                                    evidence_packets=packets, source_coverage=result['source_coverage']))).messages[-1]['content']
    assert query in wire and complete_source_in_transport(wire, ONE)
    assert all(original[r['document_id']]['content'] in wire for r in result['results'])
    assert all(r['original_source_receipt']['candidate_included'] for r in result['source_coverage'])


@pytest.mark.parametrize('tail', ['别核对原始资料。', '请别核对原始资料。'])
def test_true_negative_imperatives_after_closed_cross_domain_reads_still_defer(tail):
    from db.memory_source_search import resolve_source_read_plan
    query = '读取包含「QZ11-A01」的原话。读取《合成加速受理》的完整原始正文。' + tail
    assert resolve_source_read_plan(query).groups == ()


def test_distributive_output_check_is_not_a_negative_read():
    from db.memory_source_search import resolve_source_read_plan
    from knowledge.source_expansion import requested_document_titles
    query = '读取包含「QZ11-A01」的原话。读取《合成加速受理》的完整原始正文。分别核对全部字段、条件和例外。'
    plan = resolve_source_read_plan(query)
    assert plan.groups == (('QZ11-A01',),) and requested_document_titles(query) == ('合成加速受理',)
