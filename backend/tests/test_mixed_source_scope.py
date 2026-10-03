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
@pytest.mark.parametrize('declaration', [
    '读取包含「QZ11-A01」的完整原话记录。核对编号QZ11-B01的完整原始资料。',
    '核对编号QZ11-B01的完整原始资料。读取包含「QZ11-A01」的完整原话记录。',
    '读取包含「QZ11-A01」和「登记页」的完整原话记录。核对编号QZ11-B01的完整原始资料。',
])
async def test_complete_mixed_requests_keep_both_full_dependencies(tmp_path, declaration):
    db = SQLiteDB(tmp_path / 'causal.sqlite')
    put(db, 'one', ONE)
    put(db, 'two', TWO)
    noise = '其他完整第三方原始资料，费用时限容量核验及全部前提完整提供。' + '原始资料的完整前提与例外，付款不豁免。' * 500
    for i in range(4):
        put(db, 'noise-' + str(i), noise + str(i))
    put(db, 'foreign', ONE + TWO, sender_id='bob')
    query = declaration + '给出所问条目的全部字段、完整前提和例外；未取得字段及实际办理保持未知，不能把未核验改为通过。'
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), defer_budget=True).recall('role', SCOPE, query, retrieval_context=noise)
    actual = {r['text'] for r in json.loads(result.context)['records']} if result.context else set()
    assert actual == {ONE, TWO}, result.diagnostics['status']
    assert not result.diagnostics['contextual_search_enabled']
    assert result.diagnostics['source_read_scope_kind'] == 'mixed'
    assert await DatabaseCharacterMemoryRepository(db).list_memory_records('role', SCOPE) == []


@pytest.mark.parametrize('query', [
    '读取包含「QZ11-A01」的原话。核对编号QZ11-B01。',
    '读取包含「QZ11-A01」的原话。核对编号QZ11-B01的原始资料。读取包含「未闭合',
    '读取包含「QZ11-A01」的原话。核对编号QZ11-B01的原始资料。读取包含「费用」。',
    '核对编号QZ11-B01的原始资料。读取包含「QZ11-A01」或「登记页」和「封签页」的原话。',
    '核对编号QZ11-B01的原始资料。读取包含「」的原话。',
    '核对编号QZ11-B01的原始资料。读取包含「「嵌套」」的原话。',
    '不要读取包含「QZ11-A01」的原话。核对编号QZ11-B01的原始资料。',
    '朋友说：“读取包含「QZ11-A01」的原话。核对编号QZ11-B01的原始资料。”',
    '读取包含「QZ11-A01」的原话。核对编号QZ11-B01或QZ11-C01的原始资料。',
    '读取包含「QZ11-A01」的原话。核对编号QZ11-B01的原始资料。不要核对编号QZ11-B01。',
    '读取包含「QZ11-A01」的原话。核对编号QZ11-B01的原始资料。另比较QZ11-C01的条件。',
    '读取包含「QZ11-A01」的原话。核对编号QZ11-B01的原始资料。再核对另一份完整原始记录。',
])
def test_unfinished_or_ambiguous_mixed_read_never_authorizes_a_partial_prefix(query):
    from db.memory_source_search import resolve_source_read_plan
    assert resolve_source_read_plan(query).groups == ()


@pytest.mark.parametrize('query,groups,modes', [
    ('读取包含「QZ11-A01」或「QZ11-X01」的原话。按编号QZ11-B01核对原始资料。',
     (('QZ11-A01',), ('QZ11-X01',), ('QZ11-B01',)), ('literal_substring', 'literal_substring', 'identifier_token')),
    ('核对编号QZ11-B01和QZ11-X01的原始资料。读取包含「QZ11-A01」和「登记页」的原话。',
     (('QZ11-B01',), ('QZ11-X01',), ('QZ11-A01', '登记页')), ('identifier_token', 'identifier_token', 'literal_substring', 'literal_substring')),
])
def test_mixed_unknown_union_and_same_record_conjunction_remain_distinct(query, groups, modes):
    from db.memory_source_search import resolve_source_read_plan
    plan = resolve_source_read_plan(query)
    assert plan.groups == groups and tuple(atom.match_mode for atom in plan.selectors) == modes
    assert plan.matches(ONE) and plan.matches(TWO)
    assert not plan.matches('只有登记页，完整资料未提供；无法确认其他字段或办理。')


@pytest.mark.asyncio
async def test_same_spelling_keeps_substring_and_token_receipts_and_sql_scope(tmp_path):
    from db.memory_source_search import search_plan
    db = SQLiteDB(tmp_path / 'typed.sqlite')
    prefix = ONE.replace('QZ11-A01', 'QZ11-A010')
    put(db, 'v1', ONE)
    put(db, 'prefix', prefix)
    put(db, 'foreign', ONE, sender_id='bob')
    query = '读取包含「QZ11-A01」的原话。核对编号QZ11-A01的原始资料。'
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), defer_budget=True).recall('role', SCOPE, query)
    assert {r['text'] for r in json.loads(result.context)['records']} == {ONE, prefix}
    assert result.diagnostics['literal_fragment_count'] == 2
    assert result.diagnostics['literal_read_group_match_counts'] == [2, 1]
    assert result.diagnostics['literal_fragment_match_counts'] == [2, 1]
    assert result.diagnostics['source_read_selector_modes'] == ['literal_substring', 'identifier_token']
    scope = dict(scope_key=json.dumps(('role', 'web', 'identifier-read', 'alice', 'private', 'room')), owner_key=json.dumps(('web', 'identifier-read', 'alice')))
    sql, params = next(search_plan(scope, query, limit=None, dialect='postgres'))
    assert 'POSITION(:source_fragment0 IN eligible.body)' in sql and 'eligible.body ~ :source_fragment1_pattern' in sql
    assert params['source_fragment0'] == params['source_fragment1'] == 'QZ11-A01' and 'QZ11-A01' not in sql
    db.clear_character_memories(**FIELDS)
    assert db.search_memory_sources(**FIELDS, query=query, limit=None) == []


@pytest.mark.asyncio
async def test_mixed_linked_and_fresh_reads_keep_each_member_matcher(tmp_path):
    db = SQLiteDB(tmp_path / 'fresh.sqlite')
    put(db, 'one', ONE)
    put(db, 'two', TWO)
    put(db, 'prefix', TWO.replace('QZ11-B01', 'QZ11-B010'))

    class ChangedRepository(DatabaseCharacterMemoryRepository):
        async def linked_sources(self, *args, **kwargs):
            return [r for r in db.list_memory_sources(**FIELDS, limit=100) if r['source_message_id'] == 'prefix']

        async def list_sources(self, *args, **kwargs):
            rows = await super().list_sources(*args, **kwargs)
            return [dict(r, body=r['body'].replace('QZ11-B01', 'QZ11-B01_more')) for r in rows]

    result = await SourceMemoryService(ChangedRepository(db), defer_budget=True).recall('role', SCOPE, '读取包含「QZ11-A01」的原话。核对编号QZ11-B01的原始资料。')
    assert {r['text'] for r in json.loads(result.context)['records']} == {ONE}
    assert result.diagnostics['linked_fragment_scope_omitted'] == 1
    assert result.diagnostics['fresh_recheck_omitted'] == 1
    assert result.diagnostics['literal_read_group_match_counts'] == [1, 0]


@pytest.mark.asyncio
async def test_mixed_read_retains_full_deferred_budget_packet(tmp_path):
    db = SQLiteDB(tmp_path / 'budget.sqlite')
    long = TWO + '完整第三方资料背景及全部付款不豁免前提。' * 2000 + '尾部核验条件仍不可省略，实际办理未知。'
    put(db, 'one', ONE)
    put(db, 'two', long)
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), max_chars=16384, defer_budget=True).recall('role', SCOPE, '读取包含「QZ11-A01」的原话。核对编号QZ11-B01的原始资料。')
    assert result.context == '' and result.diagnostics['status'] == 'budget_omitted'
    assert {r['text'] for r in json.loads(result.candidate_context)['records']} == {ONE, long}
