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
@pytest.mark.parametrize('declaration,expected', [
    ('请按编号QZ11-A01核对完整原始资料。', {ONE}),
    ('请分别核对编号QZ11-A01和QZ11-B01的完整原始数据。', {ONE, TWO}),
    ('核对编号QZ11-A01及QZ11-X01的完整原始资料。', {ONE}),
])
async def test_complete_unquoted_identifier_requests_do_not_expand_to_history(tmp_path, declaration, expected):
    db = SQLiteDB(tmp_path / 'causal.sqlite')
    put(db, 'one', ONE)
    put(db, 'two', TWO)
    put(db, 'near-miss', TWO.replace('QZ11-B01', 'QZ11-X010'))
    put(db, 'prefix-collision', ONE.replace('QZ11-A01', 'QZ11-A010'))
    noise = '其他完整第三方原始资料，费用时限容量核验及全部前提完整提供。' + '原始资料的完整前提与例外，付款不豁免。' * 500
    for i in range(4):
        put(db, 'noise-' + str(i), noise + str(i))
    put(db, 'foreign', ONE + TWO, sender_id='bob')
    query = declaration + '给出所问条目的全部字段、完整前提和例外；未取得字段及实际办理保持未知，不能把未核验改为通过。'
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), defer_budget=True).recall('role', SCOPE, query, retrieval_context=noise)
    actual = {r['text'] for r in json.loads(result.context)['records']} if result.context else set()
    assert actual == expected, result.diagnostics['status']
    assert not result.diagnostics['contextual_search_enabled']
    assert await DatabaseCharacterMemoryRepository(db).list_memory_records('role', SCOPE) == []


@pytest.mark.parametrize('query', [
    '核对编号QZ11-A01和QZ11-B01的完整原始资料。',
    '请逐项查看资料编号：QZ11-A01、QZ11-B01的全部原始数据。',
    '按条目编号 QZ11-A01，QZ11-B01 分别核对原始记录。',
    '读取编号QZ11-A01的原话。再核查编号QZ11-B01的原始发言。',
    '核对编号QZ11-A01的原始资料。读取完整原话仅用于核对依据，同时保留第三方主体。再说明已知费用。',
])
def test_complete_id_dependencies_use_independent_token_roots(query):
    from db.memory_source_search import resolve_source_read_plan
    plan = resolve_source_read_plan(query)
    expected = ('QZ11-A01',) if 'QZ11-B01' not in query else ('QZ11-A01', 'QZ11-B01')
    assert plan.groups == tuple((value,) for value in expected)
    assert plan.match_mode == 'identifier_token'


@pytest.mark.parametrize('query', [
    '朋友说：“核对编号QZ11-A01的原始资料。”',
    '不要核对编号QZ11-A01的完整原始资料。',
    '核对编号QZ11-A01及无编号资料的原始数据。',
    '核对编号QZ11-A01或QZ11-B01的原始资料。',
    '核对编号QZ11-A01的原始资料。不要核对编号QZ11-A01。',
    '核对编号QZ11-A01的原始资料。另比较QZ11-B01的条件。',
    '核对编号QZ11-A01的原始资料。再核对其他完整原始记录。',
    '核对编号QZ11-A01的原始资料。核对另一份目录。',
    '核对编号QZ11-A01的原始资料。读取包含「QZ11-B01」的原话。',
    '读取包含「QZ11-A01」的原话。核对编号QZ11-B01的原始资料。',
    '核对编号Alpha和Beta的原始资料。',
    '核对编号QZ11-A01的原始资料。朋友说：“未闭合',
])
def test_incomplete_quoted_negative_or_mixed_dependencies_do_not_authorize_a_partial_plan(query):
    from db.memory_source_search import resolve_source_read_plan
    assert resolve_source_read_plan(query).groups == ()


def test_every_whole_token_occurrence_and_version_with_same_sql_scope(tmp_path):
    from db.memory_source_search import search_plan
    from knowledge.source_read_plan import identifier_match
    db = SQLiteDB(tmp_path / 'token.sqlite')
    repeated = '先出现QZ11-A010不是目标；之后提供完整目标：' + ONE
    put(db, 'v1', repeated)
    put(db, 'v2', ONE + '再次提供同一编号的完整版本，不证明已经执行。')
    put(db, 'prefix', ONE.replace('QZ11-A01', 'QZ11-A010'))
    put(db, 'suffix', ONE.replace('QZ11-A01', 'QZ11-A01_more'))
    put(db, 'case', ONE.replace('QZ11-A01', 'qz11-a01'))
    put(db, 'foreign-role', ONE, character_id='foreign')
    put(db, 'foreign-room', ONE, conversation_id='foreign')
    put(db, 'foreign-user', ONE, sender_id='bob')
    query = '核对编号QZ11-A01的完整原始资料。'
    assert {r['source_message_id'] for r in db.search_memory_sources(**FIELDS, query=query, limit=None)} == {'v1', 'v2'}
    scope = dict(scope_key=json.dumps(('role', 'web', 'identifier-read', 'alice', 'private', 'room')), owner_key=json.dumps(('web', 'identifier-read', 'alice')))
    sql, params = next(search_plan(scope, query, limit=None, dialect='postgres'))
    assert 'eligible.body ~ :source_fragment0_pattern' in sql and 'QZ11-A01' not in sql
    assert params['source_fragment0'] == 'QZ11-A01' and identifier_match('QZ11-A01', repeated)
    assert not identifier_match('QZ11-A01', 'QZ11-A010 QZ11-A01_more qz11-a01')
    db.clear_character_memories(**FIELDS)
    assert db.search_memory_sources(**FIELDS, query=query, limit=None) == []


@pytest.mark.asyncio
async def test_linked_prefix_and_fresh_suffix_are_not_identifier_evidence(tmp_path):
    db = SQLiteDB(tmp_path / 'fresh.sqlite')
    put(db, 'one', ONE)
    put(db, 'prefix', ONE.replace('QZ11-A01', 'QZ11-A010'))

    class ChangedRepository(DatabaseCharacterMemoryRepository):
        async def linked_sources(self, *args, **kwargs):
            return [r for r in db.list_memory_sources(**FIELDS, limit=100) if r['source_message_id'] == 'prefix']

        async def list_sources(self, *args, **kwargs):
            rows = await super().list_sources(*args, **kwargs)
            return [dict(r, body=r['body'].replace('QZ11-A01', 'QZ11-A01_more')) for r in rows]

    result = await SourceMemoryService(ChangedRepository(db), defer_budget=True).recall('role', SCOPE, '核对编号QZ11-A01的完整原始资料。')
    assert result.context == result.candidate_context == ''
    assert result.diagnostics['linked_fragment_scope_omitted'] == 1
    assert result.diagnostics['fresh_recheck_omitted'] == 1
    assert result.diagnostics['literal_fragment_match_counts'] == [0]


@pytest.mark.asyncio
async def test_id_roots_keep_full_budget_packet_and_quoted_substring_contract(tmp_path):
    db = SQLiteDB(tmp_path / 'budget.sqlite')
    long = TWO + '完整第三方资料背景及全部付款不豁免前提。' * 2000 + '尾部核验条件仍不可省略，实际办理未知。'
    put(db, 'one', ONE)
    put(db, 'two', long)
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), max_chars=16384, defer_budget=True).recall('role', SCOPE, '核对编号QZ11-A01和QZ11-B01的完整原始资料。')
    assert result.context == '' and result.diagnostics['status'] == 'budget_omitted'
    assert {r['text'] for r in json.loads(result.candidate_context)['records']} == {ONE, long}
    put(db, 'quoted-prefix', ONE.replace('QZ11-A01', 'QZ11-A010'))
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), defer_budget=True).recall('role', SCOPE, '读取包含「QZ11-A01」的完整原话记录。')
    assert result.diagnostics['source_read_scope_kind'] == 'literal_substring'
    assert {r['text'] for r in json.loads(result.context)['records']} == {ONE, ONE.replace('QZ11-A01', 'QZ11-A010')}
