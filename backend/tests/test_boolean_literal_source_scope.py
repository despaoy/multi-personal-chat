"""Same-record conjunctions cannot be satisfied by combining separate speech."""
import json
from datetime import datetime, timezone

import pytest

from character.models import UserScope
from character.source_memory import SourceMemoryService
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

SCOPE = UserScope('web', 'boolean-read', 'alice', 'room', 'private')
FIELDS = dict(character_id='role', platform='web', adapter='boolean-read', sender_id='alice', conversation_type='private', conversation_id='room')
ONE = '第三方青舟目录 QZ9-A1 及 QZ9-A2 同属这一份原话：费用12元，时限3小时，容量12，已核验；登记页和封签页齐全才受理，封签缺失暂停，付款不豁免。不是本人或角色经历，实际办理未知。'
TWO = '第三方青舟目录 QZ9-B1 及 QZ9-B2 同属另一份原话：费用13元，时限4小时，容量11，未核验；登记页和封签页齐全且核验通过才受理，核验未过须暂停，付款不豁免。实际办理未知。'


def put(db, identity, body, **extra):
    db.capture_memory_source(**(FIELDS | extra), source_message_id=identity, body=body, observed_at=datetime.now(timezone.utc))


@pytest.mark.asyncio
@pytest.mark.parametrize('locator,expected', [
    ('「QZ9-A1」和「QZ9-A2」', {ONE}),
    ('「QZ9-A1」和「QZ9-B1」', set()),
    ('「QZ9-A1」或「QZ9-B1」', {ONE, TWO}),
])
async def test_same_record_boolean_read_does_not_fall_back_to_large_history(tmp_path, locator, expected):
    db = SQLiteDB(tmp_path / 'causal.sqlite')
    put(db, 'one', ONE)
    put(db, 'two', TWO)
    noise = '其他完整第三方目录，费用、时限、容量、核验及受理前提全部给全。' + '另一条目录的完整条件与例外，付款不能豁免。' * 500
    for i in range(4):
        put(db, 'noise-' + str(i), noise + str(i))
    put(db, 'foreign', ONE + TWO, sender_id='bob')
    query = '读取包含' + locator + '的完整原话记录。核对该范围的费用、时限、容量、核验、前提与例外，保留第三方主体和未知实际办理。'
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), defer_budget=True).recall('role', SCOPE, query, retrieval_context=noise)
    actual = {r['text'] for r in json.loads(result.context)['records']} if result.context else set()
    assert actual == expected, result.diagnostics['status']
    assert not result.diagnostics['contextual_search_enabled']
    assert await DatabaseCharacterMemoryRepository(db).list_memory_records('role', SCOPE) == []


@pytest.mark.parametrize('query,expected', [
    ('读取包含「QZ9-A1」和「QZ9-A2」的完整原话记录。', (('QZ9-A1', 'QZ9-A2'),)),
    ('请查看含有“QZ9-A1”与“QZ9-A2”的全部用户原话。', (('QZ9-A1', 'QZ9-A2'),)),
    ('检索包含『QZ9-A1』及『QZ9-A2』的完整原始发言记录。', (('QZ9-A1', 'QZ9-A2'),)),
    ('查找包含「QZ9-A1」并且「QZ9-A2」的原话；', (('QZ9-A1', 'QZ9-A2'),)),
    ('读取包含「QZ9-A1」或者「QZ9-B1」的完整原话记录。', (('QZ9-A1',), ('QZ9-B1',))),
    ('读取包含「QZ9-A1」和「QZ9-A1」的原话。', (('QZ9-A1',),)),
    ('读取包含「QZ9-A1」和「QZ9-A2」的原话。再读取包含「QZ9-B1」和「QZ9-B2」的原话。核对各版本。', (('QZ9-A1', 'QZ9-A2'), ('QZ9-B1', 'QZ9-B2'))),
])
def test_closed_groups_have_explicit_same_record_semantics(query, expected):
    from db.memory_source_search import literal_source_read_groups
    assert literal_source_read_groups(query) == expected


@pytest.mark.parametrize('query', [
    '读取包含「QZ9-A1」和「QZ9-A2」或「QZ9-B1」的原话。',
    '读取包含「QZ9-A1」或「QZ9-A2」和「QZ9-B1」的原话。',
    '朋友说：“读取包含「QZ9-A1」和「QZ9-A2」的完整原话记录。”',
    '不要读取包含「QZ9-A1」和「QZ9-A2」的原话。',
    '读取包含「QZ9-A1」和「QZ9-A2」的原话。不要读取包含「QZ9-B1」的原话。',
    '读取包含「QZ9-A1」和「QZ9-A2」的原话。再读取这段原话。',
    '读取包含「QZ9-A1」和「QZ9-A2」的原话。另外核对其他记录。',
    '读取包含「QZ9-A1」和「QZ9-A2的原话。',
    '读取包含「QZ9-A1」和「嵌套“QZ9-A2”」的原话。',
    '读取包含「QZ9-A1」和「QZ9-A2\n尾部」的原话。',
    '读取包含「QZ9-A1」除了「QZ9-A2」的原话。',
])
def test_ambiguous_or_unauthorized_groups_cannot_authorize_partial_scope(query):
    from db.memory_source_search import literal_source_read_groups
    assert literal_source_read_groups(query) == ()


def test_exact_bound_boolean_sql_retains_all_versions_and_security_scope(tmp_path):
    from db.memory_source_search import search_plan
    db = SQLiteDB(tmp_path / 'sql.sqlite')
    literal = "QZ9_%'--"
    body = ONE + '\n精确额外目录编号：' + literal
    put(db, 'v1', body)
    put(db, 'v2', body + '这是后来再次提供的版本，不证明当前办理。')
    put(db, 'partial', ONE)
    put(db, 'wildcard', ONE + '编号QZ9_anything。')
    put(db, 'other-role', body, character_id='foreign')
    put(db, 'other-room', body, conversation_id='foreign')
    put(db, 'other-user', body, sender_id='bob')
    query = '读取包含「QZ9-A1」和「' + literal + '」的完整原话记录。'
    rows = db.search_memory_sources(**FIELDS, query=query, limit=None)
    assert {r['source_message_id'] for r in rows} == {'v1', 'v2'}
    scope = dict(scope_key=json.dumps(('role', 'web', 'boolean-read', 'alice', 'private', 'room')), owner_key=json.dumps(('web', 'boolean-read', 'alice')))
    pgsql, params = next(search_plan(scope, query, limit=None, dialect='postgres'))
    assert 'POSITION(:source_fragment0 IN eligible.body) > 0 AND POSITION(:source_fragment1 IN eligible.body) > 0' in pgsql
    assert params['source_fragment1'] == literal and literal not in pgsql
    db.clear_character_memories(**FIELDS)
    assert db.search_memory_sources(**FIELDS, query=query, limit=None) == []


@pytest.mark.asyncio
async def test_linked_partial_record_cannot_satisfy_conjunction(tmp_path):
    db = SQLiteDB(tmp_path / 'linked.sqlite')
    put(db, 'one', ONE)
    put(db, 'partial', TWO)

    class LinkedPartialRepository(DatabaseCharacterMemoryRepository):
        async def linked_sources(self, *args, **kwargs):
            return [r for r in db.list_memory_sources(**FIELDS, limit=100) if r['source_message_id'] == 'partial']

    query = '读取包含「QZ9-A1」和「QZ9-A2」的完整原话记录。'
    result = await SourceMemoryService(LinkedPartialRepository(db), defer_budget=True).recall('role', SCOPE, query)
    assert [r['text'] for r in json.loads(result.context)['records']] == [ONE]
    assert result.diagnostics['linked_fragment_scope_omitted'] == 1
    assert result.diagnostics['literal_read_group_sizes'] == [2]
    assert result.diagnostics['literal_read_group_match_counts'] == [1]


@pytest.mark.asyncio
async def test_fresh_record_must_still_satisfy_every_conjunct(tmp_path):
    db = SQLiteDB(tmp_path / 'fresh.sqlite')
    put(db, 'one', ONE)

    class ChangedRepository(DatabaseCharacterMemoryRepository):
        async def list_sources(self, *args, **kwargs):
            rows = await super().list_sources(*args, **kwargs)
            return [dict(r, body=r['body'].replace('QZ9-A2', 'removed')) for r in rows]

    query = '读取包含「QZ9-A1」和「QZ9-A2」的完整原话记录。'
    result = await SourceMemoryService(ChangedRepository(db), defer_budget=True).recall('role', SCOPE, query)
    assert result.context == result.candidate_context == ''
    assert result.diagnostics['fresh_recheck_omitted'] == 1
    assert result.diagnostics['literal_read_group_match_counts'] == [0]


@pytest.mark.asyncio
async def test_or_read_keeps_whole_long_original_in_deferred_packet(tmp_path):
    db = SQLiteDB(tmp_path / 'budget.sqlite')
    long = TWO + '完整第三方资料背景。' * 2400 + '尾部条件：这只是第三方目录，不能证明本人已经实际办理。'
    put(db, 'one', ONE)
    put(db, 'two', long)
    query = '读取包含「QZ9-A1」或「QZ9-B1」的完整原话记录。'
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), max_chars=16384, defer_budget=True).recall('role', SCOPE, query)
    assert result.context == '' and result.diagnostics['status'] == 'budget_omitted'
    assert {r['text'] for r in json.loads(result.candidate_context)['records']} == {ONE, long}
