"""Multiple independently closed source reads preserve every literal root."""

import json
from datetime import datetime, timezone

import pytest

from character.models import UserScope
from character.source_memory import SourceMemoryService
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

SCOPE = UserScope('web', 'plural-read', 'alice', 'room', 'private')
FIELDS = dict(character_id='role', platform='web', adapter='plural-read', sender_id='alice', conversation_type='private', conversation_id='room')
ONE = '第三方青舟目录 QZ8-01：费用12元、时限3小时、容量12、已核验；登记页和封签页必须同时齐全，封签缺失须暂停，付款不豁免。不是本人或角色经历，实际办理未知。'
TWO = '第三方青舟目录 QZ8-02：费用13元、时限3小时、容量12、已核验；该目录的登记页和封签页缺一不可，缺封签暂停，付款不豁免。只是第三方资料，实际办理未知。'


def put(db, identity, body, **extra):
    db.capture_memory_source(**(FIELDS | extra), source_message_id=identity, body=body, observed_at=datetime.now(timezone.utc))


@pytest.mark.asyncio
@pytest.mark.parametrize('second', ['读取包含「QZ8-02」的完整原话记录。', '再查看含有“QZ8-02”的全部用户原话。', '另外检索包含『QZ8-02』的完整原始发言记录。'])
async def test_two_closed_source_reads_do_not_expand_to_history_union(tmp_path, second):
    db = SQLiteDB(tmp_path / 'two.sqlite')
    put(db, 'one', ONE)
    put(db, 'two', TWO)
    noise = '其他完整第三方目录，费用、时限、容量、核验、登记页及封签页条件全部已给全。' + '另一个目录的完整前提及付款不豁免条件。' * 400
    for i in range(4):
        put(db, 'noise-' + str(i), noise + str(i))
    put(db, 'foreign', ONE + TWO, sender_id='bob')
    query = '读取包含「QZ8-01」的完整原话记录。' + second + '核对这两份目录的全部费用、时限、容量、核验、前提和例外，保留第三方主体与未知办理结果。'
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), max_chars=16384, defer_budget=True).recall('role', SCOPE, query, retrieval_context=noise)
    assert result.diagnostics['status'] == 'available'
    assert {row['text'] for row in json.loads(result.context)['records']} == {ONE, TWO}
    assert not result.diagnostics['contextual_search_enabled']
    assert await DatabaseCharacterMemoryRepository(db).list_memory_records('role', SCOPE) == []


@pytest.mark.parametrize('query', [
    '请读取包含「QZ8-01」的完整原话记录。请再读取包含「QZ8-02」的完整原话记录。核对全部条件。',
    '查找含有“QZ8-01”的原话；同时查看含有“QZ8-02”的全部原始发言记录。比较各版本。',
    '读取包含『QZ8-01』的完整原话记录。\n另检索包含『QZ8-02』的完整原话记录。',
    '读取包含「QZ8-01」的完整原话记录。读取包含「QZ8-02」的完整原话记录。只核对保存的资料。',
])
def test_independently_closed_declarations_form_literal_union(query):
    from db.memory_source_search import literal_source_fragment, literal_source_fragments
    assert literal_source_fragments(query) == ('QZ8-01', 'QZ8-02')
    assert literal_source_fragment(query) == ''


@pytest.mark.parametrize('query', [
    '朋友说：“读取包含「QZ8-01」的完整原话记录。读取包含「QZ8-02」的完整原话记录。”',
    '不要读取包含「QZ8-01」的完整原话记录。读取包含「QZ8-02」的完整原话记录。',
    '读取包含「QZ8-01」的完整原话记录。不要读取包含「QZ8-02」的完整原话记录。',
    '读取包含「QZ8-01」的完整原话记录。再读取这段原话。',
    '读取包含「QZ8-01」和「QZ8-02」的完整原话记录。',
    '读取包含「QZ8-01」或「QZ8-02」的完整原话记录。',
    '读取包含「QZ8-01」的完整原话记录。再读取包含「QZ8-02的完整原话记录。',
    '读取包含「QZ8-01」的完整原话记录。再读取包含「嵌套“QZ8-02”」的完整原话记录。',
    '读取包含「QZ8-01」的完整原话记录。再读取包含「QZ8-02\n尾部」的完整原话记录。',
    '读取包含「QZ8-01」的完整原话记录。另外核对其他原始记录。',
])
def test_unknown_or_negated_member_cannot_authorize_partial_union(query):
    from db.memory_source_search import literal_source_fragments
    assert literal_source_fragments(query) == ()


@pytest.mark.asyncio
async def test_shared_literal_roots_keep_one_whole_record_and_both_match_receipts(tmp_path):
    db = SQLiteDB(tmp_path / 'shared.sqlite')
    combined = ONE + '\n' + TWO + '两个编号同属一条原话，不证明实际执行。'
    put(db, 'both', combined)
    query = '读取包含「QZ8-01」的完整原话记录。再读取包含「QZ8-02」的完整原话记录。核对两个编号。'
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), defer_budget=True).recall('role', SCOPE, query)
    assert [r['text'] for r in json.loads(result.context)['records']] == [combined]
    assert result.diagnostics['literal_fragment_count'] == 2
    assert result.diagnostics['literal_fragment_match_counts'] == [1, 1]


@pytest.mark.asyncio
async def test_missing_literal_member_keeps_known_original_and_unknown_gap(tmp_path):
    db = SQLiteDB(tmp_path / 'missing.sqlite')
    put(db, 'one', ONE)
    put(db, 'similar', TWO.replace('QZ8-02', 'QZ8-020'))
    query = '读取包含「QZ8-01」的完整原话记录。再读取包含「未提供的第三方目录ZQ9」的完整原话记录。比较可核对的部分。'
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), defer_budget=True).recall('role', SCOPE, query, retrieval_context=TWO)
    assert [r['text'] for r in json.loads(result.context)['records']] == [ONE]
    assert result.diagnostics['literal_fragment_match_counts'] == [1, 0]
    assert not result.diagnostics['contextual_search_enabled']
    assert await DatabaseCharacterMemoryRepository(db).list_memory_records('role', SCOPE) == []


def test_union_locators_are_exact_sql_parameters_with_owner_and_conversation_scope(tmp_path):
    db = SQLiteDB(tmp_path / 'bound.sqlite')
    locator = "ZQ9_%'--"
    exact = '第三方资料编号' + locator + '，费用10元、完整条件是登记页和封签页齐全；不证明实际办理。'
    put(db, 'one', ONE)
    put(db, 'two', exact)
    put(db, 'like-wildcard', '第三方资料ZQ9_anything，完整登记条件未核验。')
    put(db, 'other-room', exact, conversation_id='foreign-room')
    put(db, 'other-role', exact, character_id='foreign-role')
    query = '读取包含「QZ8-01」的完整原话记录。读取包含「' + locator + '」的完整原话记录。核对全部条件。'
    rows = db.search_memory_sources(**FIELDS, query=query, limit=None)
    assert {r['source_message_id'] for r in rows} == {'one', 'two'}
    assert {r['body'] for r in rows} == {ONE, exact}
    db.clear_character_memories(**FIELDS)
    assert db.search_memory_sources(**FIELDS, query=query, limit=None) == []


@pytest.mark.asyncio
async def test_fresh_revocation_cannot_resurrect_any_union_member(tmp_path):
    db = SQLiteDB(tmp_path / 'erase.sqlite')
    put(db, 'one', ONE)
    put(db, 'two', TWO)

    class RevokingRepository(DatabaseCharacterMemoryRepository):
        async def search_sources(self, *args, **kwargs):
            rows = await super().search_sources(*args, **kwargs)
            db.clear_character_memories(**FIELDS)
            return rows

    query = '查找包含「QZ8-01」的完整原话记录。再读取包含「QZ8-02」的完整原话记录。保留完整条件。'
    result = await SourceMemoryService(RevokingRepository(db), defer_budget=True).recall('role', SCOPE, query)
    assert result.context == result.candidate_context == ''
    assert result.diagnostics['fresh_recheck_omitted'] == 2
    assert result.diagnostics['literal_fragment_match_counts'] == [0, 0]


@pytest.mark.asyncio
async def test_combined_budget_never_substitutes_one_short_member(tmp_path):
    db = SQLiteDB(tmp_path / 'whole.sqlite')
    long = TWO + '完整第三方资料背景说明。' * 2000 + '以上只是小说里的目录，不是本人未来安排。'
    put(db, 'one', ONE)
    put(db, 'two', long)
    query = '读取包含「QZ8-01」的完整原话记录。再读取包含「QZ8-02」的完整原话记录。保留每条末尾的条件、否定和主体。'
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), max_chars=16384, defer_budget=True).recall('role', SCOPE, query)
    assert result.context == '' and result.diagnostics['status'] == 'budget_omitted'
    assert {r['text'] for r in json.loads(result.candidate_context)['records']} == {ONE, long}
