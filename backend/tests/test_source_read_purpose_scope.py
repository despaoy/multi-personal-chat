"""A closed evidence read remains complete despite nominal output explanation."""
import json
from datetime import datetime, timezone

import pytest

from character.models import UserScope
from character.source_memory import SourceMemoryService
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

SCOPE = UserScope('web', 'read-purpose', 'alice', 'room', 'private')
FIELDS = dict(character_id='role', platform='web', adapter='read-purpose', sender_id='alice', conversation_type='private', conversation_id='room')
ONE = '第三方青舟目录 QZ10-A1 及 QZ10-A2，同一份原话：费用12元，时限3小时，容量12，已核验；登记页和封签页齐全才受理，缺封签暂停，付款不豁免。只是第三方资料，实际办理未知。'
TWO = '第三方青舟目录 QZ10-B1，另一份原话：费用13元，时限4小时，容量11，未核验；登记页、封签页齐全且核验通过才受理，核验未过暂停，付款不豁免。实际办理未知。'


def put(db, identity, body, **extra):
    db.capture_memory_source(**(FIELDS | extra), source_message_id=identity, body=body, observed_at=datetime.now(timezone.utc))


@pytest.mark.asyncio
@pytest.mark.parametrize('locator,tail,expected', [
    ('「QZ10-A1」和「QZ10-A2」', '读取完整原话仅用于取得核对依据，回复只给指定条目的全部字段、完整前提和例外。', {ONE}),
    ('「QZ10-A1」或「QZ10-B1」', '同时要求保留第三方主体和未知实际办理，并且逐项判断容量与核验条件。', {ONE, TWO}),
    ('「QZ10-A1」或「未提供的目录QZ10-C1」', '查看上述原话只是为了核对完整条件；另外比较上述资料的已知字段，缺失内容保持未知。', {ONE}),
])
async def test_nominal_read_purpose_or_output_condition_does_not_expand_source_scope(tmp_path, locator, tail, expected):
    db = SQLiteDB(tmp_path / 'causal.sqlite')
    put(db, 'one', ONE)
    put(db, 'two', TWO)
    noise = '其他完整第三方目录的费用时限容量核验和全部前提已给全。' + '第三方资料背景与完整付款不豁免条件。' * 600
    for i in range(4):
        put(db, 'noise-' + str(i), noise + str(i))
    put(db, 'foreign', ONE + TWO, sender_id='bob')
    query = '读取包含' + locator + '的完整原话记录。' + tail
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), defer_budget=True).recall('role', SCOPE, query, retrieval_context=noise)
    actual = {r['text'] for r in json.loads(result.context)['records']} if result.context else set()
    assert actual == expected, result.diagnostics['status']
    assert not result.diagnostics['contextual_search_enabled']
    assert await DatabaseCharacterMemoryRepository(db).list_memory_records('role', SCOPE) == []


@pytest.mark.parametrize('tail', [
    '读取原话只用于核对依据，回复只包含指定字段。',
    '读取这些完整用户原话只是为了核对前提。',
    '查看前述原始发言记录仅仅作为核对依据。',
    '复述该原话只作为解释依据，回复不输出其他条目。',
    '同时判断容量是否满足条件，并且保留未核验状态。',
    '另外比较上述资料的费用和时限，实际办理未知。',
    '同时核对这两份目录的原始字段并比较费用。',
    '朋友说：“另外读取包含「其他编号QZ10-C1」的原话。”读取上述原话仅用于核对依据。',
])
def test_closed_scope_allows_nominal_evidence_reference_and_output_constraints(tail):
    from db.memory_source_search import literal_source_read_groups
    query = '读取包含「QZ10-A1」或「QZ10-B1」的完整原话记录。' + tail
    assert literal_source_read_groups(query) == (('QZ10-A1',), ('QZ10-B1',))


@pytest.mark.parametrize('tail', [
    '另外读取包含「QZ10-C1」的原话，仅用于核对依据。',
    '读取包含「QZ10-C1」的原话仅用于核对依据。',
    '读取其他完整原话仅用于核对依据。',
    '不要读取上述原话仅用于核对依据。',
    '再读取上述原话仅用于核对依据。',
    '另外核对其他原始记录。',
    '同时检查另一份目录。',
    '此外核对不同版本的原话。',
    '还需提供其他完整资料。',
    '另外比较这两份目录与另一条原话的条件。',
    '另外核对指定的其他原话。',
    '读取完整原话仅用于核对依据；再检索其他原始记录。',
    '读取完整原话仅用于核对依据。另外核对其他资料。',
    '读取完整原话仅用于核对依据。再读取这段原话。',
])
def test_purpose_explanation_never_waives_a_real_additional_or_negated_read(tail):
    from db.memory_source_search import literal_source_read_groups
    query = '读取包含「QZ10-A1」的完整原话记录。核对全部条件。' + tail
    assert literal_source_read_groups(query) == ()


@pytest.mark.asyncio
async def test_complete_packet_retains_long_tail_and_fresh_erasure_with_purpose(tmp_path):
    db = SQLiteDB(tmp_path / 'whole.sqlite')
    long = TWO + '完整第三方背景资料，不证明实际办理。' * 2000 + '尾部未核验条件：不得先付款后省略登记页与封签页核验。'
    put(db, 'one', ONE)
    put(db, 'two', long)
    query = '读取包含「QZ10-A1」或「QZ10-B1」的原话。读取完整原话仅用于核对完整依据，同时要求保留所有前提。'
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), max_chars=16384, defer_budget=True).recall('role', SCOPE, query)
    assert result.context == '' and result.diagnostics['status'] == 'budget_omitted'
    assert {r['text'] for r in json.loads(result.candidate_context)['records']} == {ONE, long}

    class RevokingRepository(DatabaseCharacterMemoryRepository):
        async def search_sources(self, *args, **kwargs):
            rows = await super().search_sources(*args, **kwargs)
            db.clear_character_memories(**FIELDS)
            return rows

    result = await SourceMemoryService(RevokingRepository(db), defer_budget=True).recall('role', SCOPE, query)
    assert result.context == result.candidate_context == ''
    assert result.diagnostics['fresh_recheck_omitted'] == 2
    assert result.diagnostics['literal_read_group_match_counts'] == [0, 0]


@pytest.mark.asyncio
async def test_already_read_speech_as_evidence_is_not_an_additional_read(tmp_path):
    db = SQLiteDB(tmp_path / 'passive.sqlite')
    put(db, 'one', ONE)
    noise = '其他完整第三方目录的全部条件已给全，费用时限及付款不豁免。' + '原始发言的完整条件与未知实际办理。' * 3000
    put(db, 'noise', noise)
    query = '读取包含「QZ10-A1」的完整原话记录。回复只给指定条目，完整读取的原话作为核对依据，同时保留所有前提。'
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), defer_budget=True).recall('role', SCOPE, query, retrieval_context=noise)
    assert result.context, result.diagnostics['status']
    assert [r['text'] for r in json.loads(result.context)['records']] == [ONE]
    assert not result.diagnostics['contextual_search_enabled']


@pytest.mark.parametrize('tail', [
    '另外查看的原话作为核对依据。',
    '其他已经读取的原话作为核对依据。',
    '不要复述的原话作为核对依据。',
])
def test_nominal_past_read_cannot_waive_extra_or_negative_scope(tail):
    from db.memory_source_search import literal_source_read_groups
    assert literal_source_read_groups('读取包含「QZ10-A1」的原话。' + tail) == ()
