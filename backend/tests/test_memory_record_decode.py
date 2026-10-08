import pytest
from repositories.character_memory import (
    DatabaseCharacterMemoryRepository,
    _decode_memory_record,
    relationship_from_record,
)

from character.models import MemoryItem, UserScope
from db.database import SQLiteDB


@pytest.mark.parametrize('field', ['source_message_ids_json', 'evidence_json', 'metadata_json'])
@pytest.mark.parametrize('value', ['', 'private-broken-json', 'null', '42', '"text"'])
def test_malformed_json_is_not_empty_evidence(field, value):
    with pytest.raises(ValueError, match=field) as error:
        _decode_memory_record({field: value})
    assert 'private-broken-json' not in str(error.value)


@pytest.mark.parametrize(('field', 'value'), [('evidence_json', {}), ('source_message_ids_json', {}), ('metadata_json', [])])
def test_valid_json_with_wrong_shape_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        _decode_memory_record({field: value})


def test_legitimate_absence_and_legacy_single_source_are_preserved():
    row = _decode_memory_record({'source_message_id': 'original', 'source_message_ids_json': None,
                                 'evidence_json': '[]', 'metadata_json': '{}'})
    assert row['source_message_ids'] == ['original'] and row['evidence'] == [] and row['metadata'] == {}
    assert relationship_from_record(None).stage == 'stranger'
    assert relationship_from_record({'relationship_stage': 'close'}).stage == 'close'


@pytest.mark.parametrize('stage', ['', 'unknown-private-stage', None])
def test_corrupt_relationship_stage_cannot_become_stranger(stage):
    with pytest.raises(ValueError, match='relationship_stage') as error:
        relationship_from_record({'relationship_stage': stage})
    assert 'unknown-private-stage' not in str(error.value)


@pytest.mark.asyncio
async def test_stored_corruption_fails_both_scoped_list_and_exact_lookup(tmp_path):
    database = SQLiteDB(tmp_path / 'decode.sqlite')
    repo = DatabaseCharacterMemoryRepository(database)
    scope = UserScope('test', 'decode', 'reader', 'reader', 'private')
    row = await repo.append_claim('role', scope, MemoryItem('', 'user_fact', '合成事实', .9), memory_key='fact_fixture')
    conn = database._get_connection()
    conn.execute('UPDATE character_memories SET evidence_json=? WHERE id=?', ('private-broken-json', row['id']))
    conn.commit()
    with pytest.raises(ValueError, match='evidence_json'):
        await repo.list_memory_records('role', scope)
    with pytest.raises(ValueError, match='evidence_json'):
        await repo.get_memory_record(row['id'], 'role', scope)
