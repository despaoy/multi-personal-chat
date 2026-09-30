from datetime import datetime, timedelta, timezone

import pytest

from db.database import SQLiteDB
from db.memory_source import ClaimSourceRevokedError

SCOPE = dict(character_id='role', platform='web', adapter='audit', sender_id='user',
             conversation_type='private', conversation_id='user')


def capture(db, source, body='原话', scope=None):
    return db.capture_memory_source(**(scope or SCOPE), source_message_id=source, body=body,
                                     observed_at=datetime.now(timezone.utc))


def test_source_only_erasure_preserves_other_speech_and_blocks_retry(tmp_path):
    db = SQLiteDB(tmp_path / 'sources.sqlite')
    capture(db, 'remove', '我养的狗叫松糕。')
    capture(db, 'keep', '周六去植物园散步。')
    assert db.erase_unlinked_memory_sources(**SCOPE, source_message_ids=['remove']) == 1
    assert db.erase_unlinked_memory_sources(**SCOPE, source_message_ids=['remove']) == 0
    assert [r['source_message_id'] for r in db.list_memory_sources(**SCOPE)] == ['keep']
    assert db.search_memory_sources(**SCOPE, query='松糕') == []
    assert len(db.search_memory_sources(**SCOPE, query='植物园')) == 1
    assert db.capture_memory_source(**SCOPE, source_message_id='remove', body='不得恢复',
        observed_at=datetime.now(timezone.utc) + timedelta(seconds=1)) == 'revoked'
    with pytest.raises(ClaimSourceRevokedError):
        db.append_character_memory_claim(**SCOPE, memory_type='user_fact', memory_key='pet',
                                          content='不得恢复', source_message_id='remove')


def test_source_only_erasure_does_not_cross_owner_or_character(tmp_path):
    db = SQLiteDB(tmp_path / 'scope.sqlite')
    other = SCOPE | dict(character_id='other')
    capture(db, 'shared', scope=other)
    assert db.erase_unlinked_memory_sources(**SCOPE, source_message_ids=['shared']) == 0
    assert len(db.list_memory_sources(**other)) == 1


def test_source_gaining_claim_before_delete_rolls_back_entire_batch(tmp_path):
    db = SQLiteDB(tmp_path / 'linked.sqlite')
    for source in ('unlinked', 'linked'):
        capture(db, source)
    db.append_character_memory_claim(**SCOPE, memory_type='user_fact', memory_key='key',
                                     content='现有事实', source_message_id='linked')
    with pytest.raises(ValueError, match='linked'):
        db.erase_unlinked_memory_sources(**SCOPE, source_message_ids=['unlinked', 'linked'])
    assert len(db.list_memory_sources(**SCOPE)) == 2
    assert len(db.list_character_memory_claims(**SCOPE)) == 1


@pytest.mark.parametrize('ids', ['source', [], [''], [str(i) for i in range(201)]])
def test_source_only_erasure_requires_explicit_bounded_ids(tmp_path, ids):
    db = SQLiteDB(tmp_path / 'bounds.sqlite')
    with pytest.raises(ValueError):
        db.erase_unlinked_memory_sources(**SCOPE, source_message_ids=ids)


def test_concurrent_source_delete_and_claim_have_no_half_state(tmp_path):
    from evaluation.memory_source_storage_probe import exercise_source_erasure

    assert exercise_source_erasure(SQLiteDB(tmp_path / 'race.sqlite'))['concurrent_claim_delete'] == 12


def test_source_claim_check_uses_reverse_index(tmp_path):
    db = SQLiteDB(tmp_path / 'index.sqlite')
    rows = db._get_connection().execute('EXPLAIN QUERY PLAN SELECT memory_id FROM memory_source_links '
                                       'WHERE source_key IN (?) LIMIT 1', ('key',)).fetchall()
    assert 'idx_memory_source_links_source' in ' '.join(row['detail'] for row in rows)


async def test_repository_exposes_source_only_delete_without_claim_fallback(tmp_path):
    from character.context_builder import build_user_scope
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    db = SQLiteDB(tmp_path / 'repository.sqlite')
    capture(db, 'source')
    repo = DatabaseCharacterMemoryRepository(db)
    scope = build_user_scope('web', 'audit', 'user', 'browser-session', 'private')
    assert await repo.erase_unlinked_sources('role', scope, source_message_ids=('source',)) == 1
    assert await repo.list_sources('role', scope) == []
