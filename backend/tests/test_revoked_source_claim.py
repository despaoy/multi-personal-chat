from datetime import datetime, timedelta, timezone

import pytest

from db.database import SQLiteDB
from db.memory_source import ClaimSourceRevokedError

SCOPE = dict(character_id='role', platform='test', adapter='test', sender_id='user',
             conversation_type='private', conversation_id='room')


@pytest.mark.parametrize('legacy,fresh_receipt', [(False, False), (False, True), (True, False)])
def test_revoked_identity_cannot_restore_claim_even_with_new_receipt(tmp_path, legacy, fresh_receipt):
    db = SQLiteDB(tmp_path / 'revoked.sqlite')
    db.capture_memory_source(**SCOPE, source_message_id='deleted', body='原始记录',
                             observed_at=datetime.now(timezone.utc))
    old = db.append_character_memory_claim(**SCOPE, memory_type='user_fact', memory_key='old',
                                           content='原始记录', source_message_id='deleted')
    db.erase_character_memories(**SCOPE, memory_id=old['id'])
    writer = db.add_or_update_character_memory if legacy else db.append_character_memory_claim
    extra = dict(observed_at=datetime.now(timezone.utc).isoformat()) if fresh_receipt and not legacy else {}
    with pytest.raises(ClaimSourceRevokedError):
        writer(**SCOPE, memory_type='user_fact', memory_key='restored', content='不应恢复',
               source_message_id='deleted', **extra)
    assert db.list_character_memory_claims(**SCOPE) == []
    assert db.list_memory_sources(**SCOPE) == []
    assert not db._get_connection().execute('SELECT * FROM memory_source_links').fetchall()
    # The rejection must not poison the transaction or prohibit new speech.
    assert db.capture_memory_source(**SCOPE, source_message_id='new', body='新记录',
                                    observed_at=datetime.now(timezone.utc) + timedelta(seconds=1)) == 'recorded'
    writer(**SCOPE, memory_type='user_fact', memory_key='new', content='新记录', source_message_id='new')
    assert len(db.list_character_memory_claims(**SCOPE)) == 1


def test_legacy_upsert_from_revoked_source_rolls_back_existing_value(tmp_path):
    db = SQLiteDB(tmp_path / 'upsert.sqlite')
    at = datetime.now(timezone.utc)
    db.capture_memory_source(**SCOPE, source_message_id='deleted', body='旧记录', observed_at=at)
    target = db.append_character_memory_claim(**SCOPE, memory_type='user_fact', memory_key='old',
                                              content='旧记录', source_message_id='deleted')
    db.erase_character_memories(**SCOPE, memory_id=target['id'])
    db.capture_memory_source(**SCOPE, source_message_id='new', body='有效记录',
                             observed_at=datetime.now(timezone.utc) + timedelta(seconds=1))
    kept = db.add_or_update_character_memory(**SCOPE, memory_type='user_fact', memory_key='keep',
                                             content='有效记录', source_message_id='new')
    with pytest.raises(ClaimSourceRevokedError):
        db.add_or_update_character_memory(**SCOPE, memory_type='user_fact', memory_key='keep',
                                           content='错误覆盖', source_message_id='deleted')
    rows = db.list_character_memory_claims(**SCOPE)
    assert [(r['id'], r['content'], r['source_message_id']) for r in rows] == [
        (kept['id'], '有效记录', 'new')]
