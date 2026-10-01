"""Read-only source admission shares one scoped SQL snapshot across adapters."""

from datetime import datetime, timezone

import pytest

from db.database import SQLiteDB
from db.memory_source import source_identity, source_scope

FIELDS = dict(character_id='role', platform='web', adapter='web-character', sender_id='alice', conversation_type='private', conversation_id='alice')
BODY = '完整合成原话：测试回执ZZ-123，不是真实用户资料。'
WHEN = datetime(2026, 1, 2, tzinfo=timezone.utc)


@pytest.fixture
def stored(tmp_path):
    database = SQLiteDB(tmp_path / 'admission.sqlite')
    assert database.capture_memory_source(**FIELDS, source_message_id='receipt', body=BODY, observed_at=WHEN) == 'recorded'
    return database


def admission(database, *, body=BODY, source_id='receipt', **fields):
    return database.memory_source_admission(**{**FIELDS, **fields}, source_message_id=source_id, body=body)


def identity_key():
    return source_identity(source_scope(*FIELDS.values()), 'receipt')['source_key']


def test_new_identity_is_allowed_after_prior_revocation(stored):
    stored._get_connection().execute("UPDATE memory_sources SET state='revoked', body=NULL, observed_at=NULL WHERE source_key=?", (identity_key(),))
    assert admission(stored, source_id='genuinely-new-receipt') == 'new'


def test_same_full_text_retry_retains_recorded_state(stored):
    assert admission(stored) == 'recorded'


def test_different_text_under_same_identity_is_conflict(stored):
    assert admission(stored, body='另一个完整合成陈述。') == 'conflict'


def test_revoked_tombstone_is_visible_as_state_without_original_body(stored):
    stored._get_connection().execute("UPDATE memory_sources SET state='revoked', body=NULL, observed_at=NULL WHERE source_key=?", (identity_key(),))
    state = admission(stored)
    assert state == 'revoked' and BODY not in state
    assert stored.list_memory_sources(**FIELDS) == []


def test_pending_anchor_keeps_existing_capture_fill_semantics(stored):
    stored._get_connection().execute("UPDATE memory_sources SET state='pending', body=NULL, observed_at=NULL WHERE source_key=?", (identity_key(),))
    assert admission(stored) == 'pending'


def test_owner_fence_marks_recorded_old_source_stale(stored):
    owner = source_scope(*FIELDS.values())['owner_key']
    stored._get_connection().execute('UPDATE memory_source_fences SET revoked_before=? WHERE owner_key=?', ('2026-01-03T00:00:00.000000+00:00', owner))
    assert admission(stored) == 'stale'


@pytest.mark.parametrize('changed', [
    {'sender_id': 'bob', 'conversation_id': 'bob'},
    {'character_id': 'other-role'},
    {'adapter': 'other-adapter'},
    {'conversation_type': 'group', 'conversation_id': 'room'},
])
def test_same_source_id_in_other_exact_scope_is_not_a_collision(stored, changed):
    assert admission(stored, body='不同范围的新完整发言。', **changed) == 'new'


def test_admission_is_one_readonly_query_with_no_text_result(stored):
    connection = stored._get_connection()
    before = connection.total_changes
    statements = []
    connection.set_trace_callback(statements.append)
    state = admission(stored)
    connection.set_trace_callback(None)
    assert state == 'recorded' and connection.total_changes == before
    assert len(statements) == 1 and statements[0].lstrip().upper().startswith('SELECT ')


def test_empty_current_text_is_not_a_valid_admission(stored):
    with pytest.raises(ValueError, match='Complete source text'):
        admission(stored, body=' ')
