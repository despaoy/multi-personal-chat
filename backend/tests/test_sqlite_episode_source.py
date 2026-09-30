from dataclasses import replace

import pytest

from character.context_builder import build_user_scope
from db.database import SQLiteDB
from db.integration_receipts import CREATE_SQL
from evaluation.episodic_recall_audit import select_evidence
from evaluation.sqlite_episode_source import read_scoped_episodes


def scope(**kw):
    return build_user_scope(**dict(dict(platform='web', adapter='audit', sender_id='alice',
        conversation_type='private', conversation_id='room'), **kw))


def add(db, message='周六修书架。', **kw):
    return db.add_message(dict(dict(sessionId='room', sessionType='private', platform='web',
        adapter='audit', senderId='alice', characterId='role', message=message, reply='收到。',
        createdAt='2026-09-26T10:00:00'), **kw))


@pytest.fixture
def db(tmp_path):
    return SQLiteDB(tmp_path / 'source.db')


@pytest.mark.parametrize('change', [dict(senderId='bob'), dict(characterId='other'), dict(characterId=None),
    dict(adapter='other'), dict(platform='other'), dict(sessionType='group'), dict(branchId='branch')])
def test_scope_before_limit_and_no_legacy_character(db, change):
    add(db)
    for _ in range(3):
        add(db, **change)
    result = read_scoped_episodes(db, scope(), 'role', max_rows=1)
    assert len(result.episodes) == 1 and not result.older_rows_omitted


def test_group_and_channel_and_private_are_separate(db):
    add(db, '本群', sessionType='group')
    add(db, '别群', sessionType='group', sessionId='other')
    add(db, '同号频道', sessionType='channel')
    result = read_scoped_episodes(db, scope(conversation_type='group'), 'role')
    assert [r.message for r in result.episodes] == ['本群']


def test_failed_delivery_is_not_read(db):
    add(db, traceId='failed-owner')
    conn = db._get_connection()
    conn.execute(CREATE_SQL)
    conn.execute("INSERT INTO integration_receipts VALUES ('r','failed-owner','delivery_failed','',0)")
    conn.commit()
    assert not read_scoped_episodes(db, scope(), 'role').episodes


def test_cross_session_reads_keep_sources_and_limit_is_explicit(db):
    for i in range(12):
        add(db, str(i), sessionId='s' + str(i))
    result = read_scoped_episodes(db, scope(), 'role', max_rows=3)
    assert result.older_rows_omitted
    assert [r.message for r in result.episodes] == ['9', '10', '11']
    assert [r.source_id for r in result.episodes] == sorted(r.source_id for r in result.episodes)


def test_optout_update_remains_visible_to_filter_not_silently_removed(db):
    add(db)
    add(db, '不要保存这个安排的修改。')
    source = read_scoped_episodes(db, scope(), 'role')
    result = select_evidence(list(source.episodes), '周六', source.scope, mode='suffix', max_chars=2000)
    assert result['source_ids'] == [] and result['blocked_sessions']


def test_character_and_scope_cannot_be_unspecified(db):
    with pytest.raises(ValueError):
        read_scoped_episodes(db, scope(), '')
    with pytest.raises(ValueError):
        read_scoped_episodes(db, replace(scope(), sender_id=''), 'role')


def test_real_storage_to_suffix_keeps_unmatched_later_correction(db):
    add(db, '周六去美术馆。', sessionId='s1')
    add(db, '改了，那天在家整理相册。', sessionId='s2')
    source = read_scoped_episodes(db, scope(), 'role')
    assert not source.older_rows_omitted
    result = select_evidence(list(source.episodes), '周六做什么？', source.scope, mode='suffix', max_chars=2000)
    assert len(result['source_ids']) == 2
    assert '整理相册' in result['packet']
    # No records or claims were written by retrieval.
    assert len(read_scoped_episodes(db, scope(), 'role').episodes) == 2


def test_narrative_is_rejected_until_branch_scope_is_available(db):
    add(db, '本场景', adapter='narrative', conversationId='scene-a')
    add(db, '其他场景', adapter='narrative', conversationId='scene-b')
    with pytest.raises(ValueError, match='branch-aware'):
        read_scoped_episodes(db, scope(adapter='narrative', conversation_id='scene-a'), 'role')


def test_removed_source_is_not_retained_in_an_episode_cache(db):
    add(db)
    source = read_scoped_episodes(db, scope(), 'role')
    source_id = int(source.episodes[0].source_id)
    # This disposable test database is the only deletion target.
    conn = db._get_connection()
    conn.execute('DELETE FROM messages WHERE id = ?', (source_id,))
    conn.commit()
    assert not read_scoped_episodes(db, scope(), 'role').episodes
