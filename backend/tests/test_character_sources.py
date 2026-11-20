import os
import sqlite3
from contextlib import closing

import pytest

from character.library import CharacterSettings
from db.database import SQLiteDB
from db.errors import RevisionConflictError
from repositories.character_chat import CharacterChatRepository
from repositories.character_library import CharacterNotFoundError
from repositories.character_sources import CharacterSourceRepository


@pytest.fixture
def sources(tmp_path):
    if os.getenv('RUN_CHARACTER_PG_INTEGRATION') == '1':
        from db.pg_database import PgDatabase, SyncPgAdapter

        db = SyncPgAdapter(PgDatabase(os.environ['DATABASE_URL']))
    else:
        db = SQLiteDB(tmp_path / 'sources.db')
    try:
        repo = CharacterSourceRepository(db)
        char = repo.characters.create('owner', CharacterSettings(name='旅人', persona='谨慎', greeting='你好'))
        yield repo, char
    finally:
        if isinstance(db, SQLiteDB):
            db.close_connection()
        else:
            db.close()


def publish(repo, char, content='第一行原文\n第二行原文\n第三行原文'):
    return repo.publish('owner', char['id'], 1, [{'id': 'chapter1', 'title': '第一章', 'content': content}])


def test_sqlite_source_sequence_upgrade_preserves_old_editions(tmp_path, monkeypatch):
    from repositories import character_sources

    path = tmp_path / 'old-editions.db'
    db = SQLiteDB(path)
    try:
        repo = CharacterSourceRepository(db)
        char = repo.characters.create('owner', CharacterSettings(name='旅人', persona='谨慎', greeting='你好'))
        first = publish(repo, char, '曾经保存的版本')
    finally:
        db.close_connection()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute('DROP INDEX ix_character_source_edition')
        connection.execute('ALTER TABLE character_source_corpora DROP COLUMN edition_number')
    reopened = SQLiteDB(path)
    try:
        repo = CharacterSourceRepository(reopened)
        monkeypatch.setattr(character_sources, 'timestamp', lambda: '2000-01-01T00:00:00+00:00')
        new = publish(repo, char, '重启并迁移后保存的新版本')
        assert [row['id'] for row in repo.list('owner', char['id'], 1)] == [new['id'], first['id']]
        assert repo.get('owner', first['id'])['sources'] == first['sources']
        repo.revoke('owner', new['id'])
        assert [row['id'] for row in repo.list('owner', char['id'], 1)] == [first['id']]
    finally:
        reopened.close_connection()


def test_editions_are_immutable_and_revocation_blocks_stale_candidates(sources):
    repo, char = sources
    first = publish(repo, char)
    second = publish(repo, char, '新版本原文')
    assert first['id'] != second['id']
    assert repo.get('owner', first['id'])['sources'][0]['content'] != second['sources'][0]['content']
    candidate = repo.chunks('owner', first['id'], max_chars=11)[0]
    candidate['title'], candidate['content'] = '伪造标题', '索引注入内容'
    verified = repo.verify_candidate('owner', char['id'], 1, first['id'], candidate)
    assert verified['title'] == '第一章' and verified['content'] == '第一行原文\n第二行原文'
    chats = CharacterChatRepository(repo.database)
    session = chats.create('owner', char['id'], '旧资料绑定', source_corpus_id=first['id'])
    turn = chats.commit_turn(session, 'first', '原文是什么？', '回复', 'test')
    repo.revoke('owner', first['id'])
    with pytest.raises(CharacterNotFoundError):
        repo.verify_candidate('owner', char['id'], 1, first['id'], candidate)
    assert repo.get('owner', second['id'])['id'] == second['id']
    assert repo.has_edition('owner', char['id'], 1, first['id'], revoked=True)
    assert not repo.has_edition('foreign-owner', char['id'], 1, first['id'], revoked=True)
    branch = chats.create('owner', char['id'], '历史分支', source_session_id=session['id'],
                          source_revision=session['revision'] + 1, source_turn_id=turn['id'])
    assert branch['source_corpus_id'] == first['id']


@pytest.mark.parametrize('field,value', [('owner_id', 'other'), ('content_sha256', 'bad'), ('line_end', 100), ('chunk_id', 'forged')])
def test_candidate_tampering_fails_before_source_admission(sources, field, value):
    repo, char = sources
    corpus = publish(repo, char)
    candidate = repo.chunks('owner', corpus['id'])[0]
    candidate[field] = value
    with pytest.raises(ValueError):
        repo.verify_candidate('owner', char['id'], 1, corpus['id'], candidate)
    with pytest.raises(CharacterNotFoundError):
        repo.get('other', corpus['id'])


def test_old_character_revision_cannot_publish_and_chunks_preserve_all_lines(sources):
    repo, char = sources
    corpus = publish(repo, char)
    chunks = repo.chunks('owner', corpus['id'], max_chars=6)
    assert [(c['line_start'], c['line_end']) for c in chunks] == [(1, 1), (2, 2), (3, 3)]
    assert '\n'.join(c['content'] for c in chunks) == corpus['sources'][0]['content']
    repo.characters.update('owner', char['id'], 1, CharacterSettings(name='旅人新版', persona='谨慎', greeting='你好'))
    with pytest.raises(RevisionConflictError):
        publish(repo, char)
    assert repo.get('owner', corpus['id'])['character_revision'] == 1


def test_index_receipt_requires_current_model_and_live_source(sources):
    repo, char = sources
    corpus = publish(repo, char)
    with pytest.raises(ValueError, match='索引'):
        repo.require_indexed('owner', corpus['id'], 'character-sources', 'model-v1')
    receipt = {'corpus_digest': corpus['content_digest'], 'index_name': 'character-sources',
               'embedding_identity': 'model-v1', 'chunk_count': 1}
    repo.mark_indexed('owner', corpus['id'], receipt)
    assert repo.require_indexed('owner', corpus['id'], 'character-sources', 'model-v1')['id'] == corpus['id']
    with pytest.raises(ValueError, match='索引'):
        repo.require_indexed('owner', corpus['id'], 'character-sources', 'model-v2')
    repo.revoke('owner', corpus['id'])
    with pytest.raises(CharacterNotFoundError):
        repo.mark_indexed('owner', corpus['id'], receipt)


def test_index_retry_fences_old_workers_and_survives_repository_recreation(sources):
    repo, char = sources
    corpus = publish(repo, char)
    identity = corpus['id']
    receipt = {'corpus_digest': corpus['content_digest'], 'index_name': 'character-sources',
               'embedding_identity': 'model-v1', 'chunk_count': 1}
    assert corpus['index_status'] == 'pending' and corpus['index_attempt'] == 1
    assert {'corpus_id': identity, 'owner': 'owner', 'attempt': 1} in repo.pending_index_jobs()
    repo.set_index_job_status('owner', identity, 1, 'submitted')
    repo = CharacterSourceRepository(repo.database)
    assert repo.get('owner', identity)['index_status'] == 'submitted'
    assert all(job['corpus_id'] != identity for job in repo.pending_index_jobs())
    with pytest.raises(RevisionConflictError):
        repo.retry_index_job('owner', identity, 1)
    repo.set_index_job_status('owner', identity, 1, 'failed')
    with pytest.raises(CharacterNotFoundError):
        repo.retry_index_job('other', identity, 1)
    retried = repo.retry_index_job('owner', identity, 1)
    assert retried['index_attempt'] == 2 and retried['index_status'] == 'pending'
    with pytest.raises(RevisionConflictError):
        repo.mark_indexed('owner', identity, receipt, expected_attempt=1)
    with pytest.raises(RevisionConflictError):
        repo.set_index_job_status('owner', identity, 1, 'failed')
    repo.mark_indexed('owner', identity, receipt, expected_attempt=2)
    with pytest.raises(RevisionConflictError):
        repo.set_index_job_status('owner', identity, 2, 'failed')
    assert repo.require_indexed('owner', identity, 'character-sources', 'model-v1')['index_status'] == 'completed'


def test_revocation_cancels_pending_index_job(sources):
    repo, char = sources
    corpus = publish(repo, char)
    repo.revoke('owner', corpus['id'])
    assert all(job['corpus_id'] != corpus['id'] for job in repo.pending_index_jobs())
    with pytest.raises(CharacterNotFoundError):
        repo.set_index_job_status('owner', corpus['id'], 1, 'submitted')
