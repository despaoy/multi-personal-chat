"""Real SQLite/API contracts for selected public sources and independent copies."""
import importlib.util
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.character_library import repository, router
from app.dependencies import get_current_user
from character.library import CharacterSettings
from db.database import SQLiteDB
from db.errors import RevisionConflictError
from repositories.character_chat import CharacterChatRepository
from repositories.character_library import CharacterLibraryRepository, CharacterNotFoundError
from repositories.character_sources import CharacterSourceRepository


@pytest.fixture
def env(tmp_path):
    db = SQLiteDB(tmp_path / 'publications.db')
    library = CharacterLibraryRepository(db)
    sources = CharacterSourceRepository(db)
    author = library.create('author', CharacterSettings(name='旅人', persona='谨慎', greeting='你好'))
    corpus = sources.publish('author', author['id'], 1, [
        {'id': 'public', 'title': '灯塔', 'content': '可以分享的原文'},
        {'id': 'unselected', 'title': '未选入', 'content': 'UNSELECTED_SECRET'},
    ])
    yield library, sources, author, corpus
    db.close_connection()


def publish(env, corpus=None, ids=None):
    library, _, author, _ = env
    preview = library.publish_preview('author', author['id'], author['revision'], corpus, ids)
    return library.publish('author', author['id'], author['revision'], True, corpus, ids,
                           preview['preview_id'], preview['previous_publication_id'])


def copy(env):
    library, _, _, corpus = env
    publication = publish(env, corpus['id'], ['public'])
    result = library.copy_shared('recipient', publication['share_token'], publication['publication_id'], 'copy-1')
    return publication, result


def test_default_excludes_sources_and_only_selected_content_crosses_accounts(env):
    library, sources, author, corpus = env
    chats = CharacterChatRepository(library.database)
    session = chats.create('author', author['id'], 'private')
    chats.commit_turn(session, 'private-turn', 'PRIVATE_CHAT', 'PRIVATE_REPLY', 'test', state={
        'relationship': {'stage': 'close', 'summary': 'PRIVATE_RELATIONSHIP'},
        'memories': [{'id': 'secret', 'content': 'PRIVATE_MEMORY', 'status': 'confirmed'}]})
    assert library.shared(publish(env)['share_token'])['sources'] == []
    publication, copied = copy(env)
    shared = library.shared(publication['share_token'])
    assert set(shared) == {'id', 'settings_revision', 'publication_id', 'settings', 'sources'}
    assert shared['sources'] == [corpus['sources'][0]]
    assert all(secret not in json.dumps(shared) for secret in ('UNSELECTED_SECRET', 'PRIVATE_CHAT', 'PRIVATE_MEMORY', 'PRIVATE_RELATIONSHIP'))
    local = sources.list('recipient', copied['id'], 1)[0]
    assert local['id'] != corpus['id']
    assert sources.get('recipient', local['id'])['sources'] == shared['sources']
    with pytest.raises(CharacterNotFoundError):
        sources.get('author', local['id'])
    first = chats.create('recipient', copied['id'], 'first meeting')
    assert first['baseline']['history'] == [] and first['state']['memories'] == []
    assert first['state']['relationship']['stage'] == 'stranger'
    assert first['source_corpus_id'] == local['id']


def test_copy_has_fresh_index_outbox_and_retry_fences_old_attempts(env):
    library, sources, author, corpus = env
    sources.mark_indexed('author', corpus['id'], {'corpus_digest': corpus['content_digest'], 'chunk_count': 2,
                                                'index_name': 'authors-index', 'embedding_identity': 'model'})
    _, copied = copy(env)
    local = sources.get('recipient', sources.list('recipient', copied['id'], 1)[0]['id'])
    assert local['index_receipt_json'] is None and local['index_status'] == 'pending' and local['index_attempt'] == 1
    assert {'owner': 'recipient', 'corpus_id': local['id'], 'attempt': 1} in sources.pending_index_jobs()
    sources.set_index_job_status('recipient', local['id'], 1, 'failed')
    assert sources.retry_index_job('recipient', local['id'], 1)['index_attempt'] == 2
    with pytest.raises(RevisionConflictError):
        sources.mark_indexed('recipient', local['id'], {'corpus_digest': local['content_digest'], 'chunk_count': 1}, expected_attempt=1)


@pytest.mark.parametrize('invalid', ['owner', 'character', 'revision', 'missing', 'duplicate', 'unpaired', 'revoked'])
def test_selection_authority_and_validation(env, invalid):
    library, sources, author, corpus = env
    owner, character, revision, corpus_id, ids = 'author', author['id'], 1, corpus['id'], ['public']
    if invalid == 'owner':
        owner = 'other'
    elif invalid == 'character':
        character = library.create('author', CharacterSettings(name='other', persona='x', greeting='x'))['id']
    elif invalid == 'revision':
        library.update('author', character, 1, CharacterSettings(name='new', persona='x', greeting='x'))
        revision = 2
    elif invalid == 'missing':
        ids = ['missing']
    elif invalid == 'duplicate':
        ids = ['public', 'public']
    elif invalid == 'unpaired':
        ids = []
    else:
        sources.revoke('author', corpus_id)
    with pytest.raises((CharacterNotFoundError, ValueError)):
        library.publish_preview(owner, character, revision, corpus_id, ids)
    assert library.get('author', author['id'])['share_token'] is None


def test_publication_preview_tampering_and_stale_confirmation(env):
    library, _, author, corpus = env
    preview = library.publish_preview('author', author['id'], 1, corpus['id'], ['public'])
    with pytest.raises(RevisionConflictError):
        library.publish('author', author['id'], 1, True, corpus['id'], ['unselected'], preview['preview_id'], None)
    publish(env)
    with pytest.raises(RevisionConflictError):
        library.publish('author', author['id'], 1, True, corpus['id'], ['public'], preview['preview_id'], None)


@pytest.mark.parametrize('action', ['close', 'revoke'])
def test_revocation_denies_new_read_and_copy_but_keeps_completed_copy(env, action):
    library, sources, author, corpus = env
    publication, copied = copy(env)
    if action == 'close':
        library.publish('author', author['id'], 1, False)
    else:
        sources.revoke('author', corpus['id'])
    with pytest.raises(CharacterNotFoundError):
        library.shared(publication['share_token'])
    with pytest.raises(CharacterNotFoundError):
        library.copy_shared('new-recipient', publication['share_token'], publication['publication_id'], 'new-copy')
    assert library.copy_shared('recipient', publication['share_token'], publication['publication_id'], 'copy-1')['id'] == copied['id']
    local = sources.list('recipient', copied['id'], 1)[0]
    assert sources.get('recipient', local['id'])['sources'][0]['content'] == '可以分享的原文'
    assert not library.source_update('recipient', copied['id'])['available']


def test_same_settings_new_sources_require_confirmed_update_and_keep_old_sessions(env):
    library, sources, author, corpus = env
    publication, copied = copy(env)
    chats = CharacterChatRepository(library.database)
    old = chats.create('recipient', copied['id'], 'old')
    newer = sources.publish('author', author['id'], 1, [{'id': 'public', 'title': '灯塔新版', 'content': '更新原文'}])
    assert not library.source_update('recipient', copied['id'])['available']
    next_publication = publish(env, newer['id'], ['public'])
    preview = library.source_update('recipient', copied['id'])
    assert preview['available'] and preview['source_revision'] == 1 and preview['changes'] == {}
    assert preview['publication_id'] != publication['publication_id']
    assert preview['sources'][0]['content'] == '更新原文'
    assert library.get('recipient', copied['id'])['revision'] == 1
    imported = library.import_source_update('recipient', copied['id'], 1, 1, next_publication['publication_id'], preview['local_corpus_id'])
    assert imported['revision'] == 2
    assert chats.get('recipient', old['id'])['source_corpus_id'] == old['source_corpus_id']
    assert chats.get('recipient', old['id'])['character_revision'] == 1
    fresh = chats.create('recipient', copied['id'], 'new')
    assert fresh['source_corpus_id'] != old['source_corpus_id']
    assert sources.get('recipient', fresh['source_corpus_id'])['sources'] == preview['sources']
    assert library.import_source_update('recipient', copied['id'], 1, 1, next_publication['publication_id'], preview['local_corpus_id'])['revision'] == 2
    assert len(sources.list('recipient', copied['id'], 2)) == 1
    # Removing all selected sources is also a new publication and an explicit update.
    publish(env)
    preview = library.source_update('recipient', copied['id'])
    assert preview['available'] and preview['sources'] == []
    library.import_source_update('recipient', copied['id'], 2, 1, preview['publication_id'], preview['local_corpus_id'])
    assert chats.create('recipient', copied['id'], 'no sources')['source_corpus_id'] is None
    assert sources.get('recipient', fresh['source_corpus_id'])['sources'][0]['content'] == '更新原文'


@pytest.mark.parametrize('change', ['settings', 'local_sources', 'local_revoke', 'author_republish', 'author_revoke'])
def test_update_refuses_changes_after_preview(env, change):
    library, sources, author, corpus = env
    _, copied = copy(env)
    publish(env)  # Explicitly removes sources from next version.
    preview = library.source_update('recipient', copied['id'])
    if change == 'settings':
        library.update('recipient', copied['id'], 1, CharacterSettings(name='my edit', persona='x', greeting='x'))
    elif change == 'local_sources':
        sources.publish('recipient', copied['id'], 1, [{'id': 'local', 'title': 'my work', 'content': 'keep'}])
    elif change == 'local_revoke':
        sources.revoke('recipient', preview['local_corpus_id'])
    elif change == 'author_republish':
        publish(env, corpus['id'], ['unselected'])
    else:
        library.publish('author', author['id'], 1, False)
    with pytest.raises(RevisionConflictError):
        library.import_source_update('recipient', copied['id'], 1, 1, preview['publication_id'], preview['local_corpus_id'])
    assert library.get('recipient', copied['id'])['revision'] == (2 if change == 'settings' else 1)


def test_copy_rechecks_source_revocation_inside_transaction(env, monkeypatch):
    library, sources, _, corpus = env
    publication = publish(env, corpus['id'], ['public'])
    commit = library.database.revision_transaction

    def revoke_then_commit(statements):
        if any('INSERT INTO character_publication_copies' in sql for sql, _, _ in statements):
            sources.revoke('author', corpus['id'])
        return commit(statements)

    monkeypatch.setattr(library.database, 'revision_transaction', revoke_then_commit)
    with pytest.raises(RevisionConflictError):
        library.copy_shared('recipient', publication['share_token'], publication['publication_id'], 'retry')
    assert library.list('recipient') == []
    assert library.database.execute_sql("SELECT * FROM character_source_corpora WHERE owner_user_id='recipient'") == []


def test_source_insertion_failure_rolls_back_copy_and_retry_can_succeed(env, monkeypatch):
    library, _, _, corpus = env
    publication = publish(env, corpus['id'], ['public'])
    commit = library.database.revision_transaction

    def fail_at_sources(statements):
        return commit([(sql.replace('INSERT INTO character_source_corpora', 'INSERT INTO missing_table'), params, count)
                       for sql, params, count in statements])

    monkeypatch.setattr(library.database, 'revision_transaction', fail_at_sources)
    with pytest.raises(sqlite3.OperationalError):
        library.copy_shared('recipient', publication['share_token'], publication['publication_id'], 'retry')
    assert library.list('recipient') == []
    assert library.database.execute_sql('SELECT * FROM character_publication_copies') == []
    monkeypatch.setattr(library.database, 'revision_transaction', commit)
    result = library.copy_shared('recipient', publication['share_token'], publication['publication_id'], 'retry')
    assert library.copy_shared('recipient', publication['share_token'], publication['publication_id'], 'retry')['id'] == result['id']


def test_concurrent_copy_retries_commit_one_independent_copy(env):
    library, sources, _, corpus = env
    publication = publish(env, corpus['id'], ['public'])

    def do_copy(_):
        try:
            return library.copy_shared('recipient', publication['share_token'], publication['publication_id'], 'same-request')['id']
        finally:
            library.database.close_connection()

    with ThreadPoolExecutor(max_workers=2) as pool:
        identities = list(pool.map(do_copy, range(2)))
    assert identities[0] == identities[1]
    assert len(library.list('recipient')) == 1
    assert len(sources.list('recipient', identities[0], 1)) == 1
    other = publish(env)
    with pytest.raises(RevisionConflictError):
        library.copy_shared('recipient', other['share_token'], other['publication_id'], 'same-request')


def test_http_preview_confirmation_account_isolation_and_v1_unchanged(env):
    library, _, author, corpus = env
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[repository] = lambda: library
    app.dependency_overrides[get_current_user] = lambda: {'user_id': 'author'}
    base = f"/api/character-library/{author['id']}"
    with TestClient(app) as client:
        selection = {'revision': 1, 'corpus_id': corpus['id'], 'source_ids': ['public']}
        assert client.put(base + '/sharing', json={**selection, 'enabled': True}).status_code == 422
        preview = client.post(base + '/sharing-preview', json=selection).json()
        confirmation = {**selection, 'enabled': True, 'preview_id': preview['preview_id'], 'previous_publication_id': None}
        publication = client.put(base + '/sharing', json=confirmation).json()
        assert client.put(base + '/sharing', json=confirmation).json() == publication
        exported = client.get(base + '/export').json()
        assert set(exported) == {'format', 'settings'} and exported['format'] == 'multipersonal-character-v1'
        assert client.post('/api/character-library/import', json={**exported, 'sources': corpus['sources']}).status_code == 422
        app.dependency_overrides[get_current_user] = lambda: {'user_id': 'recipient'}
        assert client.post(base + '/sharing-preview', json=selection).status_code == 404
        url = f"/api/character-library/shared/{publication['share_token']}"
        assert client.post(url + '/copy').status_code == 422
        shared = client.get(url).json()
        body = {'publication_id': shared['publication_id'], 'request_id': 'http-copy'}
        copied = client.post(url + '/copy', json=body)
        assert copied.status_code == 201
        assert client.post(url + '/copy', json=body).json()['id'] == copied.json()['id']
        app.dependency_overrides[get_current_user] = lambda: {'user_id': 'other'}
        assert client.get('/api/character-library/' + copied.json()['id']).status_code == 404


def test_migration_preserves_legacy_settings_only_shares_and_is_repeatable(tmp_path):
    import sqlalchemy as sa
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    def migration(name, connection):
        spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'alembic' / 'versions' / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.op = Operations(MigrationContext.configure(connection))
        return module

    engine = sa.create_engine('sqlite:///' + str(tmp_path / 'migration.db'))
    with engine.begin() as connection:
        migration('017_character_library', connection).upgrade()
        connection.execute(sa.text("INSERT INTO character_library (id,owner_user_id,revision,shared_revision,share_token,created_at,updated_at) VALUES ('old','owner',2,1,'token','now','now')"))
        settings = CharacterSettings(name='published', persona='x', greeting='x').model_dump_json()
        connection.execute(sa.text("INSERT INTO character_library_versions VALUES ('old',1,:settings,'now')"), {'settings': settings})
        connection.execute(sa.text("INSERT INTO character_library_versions VALUES ('old',2,:settings,'now')"), {'settings': settings.replace('published', 'draft')})
        upgrade = migration('028_character_publications', connection)
        upgrade.upgrade()
        upgrade.upgrade()
        row = connection.execute(sa.text('SELECT * FROM character_publications')).mappings().one()
        assert row['settings_json'] == settings and row['sources_json'] == '[]'
        assert connection.execute(sa.text('SELECT publication_id FROM character_library')).scalar() == row['id']
        upgrade.downgrade()
        assert connection.execute(sa.text('SELECT COUNT(*) FROM character_library_versions')).scalar() == 2
        assert 'publication_id' not in {c['name'] for c in sa.inspect(connection).get_columns('character_library')}
    engine.dispose()


def test_editing_copied_settings_preserves_sources_without_mutating_old_session(env):
    library, sources, _, _ = env
    _, copied = copy(env)
    chats = CharacterChatRepository(library.database)
    old = chats.create('recipient', copied['id'], 'old')
    edited = CharacterSettings.model_validate(copied['settings'])
    edited.greeting = '自己的开场白'
    library.update('recipient', copied['id'], 1, edited)
    new = chats.create('recipient', copied['id'], 'new')
    assert new['character']['settings']['greeting'] == '自己的开场白'
    assert new['source_corpus_id'] != old['source_corpus_id']
    old_corpus = sources.get('recipient', old['source_corpus_id'])
    new_corpus = sources.get('recipient', new['source_corpus_id'])
    assert old_corpus['sources'] == new_corpus['sources']
    assert new_corpus['index_status'] == 'pending' and new_corpus['index_receipt_json'] is None
    assert chats.get('recipient', old['id'])['character']['settings']['greeting'] == '你好'


@pytest.mark.parametrize('operation', ['publish', 'import', 'edit'])
def test_transaction_failures_leave_heads_and_versions_unchanged(env, monkeypatch, operation):
    library, sources, author, corpus = env
    _, copied = copy(env)
    if operation == 'import':
        newer = sources.publish('author', author['id'], 1, [{'id': 'public', 'title': 'new', 'content': 'new'}])
        publish(env, newer['id'], ['public'])
    before_characters = library.database.execute_sql('SELECT * FROM character_library ORDER BY id')
    before_versions = library.database.execute_sql('SELECT * FROM character_library_versions ORDER BY character_id,revision')
    before_corpora = library.database.execute_sql('SELECT * FROM character_source_corpora ORDER BY id')
    commit = library.database.revision_transaction

    def fail_insert(statements):
        return commit([('INSERT INTO missing_table VALUES (1)', {}, 1) if sql.startswith('INSERT INTO character_')
                       else (sql, params, count) for sql, params, count in statements])

    monkeypatch.setattr(library.database, 'revision_transaction', fail_insert)
    with pytest.raises(sqlite3.OperationalError):
        if operation == 'publish':
            publish(env)
        elif operation == 'import':
            preview = library.source_update('recipient', copied['id'])
            library.import_source_update('recipient', copied['id'], 1, 1, preview['publication_id'], preview['local_corpus_id'])
        else:
            library.update('recipient', copied['id'], 1, CharacterSettings(name='edited', persona='x', greeting='x'))
    assert library.database.execute_sql('SELECT * FROM character_library ORDER BY id') == before_characters
    assert library.database.execute_sql('SELECT * FROM character_library_versions ORDER BY character_id,revision') == before_versions
    assert library.database.execute_sql('SELECT * FROM character_source_corpora ORDER BY id') == before_corpora


def test_sqlite_runtime_upgrade_preserves_existing_share(tmp_path):
    path = tmp_path / 'legacy-runtime.db'
    db = SQLiteDB(path)
    library = CharacterLibraryRepository(db)
    character = library.create('owner', CharacterSettings(name='legacy', persona='x', greeting='x'))
    db.execute_sql("UPDATE character_library SET share_token='legacy-token',shared_revision=1 WHERE id=:id", {'id': character['id']})
    db.close_connection()
    db = SQLiteDB(path)
    try:
        shared = CharacterLibraryRepository(db).shared('legacy-token')
        assert shared['settings']['name'] == 'legacy' and shared['sources'] == []
        assert shared['publication_id'].startswith('legacy-')
    finally:
        db.close_connection()


@pytest.mark.parametrize('operation', ['import', 'edit'])
def test_local_sources_changed_during_commit_are_not_silently_overwritten(env, monkeypatch, operation):
    library, sources, _, _ = env
    _, copied = copy(env)
    publish(env)
    preview = library.source_update('recipient', copied['id'])
    commit = library.database.revision_transaction
    injected = False

    def concurrent_local_publish(statements):
        nonlocal injected
        if not injected and any('INSERT INTO character_library_versions' in sql for sql, _, _ in statements):
            injected = True
            sources.publish('recipient', copied['id'], 1, [{'id': 'local', 'title': 'new local', 'content': 'keep me'}])
        return commit(statements)

    monkeypatch.setattr(library.database, 'revision_transaction', concurrent_local_publish)
    with pytest.raises(RevisionConflictError):
        if operation == 'import':
            library.import_source_update('recipient', copied['id'], 1, 1, preview['publication_id'], preview['local_corpus_id'])
        else:
            library.update('recipient', copied['id'], 1, CharacterSettings(name='edit', persona='x', greeting='x'))
    assert library.get('recipient', copied['id'])['revision'] == 1
    assert sources.get('recipient', sources.list('recipient', copied['id'], 1)[0]['id'])['sources'][0]['content'] == 'keep me'


def test_concurrent_identical_publications_keep_one_valid_token(env, monkeypatch):
    from threading import Barrier

    from repositories.character_publications import CharacterPublicationRepository

    library, _, author, _ = env
    barrier = Barrier(2)
    original = CharacterPublicationRepository.preview

    def simultaneous_preview(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        barrier.wait(timeout=10)
        return result

    monkeypatch.setattr(CharacterPublicationRepository, 'preview', simultaneous_preview)

    def do_publish(_):
        try:
            return library.publish('author', author['id'], 1, True)
        finally:
            library.database.close_connection()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(do_publish, range(2)))
    assert results[0] == results[1]
    assert library.shared(results[0]['share_token'])['publication_id'] == results[0]['publication_id']
    assert len(library.database.execute_sql('SELECT * FROM character_publications')) == 1
