"""Final conflict choices exercised through HTTP and real temporary SQLite databases."""
import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_character_source_merge import current_documents, doc, local_edit, preview, publish_author
from test_character_source_merge import env as merge_env

from api.character_library import repository, router
from app.dependencies import get_current_user
from character.library import CharacterSettings
from repositories.character_chat import CharacterChatRepository
from repositories.character_source_merge import plan_source_merge

env = merge_env


@pytest.fixture
def api(env):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[repository] = lambda: env['library']
    app.dependency_overrides[get_current_user] = lambda: {'user_id': 'recipient'}
    with TestClient(app) as client:
        yield client, app, f"/api/character-library/{env['recipient']['id']}/source-update"


def conflict(env):
    local_edit(env, [doc('a', 'local'), doc('b', 'local-b'), doc('own')])
    publish_author(env, [doc('a', 'author'), doc('remote')])
    plan = preview(env)
    return {key: plan[key] for key in ('revision', 'source_revision', 'publication_id', 'local_corpus_id')}


def final(api, request, choices):
    client, _, url = api
    response = client.post(url + '/preview', json={**request, 'choices': choices})
    assert response.status_code == 200, response.text
    return response.json(), {**request, 'choices': choices, 'preview_id': response.json()['preview_id']}


def test_mixed_choices_final_preview_exact_retry_and_old_session(env, api):
    request = conflict(env)
    chats = CharacterChatRepository(env['db'])
    old = chats.create('recipient', env['recipient']['id'], 'old')
    result, confirmation = final(api, request, {'a': 'local', 'b': 'author'})
    expected = [doc('a', 'local'), doc('own'), doc('remote')]
    assert result['sources'] == expected
    assert {a['action'] for a in result['source_merge']['actions']} >= {'choose_local_version', 'choose_author_deletion'}
    client, app, url = api
    assert client.post(url, json=confirmation).status_code == 200
    assert current_documents(env) == expected
    assert chats.get('recipient', old['id'])['character_revision'] == 1
    assert chats.get('recipient', old['id'])['source_corpus_id'] == old['source_corpus_id']
    new = chats.create('recipient', env['recipient']['id'], 'new')
    corpus = env['sources'].get('recipient', new['source_corpus_id'])
    assert corpus['sources'] == expected and corpus['index_status'] == 'pending'
    assert corpus['index_receipt_json'] is None
    assert client.post(url, json={**confirmation, 'choices': {'a': 'author', 'b': 'author'}}).status_code == 409
    env['library'].publish('author', env['author']['id'], 1, False)
    assert client.post(url, json=confirmation).json()['revision'] == 2
    assert len(env['db'].execute_sql('SELECT * FROM character_source_update_receipts')) == 1
    app.dependency_overrides[get_current_user] = lambda: {'user_id': 'author'}
    assert client.post(url, json=confirmation).status_code == 404
    assert client.post(url + '/preview', json={**request, 'choices': {'a': 'local', 'b': 'author'}}).status_code == 404


@pytest.mark.parametrize('side,expected', [('local', []), ('author', [doc('a', 'author')])])
def test_local_deletion_choice(env, api, side, expected):
    current = env['sources'].list('recipient', env['recipient']['id'], 1)[0]
    env['sources'].revoke('recipient', current['id'])
    publish_author(env, [doc('a', 'author')])
    plan = preview(env)
    request = {k: plan[k] for k in ('revision', 'source_revision', 'publication_id', 'local_corpus_id')}
    result, confirmation = final(api, request, {'a': side})
    assert result['sources'] == expected
    assert api[0].post(api[2], json=confirmation).status_code == 200
    assert current_documents(env) == expected


@pytest.mark.parametrize('choices', [{}, {'a': 'local'}, {'a': 'local', 'b': 'author', 'own': 'local'},
                                      {'a': 'local', 'b': 'author', 'unknown': 'author'}, {'a': 'merge', 'b': 'local'}, None])
def test_missing_unknown_nonconflict_invalid_choices_write_nothing(env, api, choices):
    request = conflict(env)
    response = api[0].post(api[2] + '/preview', json={**request, 'choices': choices})
    assert response.status_code == 422
    assert env['library'].get('recipient', env['recipient']['id'])['revision'] == 1
    assert env['db'].execute_sql('SELECT * FROM character_source_update_receipts') == []


@pytest.mark.parametrize('payload', [{'sources': [doc('injected')]}, {'preview_id': '0' * 64}, {}])
def test_preview_does_not_accept_body_or_missing_choices(env, api, payload):
    request = conflict(env)
    assert api[0].post(api[2] + '/preview', json={**request, **payload}).status_code == 422


def test_confirmation_requires_pair_and_binds_choices_and_result(env, api):
    request = conflict(env)
    _, confirmation = final(api, request, {'a': 'local', 'b': 'author'})
    for key in ('choices', 'preview_id'):
        invalid = {**confirmation}
        del invalid[key]
        assert api[0].post(api[2], json=invalid).status_code == 422
    for invalid in ({**confirmation, 'preview_id': '0' * 64},
                    {**confirmation, 'choices': {'a': 'author', 'b': 'author'}}):
        assert api[0].post(api[2], json=invalid).status_code == 409
    assert api[0].post(api[2], json={**confirmation, 'sources': [doc('injected')]}).status_code == 422
    assert env['library'].get('recipient', env['recipient']['id'])['revision'] == 1


@pytest.mark.parametrize('change', ['local', 'author', 'baseline_revoke', 'incoming_revoke', 'close_share', 'local_settings'])
def test_final_preview_expiration(env, api, change):
    request = conflict(env)
    _, confirmation = final(api, request, {'a': 'local', 'b': 'author'})
    if change == 'local':
        local_edit(env, [doc('a', 'later')])
    elif change == 'author':
        publish_author(env, [doc('a', 'later author')])
    elif change == 'baseline_revoke':
        env['sources'].revoke('author', env['corpus']['id'])
    elif change == 'incoming_revoke':
        corpus = env['sources'].list('author', env['author']['id'], 1)[0]
        env['sources'].revoke('author', corpus['id'])
    elif change == 'close_share':
        env['library'].publish('author', env['author']['id'], 1, False)
    else:
        env['library'].update('recipient', env['recipient']['id'], 1, CharacterSettings(name='edited', persona='x', greeting='x'))
    assert api[0].post(api[2], json=confirmation).status_code == 409
    assert env['db'].execute_sql('SELECT * FROM character_source_update_receipts') == []


@pytest.mark.parametrize('mode', ['collision', 'missing_baseline', 'revoked_baseline'])
def test_choices_cannot_override_unsafe_baseline_or_identity(env, api, mode):
    if mode == 'collision':
        local_edit(env, env['initial'] + [doc('collision')])
        publish_author(env, env['initial'] + [doc('collision')])
        choices = {'collision': 'local'}
    else:
        conflict(env)
        local_edit(env, [doc('a', 'local')])
        choices = {'a': 'local'}
        if mode == 'missing_baseline':
            env['db'].execute_sql('UPDATE character_library SET source_publication_id=NULL WHERE id=:id', {'id': env['recipient']['id']})
        else:
            env['sources'].revoke('author', env['corpus']['id'])
    plan = preview(env)
    request = {k: plan[k] for k in ('revision', 'source_revision', 'publication_id', 'local_corpus_id')}
    response = api[0].post(api[2] + '/preview', json={**request, 'choices': choices})
    assert response.status_code == 409
    assert 'BASE_ONLY_SECRET' not in response.text


def test_transaction_race_rolls_back_receipt_and_version(env, api, monkeypatch):
    request = conflict(env)
    _, confirmation = final(api, request, {'a': 'local', 'b': 'author'})
    commit = env['db'].revision_transaction
    injected = False

    def raced(statements):
        nonlocal injected
        if not injected:
            injected = True
            env['sources'].revoke('author', env['corpus']['id'])
        return commit(statements)

    monkeypatch.setattr(env['db'], 'revision_transaction', raced)
    assert api[0].post(api[2], json=confirmation).status_code == 409
    assert env['library'].get('recipient', env['recipient']['id'])['revision'] == 1
    assert env['db'].execute_sql('SELECT * FROM character_source_update_receipts') == []


def test_conflict_free_final_preview_and_legacy_regression(env, api):
    publish_author(env, env['initial'] + [doc('new')])
    plan = preview(env)
    request = {k: plan[k] for k in ('revision', 'source_revision', 'publication_id', 'local_corpus_id')}
    result, confirmation = final(api, request, {})
    assert result['source_merge']['resolved_conflicts'] == []
    assert api[0].post(api[2], json=confirmation).status_code == 200
    publish_author(env, env['initial'] + [doc('next')])
    plan = preview(env)
    legacy = {k: plan[k] for k in request}
    assert api[0].post(api[2], json=legacy).status_code == 200
    assert api[0].post(api[2], json=confirmation).status_code == 409


def test_capacity_checked_after_choices():
    base = [doc('a')]
    left = [doc('a', 'local')] + [doc(f'l-{i}') for i in range(50)]
    right = [doc('a', 'author')] + [doc(f'r-{i}') for i in range(50)]
    assert plan_source_merge(base, left, right, {'a': 'local'})['reason'] == 'merged_source_capacity_exceeded'
    assert len(plan_source_merge(base, left[:-1], right, {'a': 'author'})['merged_sources']) == 100


def test_receipt_migration_upgrade_downgrade(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'alembic/versions/029_source_update_receipts.py'
    spec = importlib.util.spec_from_file_location('receipt_migration', path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    with engine.begin() as connection:
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        migration.upgrade()
        assert {c['name'] for c in sa.inspect(connection).get_columns('character_source_update_receipts')} == {
            'owner_user_id', 'character_id', 'preview_id', 'request_json', 'result_revision'}
        migration.downgrade()
        assert not sa.inspect(connection).has_table('character_source_update_receipts')
    engine.dispose()


def test_concurrent_exact_confirmation_commits_one_version(env, api, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from repositories.character_publications import CharacterPublicationRepository

    request = conflict(env)
    _, confirmation = final(api, request, {'a': 'local', 'b': 'author'})
    barrier = Barrier(2)
    original = CharacterPublicationRepository._prepare_update

    def together(self, *args):
        result = original(self, *args)
        barrier.wait(timeout=10)
        return result

    monkeypatch.setattr(CharacterPublicationRepository, '_prepare_update', together)

    def accept(_):
        try:
            return env['library'].import_source_update('recipient', env['recipient']['id'], **confirmation)
        finally:
            env['db'].close_connection()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(accept, range(2)))
    assert results[0] == results[1] and results[0]['revision'] == 2
    assert len(env['db'].execute_sql('SELECT * FROM character_source_update_receipts')) == 1
    assert len(env['sources'].list('recipient', env['recipient']['id'], 2)) == 1


def test_completed_confirmation_during_preparation_returns_receipt(env, api, monkeypatch):
    from repositories.character_publications import CharacterPublicationRepository

    request = conflict(env)
    _, confirmation = final(api, request, {'a': 'local', 'b': 'author'})
    original = CharacterPublicationRepository._prepare_update
    injected = False

    def finish_first(self, *args):
        nonlocal injected
        if not injected:
            injected = True
            env['library'].import_source_update('recipient', env['recipient']['id'], **confirmation)
        return original(self, *args)

    monkeypatch.setattr(CharacterPublicationRepository, '_prepare_update', finish_first)
    assert api[0].post(api[2], json=confirmation).json()['revision'] == 2
    assert len(env['db'].execute_sql('SELECT * FROM character_source_update_receipts')) == 1


def test_receipt_write_failure_rolls_back_then_retry_succeeds(env, api, monkeypatch):
    request = conflict(env)
    _, confirmation = final(api, request, {'a': 'local', 'b': 'author'})
    commit = env['db'].revision_transaction

    def fail_receipt(statements):
        return commit([(sql.replace('INSERT INTO character_source_update_receipts', 'INSERT INTO missing_receipt_table'), params, count)
                       for sql, params, count in statements])

    monkeypatch.setattr(env['db'], 'revision_transaction', fail_receipt)
    with pytest.raises(Exception, match='missing_receipt_table'):
        env['library'].import_source_update('recipient', env['recipient']['id'], **confirmation)
    assert env['library'].get('recipient', env['recipient']['id'])['revision'] == 1
    assert env['db'].execute_sql('SELECT * FROM character_source_update_receipts') == []
    assert env['db'].execute_sql('SELECT * FROM character_source_corpora WHERE character_id=:id AND character_revision=2',
                                 {'id': env['recipient']['id']}) == []
    monkeypatch.setattr(env['db'], 'revision_transaction', commit)
    assert api[0].post(api[2], json=confirmation).status_code == 200


def test_reopened_identical_share_invalidates_final_preview(env, api):
    request = conflict(env)
    _, confirmation = final(api, request, {'a': 'local', 'b': 'author'})
    corpus = env['sources'].list('author', env['author']['id'], 1)[0]
    env['library'].publish('author', env['author']['id'], 1, False)
    reopened = env['library'].publish('author', env['author']['id'], 1, True, corpus['id'], ['a', 'remote'])
    assert reopened['publication_id'] == request['publication_id']
    assert api[0].post(api[2], json=confirmation).status_code == 409
    _, refreshed = final(api, request, {'a': 'local', 'b': 'author'})
    assert refreshed['preview_id'] != confirmation['preview_id']
    assert api[0].post(api[2], json=refreshed).status_code == 200


def test_other_owned_copy_cannot_reuse_final_preview(env, api):
    second = env['library'].copy_shared('recipient', env['publication']['share_token'],
                                         env['publication']['publication_id'], 'second-copy')
    request = conflict(env)
    _, confirmation = final(api, request, {'a': 'local', 'b': 'author'})
    env['sources'].publish('recipient', second['id'], 1, [doc('a', 'local'), doc('b', 'local-b'), doc('own')])
    plan = env['library'].source_update('recipient', second['id'])
    forged = {**confirmation, 'local_corpus_id': plan['local_corpus_id']}
    assert api[0].post(f"/api/character-library/{second['id']}/source-update", json=forged).status_code == 409
    assert env['library'].get('recipient', second['id'])['revision'] == 1


def test_resolved_capacity_refused_by_final_preview_database(env, api):
    local_edit(env, [doc('a', 'local')] + [doc(f'l-{i}') for i in range(50)])
    publish_author(env, [doc('a', 'author')] + [doc(f'r-{i}') for i in range(50)])
    plan = preview(env)
    request = {k: plan[k] for k in ('revision', 'source_revision', 'publication_id', 'local_corpus_id')}
    assert api[0].post(api[2] + '/preview', json={**request, 'choices': {'a': 'local'}}).status_code == 409
    assert env['library'].get('recipient', env['recipient']['id'])['revision'] == 1
    assert env['db'].execute_sql('SELECT * FROM character_source_update_receipts') == []
