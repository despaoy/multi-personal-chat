"""Source merge plans, ownership, stale confirmations and old-session stability."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.character_library import repository, router
from app.dependencies import get_current_user
from character.library import CharacterSettings
from db.database import SQLiteDB
from db.errors import RevisionConflictError
from repositories.character_chat import CharacterChatRepository
from repositories.character_library import CharacterLibraryRepository, CharacterNotFoundError, SourceMergeBlockedError
from repositories.character_source_merge import plan_source_merge
from repositories.character_sources import CharacterSourceRepository


def doc(identity, content='original'):
    return {'id': identity, 'title': identity, 'content': content}


@pytest.fixture
def env(tmp_path):
    db = SQLiteDB(tmp_path / 'merge.db')
    library = CharacterLibraryRepository(db)
    sources = CharacterSourceRepository(db)
    author = library.create('author', CharacterSettings(name='旅人', persona='谨慎', greeting='你好'))
    initial = [doc('a'), doc('b', 'BASE_ONLY_SECRET')]
    corpus = sources.publish('author', author['id'], 1, initial)
    publication = library.publish('author', author['id'], 1, True, corpus['id'], ['a', 'b'])
    recipient = library.copy_shared('recipient', publication['share_token'], publication['publication_id'], 'initial-copy')
    yield {'db': db, 'library': library, 'sources': sources, 'author': author, 'recipient': recipient,
           'corpus': corpus, 'publication': publication, 'initial': initial}
    db.close_connection()


def publish_author(env, documents):
    library, sources, author = env['library'], env['sources'], env['author']
    if documents:
        corpus = sources.publish('author', author['id'], 1, documents)
        return library.publish('author', author['id'], 1, True, corpus['id'], [s['id'] for s in documents])
    return library.publish('author', author['id'], 1, True)


def local_edit(env, documents):
    current = env['library'].get('recipient', env['recipient']['id'])
    return env['sources'].publish('recipient', current['id'], current['revision'], documents)


def preview(env):
    return env['library'].source_update('recipient', env['recipient']['id'])


def accept(env, plan):
    return env['library'].import_source_update('recipient', env['recipient']['id'], plan['revision'],
                                               plan['source_revision'], plan['publication_id'], plan['local_corpus_id'])


def current_documents(env):
    current = env['library'].get('recipient', env['recipient']['id'])
    editions = env['sources'].list('recipient', current['id'], current['revision'], 1)
    return env['sources'].get('recipient', editions[0]['id'])['sources'] if editions else []


@pytest.mark.parametrize('left,right,expected,action', [
    ([doc('a', 'local')], [doc('a')], [doc('a', 'local')], 'keep_local_modification'),
    ([], [doc('a')], [], 'keep_local_deletion'),
    ([doc('a')], [doc('a', 'author')], [doc('a', 'author')], 'take_author_modification'),
    ([doc('a')], [], [], 'take_author_deletion'),
    ([], [], [], 'both_deleted'),
    ([doc('a', 'same')], [doc('a', 'same')], [doc('a', 'same')], 'same_modification'),
])
def test_three_way_document_decisions(left, right, expected, action):
    result = plan_source_merge([doc('a')], left, right)
    assert result['status'] == 'ready' and result['merged_sources'] == expected
    assert result['actions'] == [{'id': 'a', 'action': action}]


@pytest.mark.parametrize('left,right,kind', [
    ([doc('a', 'local')], [doc('a', 'author')], 'both_modified'),
    ([], [doc('a', 'author')], 'local_delete_author_modify'),
    ([doc('a', 'local')], [], 'local_modify_author_delete'),
])
def test_document_conflicts_are_not_partially_merged(left, right, kind):
    result = plan_source_merge([doc('a')], left + [doc('local-only')], right)
    assert result['status'] == 'conflict' and result['merged_sources'] == []
    assert result['conflicts'] == [{'id': 'a', 'kind': kind}]


@pytest.mark.parametrize('incoming', ['same', 'different'])
def test_independent_additions_with_same_id_are_conflicts_even_if_text_matches(incoming):
    plan = plan_source_merge([], [doc('collision', 'same')], [doc('collision', incoming)])
    assert plan['conflicts'] == [{'id': 'collision', 'kind': 'addition_id_collision'}]
    assert plan['merged_sources'] == []


def test_content_similarity_does_not_merge_distinct_source_ids():
    plan = plan_source_merge([], [doc('local')], [doc('author')])
    assert plan['status'] == 'ready' and [s['id'] for s in plan['merged_sources']] == ['local', 'author']


def test_duplicate_identity_is_fail_safe():
    result = plan_source_merge([doc('a'), doc('a')], [doc('a')], [doc('a')])
    assert result['status'] == 'blocked' and result['merged_sources'] == []


def test_union_capacity_is_enforced_after_merge():
    base = [doc('base')]
    local = base + [doc(f'local-{i}') for i in range(50)]
    remote = base + [doc(f'author-{i}') for i in range(50)]
    assert plan_source_merge(base, local, remote)['reason'] == 'merged_source_capacity_exceeded'
    assert len(plan_source_merge(base, local[:-1], remote)['merged_sources']) == 100
    # Individually under 5 MB and 200000 characters per source, together over 5 MB in UTF-8.
    local = [doc(f'local-{i}', '中' * 100000) for i in range(9)]
    remote = [doc(f'author-{i}', '文' * 100000) for i in range(9)]
    result = plan_source_merge([], local, remote)
    assert result['reason'] == 'merged_source_capacity_exceeded' and result['merged_sources'] == []


def test_before_preview_local_changes_are_preserved_and_old_sessions_keep_their_binding(env):
    local = [doc('a', 'my edit'), doc('local', 'my addition')]  # b deliberately deleted.
    local_corpus = local_edit(env, local)
    chats = CharacterChatRepository(env['db'])
    old = chats.create('recipient', env['recipient']['id'], 'before author update')
    publish_author(env, env['initial'] + [doc('new-author')])
    plan = preview(env)
    assert plan['source_merge']['status'] == 'ready'
    expected = local + [doc('new-author')]
    assert plan['sources'] == expected
    assert plan['incoming_sources'] == env['initial'] + [doc('new-author')]
    imported = accept(env, plan)
    assert imported['revision'] == 2 and current_documents(env) == expected
    assert chats.get('recipient', old['id'])['source_corpus_id'] == local_corpus['id']
    assert chats.get('recipient', old['id'])['character_revision'] == 1
    fresh = chats.create('recipient', imported['id'], 'after author update')
    assert fresh['source_corpus_id'] != local_corpus['id']
    copied = env['sources'].get('recipient', fresh['source_corpus_id'])
    assert copied['sources'] == expected and copied['index_status'] == 'pending' and copied['index_receipt_json'] is None
    assert accept(env, plan)['revision'] == 2  # Lost response retry doesn't duplicate versions.
    with pytest.raises(CharacterNotFoundError):
        env['sources'].get('author', copied['id'])
    # The next baseline must be the author's last publication, not the merged local corpus.
    publish_author(env, env['initial'] + [doc('new-author'), doc('newer-author')])
    second = preview(env)
    assert second['source_merge']['status'] == 'ready'
    accept(env, second)
    assert current_documents(env) == expected + [doc('newer-author')]
    assert chats.get('recipient', old['id'])['source_corpus_id'] == local_corpus['id']


@pytest.mark.parametrize('mode', ['modify_modify', 'delete_modify', 'modify_delete', 'id_collision'])
def test_api_rejects_conflicted_import_without_overwriting_any_state(env, mode):
    if mode == 'modify_modify':
        local, remote = [doc('a', 'local')], [doc('a', 'author')]
    elif mode == 'delete_modify':
        local, remote = [doc('local')], [doc('a', 'author')]
    elif mode == 'modify_delete':
        local, remote = [doc('a', 'local')], []
    else:
        local, remote = env['initial'] + [doc('same-id')], env['initial'] + [doc('same-id')]
    local_edit(env, local)
    publish_author(env, remote)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[repository] = lambda: env['library']
    app.dependency_overrides[get_current_user] = lambda: {'user_id': 'recipient'}
    url = f"/api/character-library/{env['recipient']['id']}/source-update"
    with TestClient(app) as client:
        plan = client.get(url).json()
        assert plan['source_merge']['status'] == 'conflict'
        result = client.post(url, json={k: plan[k] for k in ('revision', 'source_revision', 'publication_id', 'local_corpus_id')})
        assert result.status_code == 409 and '不能自动覆盖' in result.json()['detail']
        assert env['library'].get('recipient', env['recipient']['id'])['revision'] == 1
        assert current_documents(env) == local
        app.dependency_overrides[get_current_user] = lambda: {'user_id': 'author'}
        assert client.get(url).status_code == 404
        assert client.post(url, json={k: plan[k] for k in ('revision', 'source_revision', 'publication_id', 'local_corpus_id')}).status_code == 404


@pytest.mark.parametrize('reason', ['missing_pointer', 'missing_row', 'foreign_character', 'revoked', 'corrupt'])
def test_unavailable_baseline_never_reconstructed_or_exposed(env, reason):
    local_edit(env, [doc('a')])  # Remove the baseline-only secret from visible recipient content.
    publish_author(env, [doc('a'), doc('new')])
    db, library = env['db'], env['library']
    if reason == 'missing_pointer':
        db.execute_sql('UPDATE character_library SET source_publication_id=NULL WHERE id=:id', {'id': env['recipient']['id']})
    elif reason == 'missing_row':
        db.execute_sql('DELETE FROM character_publications WHERE id=:id', {'id': env['publication']['publication_id']})
    elif reason == 'foreign_character':
        other = library.create('other', CharacterSettings(name='private', persona='x', greeting='x'))
        other_corpus = env['sources'].publish('other', other['id'], 1, [doc('secret', 'FOREIGN_SECRET')])
        publication = library.publish('other', other['id'], 1, True, other_corpus['id'], ['secret'])
        db.execute_sql('UPDATE character_library SET source_publication_id=:publication WHERE id=:id',
                       {'id': env['recipient']['id'], 'publication': publication['publication_id']})
    elif reason == 'revoked':
        env['sources'].revoke('author', env['corpus']['id'])
    else:
        db.execute_sql('UPDATE character_publications SET sources_json=:payload WHERE id=:id',
                       {'id': env['publication']['publication_id'], 'payload': json.dumps([doc('a', 'CORRUPT_SECRET')])})
    plan = preview(env)
    assert plan['source_merge']['status'] == 'blocked' and plan['source_merge']['reason'] == 'baseline_unavailable'
    assert all(secret not in json.dumps(plan) for secret in ('BASE_ONLY_SECRET', 'FOREIGN_SECRET', 'CORRUPT_SECRET'))
    with pytest.raises(SourceMergeBlockedError):
        accept(env, plan)
    assert current_documents(env) == [doc('a')]


def test_no_sources_baseline_is_known_empty_and_preserves_local_additions(env):
    library = env['library']
    published = publish_author(env, [])
    recipient = library.copy_shared('recipient-2', published['share_token'], published['publication_id'], 'empty-copy')
    env['sources'].publish('recipient-2', recipient['id'], 1, [doc('local')])
    publish_author(env, [doc('remote')])
    plan = library.source_update('recipient-2', recipient['id'])
    assert plan['source_merge']['status'] == 'ready' and plan['sources'] == [doc('local'), doc('remote')]


@pytest.mark.parametrize('change', ['local', 'author', 'baseline_revoke', 'close_share'])
def test_changes_after_preview_are_rejected(env, change):
    local_edit(env, [doc('a'), doc('local')])
    publish_author(env, env['initial'] + [doc('remote')])
    plan = preview(env)
    if change == 'local':
        local_edit(env, [doc('a', 'new local')])
    elif change == 'author':
        publish_author(env, [doc('a', 'new author')])
    elif change == 'baseline_revoke':
        env['sources'].revoke('author', env['corpus']['id'])
    else:
        env['library'].publish('author', env['author']['id'], 1, False)
    with pytest.raises((RevisionConflictError, SourceMergeBlockedError)):
        accept(env, plan)
    assert env['library'].get('recipient', env['recipient']['id'])['revision'] == 1


def test_baseline_revocation_during_transaction_rolls_back_everything(env, monkeypatch):
    publish_author(env, env['initial'] + [doc('new')])
    plan = preview(env)
    commit = env['db'].revision_transaction
    injected = False

    def revoke_then_commit(statements):
        nonlocal injected
        if not injected and any('UPDATE character_publications SET id=id' in sql for sql, _, _ in statements):
            injected = True
            env['sources'].revoke('author', env['corpus']['id'])
        return commit(statements)

    monkeypatch.setattr(env['db'], 'revision_transaction', revoke_then_commit)
    with pytest.raises(RevisionConflictError):
        accept(env, plan)
    assert env['library'].get('recipient', env['recipient']['id'])['revision'] == 1
    assert env['sources'].list('recipient', env['recipient']['id'], 1)
    assert env['db'].execute_sql('SELECT * FROM character_library_versions WHERE character_id=:id AND revision=2',
                                {'id': env['recipient']['id']}) == []


def test_capacity_failure_rejects_import_before_any_writes(env):
    local_edit(env, env['initial'] + [doc(f'local-{i}') for i in range(50)])
    publish_author(env, env['initial'] + [doc(f'author-{i}') for i in range(50)])
    plan = preview(env)
    assert plan['source_merge']['reason'] == 'merged_source_capacity_exceeded'
    with pytest.raises(SourceMergeBlockedError):
        accept(env, plan)
    assert env['library'].get('recipient', env['recipient']['id'])['revision'] == 1


def test_recipient_deleted_all_sources_stays_deleted_when_author_only_adds_other_source(env):
    library, sources = env['library'], env['sources']
    current = sources.list('recipient', env['recipient']['id'], 1)[0]
    sources.revoke('recipient', current['id'])
    publish_author(env, env['initial'] + [doc('new')])
    plan = preview(env)
    assert plan['local_corpus_id'] is None and plan['baseline_publication_id'] == env['publication']['publication_id']
    assert plan['sources'] == [doc('new')]
    accept(env, plan)
    assert current_documents(env) == [doc('new')]
    assert library.get('recipient', env['recipient']['id'])['revision'] == 2


def test_author_share_closure_does_not_expose_baseline_plan(env):
    local_edit(env, [doc('a')])
    publish_author(env, [doc('a'), doc('new')])
    env['library'].publish('author', env['author']['id'], 1, False)
    plan = preview(env)
    assert not plan['available'] and plan['baseline_publication_id'] is None
    assert plan['source_merge']['status'] == 'blocked'
    assert 'BASE_ONLY_SECRET' not in json.dumps(plan)


def test_missing_baseline_http_returns_explicit_resolution_error(env):
    env['db'].execute_sql('UPDATE character_library SET source_publication_id=NULL WHERE id=:id', {'id': env['recipient']['id']})
    publish_author(env, env['initial'] + [doc('new')])
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[repository] = lambda: env['library']
    app.dependency_overrides[get_current_user] = lambda: {'user_id': 'recipient'}
    url = f"/api/character-library/{env['recipient']['id']}/source-update"
    with TestClient(app) as client:
        plan = client.get(url).json()
        assert plan['baseline_publication_id'] is None and plan['source_merge']['reason'] == 'baseline_unavailable'
        result = client.post(url, json={k: plan[k] for k in ('revision', 'source_revision', 'publication_id', 'local_corpus_id')})
        assert result.status_code == 409 and '基线不可用' in result.json()['detail']
