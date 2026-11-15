from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.character_library import original_sources, router
from app.dependencies import get_current_user
from character.library import CharacterSettings
from db.database import SQLiteDB
from repositories.character_chat import CharacterChatRepository
from repositories.character_sources import CharacterSourceRepository


def test_source_api_authorization_and_immutable_session_binding(tmp_path):
    db = SQLiteDB(tmp_path / 'source-api.db')
    try:
        sources = CharacterSourceRepository(db)
        chats = CharacterChatRepository(db)
        char = sources.characters.create('owner', CharacterSettings(name='旅人', persona='谨慎', greeting='你好'))
        other = sources.characters.create('owner', CharacterSettings(name='另一个', persona='谨慎', greeting='你好'))
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: {'user_id': 'owner'}
        app.dependency_overrides[original_sources] = lambda: sources
        path = f'/api/character-library/{char["id"]}/sources'
        with TestClient(app) as client:
            payload = {'character_revision': 1, 'sources': [{'id': 'chapter1', 'title': '第一章', 'content': '原文第一版'}]}
            first_response = client.post(path, json=payload)
            assert first_response.status_code == 201
            first = first_response.json()
            retry_path = path + '/' + first['id'] + '/retry'
            assert client.post(retry_path, json={'attempt': 1}).status_code == 409
            sources.set_index_job_status('owner', first['id'], 1, 'failed')
            app.dependency_overrides[get_current_user] = lambda: {'user_id': 'other-account'}
            assert client.post(retry_path, json={'attempt': 1}).status_code == 404
            app.dependency_overrides[get_current_user] = lambda: {'user_id': 'owner'}
            assert client.post(f'/api/character-library/{other["id"]}/sources/{first["id"]}/retry', json={'attempt': 1}).status_code == 404
            retried = client.post(retry_path, json={'attempt': 1})
            assert retried.status_code == 200 and retried.json()['index_attempt'] == 2
            assert client.post(retry_path, json={'attempt': 1}).status_code == 409
            old = chats.create('owner', char['id'], '旧会话')
            assert old['source_corpus_id'] == first['id']
            turn = chats.commit_turn(old, 'test', '原作如何？', '回答', 'controlled')
            payload['sources'][0]['content'] = '原文第二版'
            second = client.post(path, json=payload).json()
            fresh = chats.create('owner', char['id'], '新会话')
            assert fresh['source_corpus_id'] == second['id']
            assert chats.get('owner', old['id'])['source_corpus_id'] == first['id']
            for turn_id in (None, turn['id']):
                child = chats.create('owner', char['id'], '延续或分支', source_session_id=old['id'], source_revision=2, source_turn_id=turn_id)
                assert child['source_corpus_id'] == first['id']
            versions = client.get(path, params={'revision': 1}).json()['corpora']
            assert {v['id'] for v in versions} == {first['id'], second['id']}
            assert all('sources_json' not in v and 'sources' not in v for v in versions)
            assert client.get(f'/api/character-library/{other["id"]}/sources/{first["id"]}').status_code == 404
            app.dependency_overrides[get_current_user] = lambda: {'user_id': 'other-account'}
            assert client.get(path, params={'revision': 1}).status_code == 404
            assert client.get(path + '/' + first['id']).status_code == 404
            assert client.delete(path + '/' + first['id']).status_code == 404
            assert client.post(path, json=payload).status_code == 404
            app.dependency_overrides[get_current_user] = lambda: {'user_id': 'owner'}
            assert client.delete(path + '/' + first['id']).status_code == 200
            assert client.get(path + '/' + first['id']).status_code == 404
            assert chats.get('owner', old['id'])['source_corpus_id'] == first['id']  # Never silently switch editions.
            payload['sources'] *= 2
            assert client.post(path, json=payload).status_code == 422
    finally:
        db.close_connection()
