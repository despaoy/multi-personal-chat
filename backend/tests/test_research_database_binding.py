"""Research data and history sampling are owned by the current application."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI

from api import preferences, retrieval_eval
from app.config import create_access_token
from app.runtime import RuntimeContainer
from db.database import SQLiteDB


def prompt(owner):
    return f'完整资料 {owner}：文档42规定课程2026-12-20开课，需12-18 19:00前确认，问安排。'


def pair(owner):
    return dict(prompt=prompt(owner), chosen='2026-12-20开课，12-18 19:00前确认。', rejected='无需确认。',
                rubric={'factuality': 1.0}, metadata={'owner': owner}, review_status='approved')


@pytest.fixture
def stores(tmp_path, monkeypatch):
    from db.adapter import db as global_db
    from knowledge import rag_helper

    databases = [SQLiteDB(tmp_path / f'owner-{i}.sqlite') for i in range(2)]
    for i, database in enumerate(databases):
        database.add_user('synthetic-owner', 'unused-signed-token-hash')
        data = pair(str(i))
        database.execute_sql(
            "INSERT INTO preference_pairs(id,prompt,chosen,rejected,rubric,annotator,metadata,review_status,created_at) "
            "VALUES ('shared',:prompt,:chosen,:rejected,:rubric,'seed',:metadata,'approved','2026-10-17')",
            dict(prompt=data['prompt'], chosen=data['chosen'], rejected=data['rejected'],
                 rubric=json.dumps(data['rubric']), metadata=json.dumps(data['metadata'])),
        )
        database.execute_sql(
            "INSERT INTO retrieval_eval_questions(id,question,expected_doc_ids,expected_doc_titles,gold_answer,category,created_at) "
            "VALUES ('shared',:question,'[\"42\"]','[]','2026-12-20','complete','2026-10-17')", {'question': prompt(str(i))},
        )
        database.add_message(dict(sessionId='fixture-session', message=prompt(str(i)), reply='完整课程安排2026-12-20，12-18 19:00前确认。',
                                  modelName='synthetic', loraName='default', traceId='owner-'+str(i)))
    forbidden = Mock(side_effect=AssertionError('Research request used the global store'))
    poison = SimpleNamespace(execute_sql=forbidden, execute_sql_insert=forbidden)
    for module in [preferences, retrieval_eval]:
        monkeypatch.setattr(module, 'db', poison, raising=False)
    monkeypatch.setattr(global_db, 'execute_sql', forbidden)
    monkeypatch.setattr(global_db, 'execute_sql_insert', forbidden)
    retrieve = Mock(return_value=[dict(document_id=42, title='课程安排', content='2026-12-20开课。')])
    monkeypatch.setattr(rag_helper, 'get_rag_helper', lambda: SimpleNamespace(retrieve_context=retrieve))
    try:
        yield SimpleNamespace(databases=databases, forbidden=forbidden, retrieve=retrieve)
    finally:
        for database in databases:
            database.close_connection()


def client(database):
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(db=database, is_pg_mode=lambda: False, startup_env={})
    app.include_router(preferences.router)
    app.include_router(retrieval_eval.router)
    token = create_access_token('synthetic-owner', 1, 'admin')
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='https://isolated',
                             headers={'Authorization': 'Bearer ' + token})


async def test_two_applications_keep_complete_research_workflows_in_their_own_store(stores):
    for i, database in enumerate(stores.databases):
        async with client(database) as api:
            listed = await api.get('/api/preferences/')
            assert listed.status_code == 200 and listed.json()['preferences'][0]['metadata']['owner'] == str(i)
            created = await api.post('/api/preferences/', json=pair('new-'+str(i)))
            assert created.status_code == 200
            pid = created.json()['id']
            assert (await api.put('/api/preferences/shared', json={'annotator': 'review-'+str(i)})).status_code == 200
            exported = await api.post('/api/preferences/export', json={'review_status': 'approved', 'limit': 10})
            assert exported.status_code == 200 and {row['prompt'] for row in exported.json()['data']} == {prompt(str(i)), prompt('new-'+str(i))}
            sampled = await api.post('/api/preferences/sample-from-history', json={'session_id': 'fixture-session', 'limit': 1, 'min_length': 10})
            assert sampled.status_code == 200 and sampled.json()['candidates'][0]['trace_id'] == 'owner-'+str(i)
            questions = await api.get('/api/retrieval-eval/questions')
            assert questions.status_code == 200 and questions.json()['questions'][0]['question'] == prompt(str(i))
            qid = 'created-'+str(i)
            created_q = await api.post('/api/retrieval-eval/questions', json=dict(id=qid, question=prompt('new-'+str(i)), expected_doc_ids=['42']))
            assert created_q.status_code == 200
            evaluated = await api.post('/api/retrieval-eval/run')
            assert evaluated.status_code == 200 and evaluated.json()['metrics']['avg_recall_at_k'] == 1
            assert {row['question'] for row in evaluated.json()['metrics']['per_question']} == {prompt(str(i)), prompt('new-'+str(i))}
            assert (await api.delete('/api/preferences/'+pid)).status_code == 200
            assert (await api.delete('/api/retrieval-eval/questions/'+qid)).status_code == 200
        assert database.execute_sql('SELECT count(*) AS n FROM preference_pairs')[0]['n'] == 1
        assert database.execute_sql('SELECT count(*) AS n FROM retrieval_eval_questions')[0]['n'] == 1
    assert stores.retrieve.call_count == 4
    stores.forbidden.assert_not_called()


@pytest.mark.parametrize('operation', ['read', 'insert'])
async def test_bound_store_failure_never_uses_global_store(stores, monkeypatch, operation):
    database = stores.databases[0]
    method = 'execute_sql' if operation == 'read' else 'execute_sql_insert'
    writer = Mock(side_effect=OSError('synthetic-bound-store-down'))
    monkeypatch.setattr(database, method, writer)
    async with client(database) as api:
        response = await api.get('/api/preferences/') if operation == 'read' else await api.post(
            '/api/retrieval-eval/questions', json=dict(id='new', question=prompt('new'), expected_doc_ids=['42']))
    assert response.status_code == 500
    writer.assert_called_once()
    stores.forbidden.assert_not_called()


async def test_anonymous_and_demoted_users_cannot_read_or_write_research_data(stores):
    database = stores.databases[0]
    async with client(database) as api:
        api.headers.pop('Authorization')
        assert (await api.get('/api/preferences/')).status_code == 401
        assert (await api.get('/api/retrieval-eval/questions')).status_code == 401
        api.headers['Authorization'] = 'Bearer ' + create_access_token('synthetic-owner', 1, 'admin')
        database.execute_sql("UPDATE users SET role='user' WHERE username='synthetic-owner'")
        assert (await api.post('/api/preferences/', json=pair('denied'))).status_code == 403
        assert (await api.delete('/api/retrieval-eval/questions/shared')).status_code == 403
    stores.retrieve.assert_not_called()
    stores.forbidden.assert_not_called()
