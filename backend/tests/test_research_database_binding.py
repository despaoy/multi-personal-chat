"""Research data and history sampling are owned by the current application."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from api import preferences, retrieval_eval
from app.config import create_access_token
from app.main import create_app
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
    app = create_app(RuntimeContainer(db=database, is_pg_mode=lambda: False,
        startup_env={'ENVIRONMENT': 'test'}, model_check=lambda: {'loaded': True}))
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


@pytest.mark.parametrize('field', ['rubric', 'metadata'])
@pytest.mark.parametrize('stored', ['{broken-json', '', '[]', 'null'])
async def test_invalid_preference_objects_fail_before_success_or_export(stores, field, stored):
    database = stores.databases[0]
    database.execute_sql('UPDATE preference_pairs SET ' + field + '=:value', {'value': stored})
    async with client(database) as api:
        response = await api.get('/api/preferences/')
        assert response.status_code == 500 and response.json() == {'detail': '偏好数据操作失败'}
        if field == 'rubric':
            exported = await api.post('/api/preferences/export', json={})
            assert exported.status_code == 500 and exported.json() == {'detail': '偏好数据操作失败'}


@pytest.mark.parametrize('stored', [None, '{}'])
async def test_absent_optional_preference_objects_and_genuine_empty_results(stores, stored):
    database = stores.databases[0]
    database.execute_sql('UPDATE preference_pairs SET rubric=:value,metadata=:value', {'value': stored})
    async with client(database) as api:
        listed = await api.get('/api/preferences/')
        exported = await api.post('/api/preferences/export', json={})
        assert listed.status_code == exported.status_code == 200
        assert listed.json()['preferences'][0]['rubric'] == listed.json()['preferences'][0]['metadata'] == {}
        assert exported.json()['data'][0]['rubric'] == {}
        database.execute_sql('DELETE FROM preference_pairs')
        listed = await api.get('/api/preferences/')
        exported = await api.post('/api/preferences/export', json={})
        assert listed.status_code == exported.status_code == 200
        assert listed.json()['preferences'] == [] and listed.json()['total'] == 0
        assert exported.json()['data'] == [] and exported.json()['count'] == 0


async def test_preference_write_failure_keeps_cause_without_logging_content(stores, monkeypatch, caplog):
    from unittest.mock import AsyncMock

    from app import main

    database = stores.databases[0]
    error = OSError('synthetic-private-preference-content')
    monkeypatch.setattr(database, 'execute_sql_insert', Mock(side_effect=error))
    handled = AsyncMock(wraps=main.global_exception_handler)
    monkeypatch.setattr(main, 'global_exception_handler', handled)
    async with client(database) as api:
        response = await api.post('/api/preferences/', json=pair('complete'))
    assert response.status_code == 500 and handled.await_args.args[1] is error
    assert 'OSError' in caplog.text and str(error) not in caplog.text


@pytest.mark.parametrize('operation', ['list', 'create', 'update', 'delete', 'export', 'sample'])
async def test_preference_store_fault_uses_shared_handler_and_recovers(stores, monkeypatch, caplog, operation):
    import sqlite3
    from unittest.mock import AsyncMock

    from app.main import create_app, global_exception_handler

    database = stores.databases[0]
    app = create_app(RuntimeContainer(db=database, is_pg_mode=lambda: False,
        startup_env={'ENVIRONMENT': 'test'}, model_check=lambda: {'loaded': True}))
    handled = AsyncMock(wraps=global_exception_handler)
    app.add_exception_handler(sqlite3.Error, handled)
    marker = 'synthetic_private_preference_store_fault'
    original = database.execute_sql
    faults = []
    operations = {
        'list': ('execute_sql', 'GET', '/api/preferences/', None),
        'create': ('execute_sql_insert', 'POST', '/api/preferences/', pair('new-complete')),
        'update': ('execute_sql', 'PUT', '/api/preferences/shared', {'annotator': 'updated-reviewer'}),
        'delete': ('execute_sql', 'DELETE', '/api/preferences/shared', None),
        'export': ('execute_sql', 'POST', '/api/preferences/export', {'review_status': 'approved', 'limit': 10}),
        'sample': ('execute_sql', 'POST', '/api/preferences/sample-from-history',
                   {'session_id': 'fixture-session', 'limit': 1, 'min_length': 10}),
    }
    method, verb, url, body = operations[operation]

    def fail(*args, **kwargs):
        try:
            return original('SELECT ' + marker)
        except sqlite3.Error as exc:
            faults.append(exc)
            raise

    def snapshot():
        return {table: original('SELECT * FROM ' + table) for table in ['preference_pairs', 'messages']}

    before = snapshot()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='https://isolated',
        headers={'Authorization': 'Bearer ' + create_access_token('synthetic-owner', 1, 'admin')}) as api:
        with monkeypatch.context() as patch:
            patch.setattr(database, method, fail)
            failed = await api.request(verb, url, json=body)
            assert failed.status_code == 500 and failed.json()['success'] is False
            assert failed.json()['error'] == 'database_error'
            assert handled.await_args.args[1] is faults[-1]
            assert marker not in failed.text and marker not in caplog.text
            assert snapshot() == before
        repaired = await api.request(verb, url, json=body)
        assert repaired.status_code == 200 and repaired.json()['success'] is True
        result = repaired.json()
        if operation == 'create':
            assert original('SELECT prompt FROM preference_pairs WHERE id=:id',
                {'id': result['id']})[0]['prompt'] == prompt('new-complete')
        elif operation == 'update':
            assert original("SELECT annotator FROM preference_pairs WHERE id='shared'")[0]['annotator'] == 'updated-reviewer'
        elif operation == 'delete':
            assert original("SELECT * FROM preference_pairs WHERE id='shared'") == []
        elif operation == 'export':
            assert result['count'] == 1 and result['data'][0]['prompt'] == prompt('0')
        elif operation == 'sample':
            assert result['total'] == 1 and result['candidates'][0]['trace_id'] == 'owner-0'
        else:
            assert result['total'] == 1 and result['preferences'][0]['metadata'] == {'owner': '0'}
    stores.forbidden.assert_not_called()
