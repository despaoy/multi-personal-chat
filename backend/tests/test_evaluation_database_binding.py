"""Evaluation requests and queued persistence share the application's store."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI

from api import evaluation
from app.config import create_access_token
from app.runtime import RuntimeContainer
from db.database import SQLiteDB
from evaluation import runtime_runner


@pytest.fixture
def stores(tmp_path, monkeypatch):
    databases = [SQLiteDB(tmp_path / f'evaluation-{i}.sqlite') for i in range(2)]
    for i, database in enumerate(databases):
        database.add_user('synthetic-admin', 'unused-signed-token-hash')
        database.execute_sql(
            "INSERT INTO gold_eval_runs(id,run_at,model_label,total_prompts,category_breakdown,metrics,config_snapshot,notes) "
            "VALUES ('shared','2026-10-18',:label,1,'{\"complete\":1}',:metrics,'{\"mock\":false}','completed')",
            {'label': str(i), 'metrics': json.dumps({'owner': str(i)})},
        )
        database.execute_sql_insert("INSERT INTO feedback(trace_id,rating,detail,created_at) VALUES (:trace,'positive',:detail,'2026-10-18')",
            {'trace': 'trace-' + str(i), 'detail': '课程12月20日开课，12月18日19点前确认；完整回复已核对。来源=' + str(i)})
    from db.adapter import db as global_database
    forbidden = Mock(side_effect=AssertionError('evaluation used the global store'))
    monkeypatch.setattr(evaluation, 'db', SimpleNamespace(execute_sql=forbidden, execute_sql_insert=forbidden), raising=False)
    monkeypatch.setattr(global_database, 'execute_sql', forbidden)
    monkeypatch.setattr(global_database, 'execute_sql_insert', forbidden)
    try:
        yield SimpleNamespace(databases=databases, forbidden=forbidden)
    finally:
        for database in databases:
            database.close_connection()


def client(database):
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(db=database, is_pg_mode=lambda: False, startup_env={})
    app.include_router(evaluation.router)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='https://isolated',
        headers={'Authorization': 'Bearer ' + create_access_token('synthetic-admin', 1, 'admin')})


async def test_two_applications_own_records_feedback_and_queued_completion(stores, monkeypatch):
    tasks = []
    scheduled_stores = []
    schedule = runtime_runner.schedule_generation_evaluation

    async def controlled_worker(run_id, options, database):
        await asyncio.to_thread(runtime_runner._update_run, database, run_id,
            metrics={'controlled': True, 'mock': False}, total=1, breakdown={'complete': 1}, note='completed')

    def tracked_schedule(run_id, options, database):
        scheduled_stores.append(database)
        task = schedule(run_id, options, database)
        tasks.append(task)
        return task

    monkeypatch.setattr(runtime_runner, 'execute_generation_evaluation', controlled_worker)
    monkeypatch.setattr(runtime_runner, 'schedule_generation_evaluation', tracked_schedule)
    try:
        for i, database in enumerate(stores.databases):
            async with client(database) as api:
                listed = await api.get('/api/evaluation/runs')
                detail = await api.get('/api/evaluation/runs/shared')
                assert listed.status_code == detail.status_code == 200
                assert listed.json()['runs'][0]['model_label'] == detail.json()['run']['metrics']['owner'] == str(i)
                created = await api.post('/api/feedback', json={'rating': 'positive', 'trace_id': 'new-' + str(i)})
                assert created.status_code == 200
                queued = await api.post('/api/evaluation/run', json={'mock': False, 'max_prompts': 1})
                assert queued.status_code == 200 and queued.json()['status'] == 'queued'
                await tasks[-1]
                completed = await api.get('/api/evaluation/runs/' + queued.json()['run_id'])
                assert completed.status_code == 200 and completed.json()['run']['notes'] == 'completed'
                assert completed.json()['run']['metrics'] == {'controlled': True, 'mock': False}
                feedback = await api.get('/api/feedback')
                assert feedback.status_code == 200 and {r['trace_id'] for r in feedback.json()['feedbacks']} == {'trace-' + str(i), 'new-' + str(i)}
        assert scheduled_stores == stores.databases
        await asyncio.sleep(0)
        assert not runtime_runner._evaluation_tasks
        stores.forbidden.assert_not_called()
    finally:
        await asyncio.gather(*tasks)


@pytest.mark.parametrize('operation', ['read', 'insert'])
async def test_bound_evaluation_database_fault_cannot_use_global_store(stores, monkeypatch, operation):
    database = stores.databases[0]
    schedule = Mock(side_effect=AssertionError('failed insert must not schedule'))
    monkeypatch.setattr(runtime_runner, 'schedule_generation_evaluation', schedule)
    method = 'execute_sql' if operation == 'read' else 'execute_sql_insert'
    monkeypatch.setattr(database, method, Mock(side_effect=OSError('synthetic-bound-database-down')))
    async with client(database) as api:
        response = await api.get('/api/evaluation/runs') if operation == 'read' else await api.post('/api/evaluation/run', json={'mock': False})
    assert response.status_code == (500 if operation == 'read' else 503)
    schedule.assert_not_called()
    stores.forbidden.assert_not_called()


async def test_scheduler_failure_marks_the_application_record_failed(stores, monkeypatch):
    database = stores.databases[0]
    monkeypatch.setattr(evaluation.secrets, 'token_hex', lambda *args: 'failed-schedule')
    monkeypatch.setattr(runtime_runner, 'schedule_generation_evaluation', Mock(side_effect=RuntimeError('scheduler-down')))
    async with client(database) as api:
        response = await api.post('/api/evaluation/run', json={'mock': False})
    assert response.status_code == 500
    record = database.execute_sql("SELECT notes,metrics FROM gold_eval_runs WHERE id='eval_failed-schedule'")[0]
    assert record['notes'] == 'failed' and json.loads(record['metrics'])['error'] == 'schedule_failed'
    assert stores.databases[1].execute_sql("SELECT id FROM gold_eval_runs WHERE id='eval_failed-schedule'") == []
    stores.forbidden.assert_not_called()


async def test_evaluation_requires_current_admin_but_feedback_allows_authenticated_user(stores):
    database = stores.databases[0]
    async with client(database) as api:
        token = api.headers.pop('Authorization')
        assert (await api.get('/api/evaluation/runs')).status_code == 401
        assert (await api.post('/api/feedback', json={'rating': 'positive'})).status_code == 401
        api.headers['Authorization'] = token
        database.execute_sql("UPDATE users SET role='user' WHERE username='synthetic-admin'")
        assert (await api.get('/api/feedback')).status_code == 403
        assert (await api.post('/api/evaluation/run', json={})).status_code == 403
        assert (await api.post('/api/feedback', json={'rating': 'positive'})).status_code == 200
    stores.forbidden.assert_not_called()
