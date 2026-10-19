"""Adapter management and authorization share one application's database."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI

from api import router as compatibility
from app.config import create_access_token
from app.runtime import RuntimeContainer
from db.database import SQLiteDB


@pytest.fixture
def stores(tmp_path, monkeypatch):
    from db import database as directory
    from db.adapter import db as global_db
    from inference import adapter_checker

    databases = [SQLiteDB(tmp_path / f"application-{i}.sqlite") for i in range(2)]
    for index, database in enumerate(databases):
        database.add_user("synthetic-owner", "unused-hash-for-signed-token")
        database.execute_sql(
            "INSERT INTO config(key,value) VALUES ('lora_router_config',:value) "
            "ON CONFLICT(key) DO UPDATE SET value=:value",
            {"value": json.dumps(dict(enabled=False, mode='manual', default_adapter=f'owner-{index}'))},
        )
    forbidden = Mock(side_effect=AssertionError("Request must not access the global database"))
    monkeypatch.setattr(global_db, 'execute_sql', forbidden)
    monkeypatch.setattr(compatibility, 'db', SimpleNamespace(execute_sql=forbidden), raising=False)
    monkeypatch.setattr(directory, 'refresh_lora_dir_map', lambda: {'fixture': '/fiction/adapter'})
    check = Mock(return_value=SimpleNamespace(compatible=True, checks={'rank': True}, warnings=[], errors=[]))
    monkeypatch.setattr(adapter_checker, 'AdapterChecker', lambda: SimpleNamespace(check_adapter=check))
    try:
        yield SimpleNamespace(databases=databases, forbidden=forbidden, check=check)
    finally:
        for database in databases:
            database.close_connection()


def client(database):
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(db=database, is_pg_mode=lambda: False, startup_env={})
    app.include_router(compatibility.router)
    token = create_access_token('synthetic-owner', 1, 'admin')
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='https://isolated',
                             headers={'Authorization': 'Bearer ' + token})


async def test_two_applications_keep_config_and_check_reports_in_their_own_store(stores, monkeypatch):
    from inference import lora_router

    unused_factory = Mock(side_effect=RuntimeError("Offline research router must not participate in config writes"))
    monkeypatch.setattr(lora_router, 'get_lora_router', unused_factory)
    for index, database in enumerate(stores.databases):
        async with client(database) as api:
            response = await api.get('/api/router/config')
            assert response.status_code == 200 and response.json()['config']['default_adapter'] == f'owner-{index}'
            before = await api.get('/api/router/adapters')
            assert before.status_code == 200 and before.json()['adapters'][0]['compatibility'] is None
            checked = await api.post('/api/router/check/fixture')
            assert checked.status_code == 200 and checked.json()['checks'] == {'rank': True}
            listed = await api.get('/api/router/adapters')
            assert listed.status_code == 200 and listed.json()['adapters'][0]['compatibility']['compatible'] is True
            updated = await api.put('/api/router/config', json={'default_adapter': f'updated-{index}'})
            assert updated.status_code == 200 and updated.json()['config']['default_adapter'] == f'updated-{index}'
        assert database.execute_sql('SELECT count(*) AS n FROM adapter_compatibility')[0]['n'] == 1
    for index, database in enumerate(stores.databases):
        value = database.execute_sql("SELECT value FROM config WHERE key='lora_router_config'")[0]['value']
        assert json.loads(value)['default_adapter'] == f'updated-{index}'
    stores.forbidden.assert_not_called()
    assert stores.check.call_count == 2
    unused_factory.assert_not_called()


@pytest.mark.parametrize('operation', ['list', 'record'])
async def test_bound_store_failure_is_reported_without_another_store(stores, monkeypatch, operation):
    database = stores.databases[0]
    reader = Mock(side_effect=OSError('synthetic-private-bound-store-fault'))
    monkeypatch.setattr(database, 'execute_sql', reader)
    async with client(database) as api:
        response = await api.get('/api/router/adapters') if operation == 'list' else await api.post('/api/router/check/fixture')
    assert response.status_code == 503 and 'synthetic-private' not in response.text
    if operation == 'record':
        assert response.json()['detail']['checkCompleted'] is True
        assert response.json()['detail']['recordSaved'] is None
    reader.assert_called_once()
    stores.forbidden.assert_not_called()


async def test_request_authentication_and_current_role_still_gate_adapter_writes(stores):
    database = stores.databases[0]
    async with client(database) as api:
        api.headers.pop('Authorization')
        assert (await api.get('/api/router/config')).status_code == 401
        api.headers['Authorization'] = 'Bearer ' + create_access_token('synthetic-owner', 1, 'admin')
        database.execute_sql("UPDATE users SET role='user' WHERE username='synthetic-owner'")
        assert (await api.post('/api/router/check/fixture')).status_code == 403
    stores.check.assert_not_called()
    stores.forbidden.assert_not_called()
