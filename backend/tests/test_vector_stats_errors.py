"""Vector status reports real dependency failures without exposing private data."""

from knowledge.index_lifecycle import default_knowledge_index as index_lifecycle
from app.runtime import RuntimeContainer

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from api import knowledge
from app import dependencies
from knowledge import vector_db as vectors


@pytest.mark.parametrize('stage', ['dependency', 'initialize', 'statistics'])
async def test_vector_stats_failure_has_safe_detail_and_original_cause(monkeypatch, caplog, stage):
    failure = ImportError('synthetic-private-vector-location') if stage == 'dependency' else OSError('synthetic-private-vector-location')
    if stage == 'dependency':
        monkeypatch.setattr(index_lifecycle, '_vector_provider', Mock(side_effect=failure))
    else:
        vector = SimpleNamespace(get_stats=Mock(side_effect=failure))
        getter = Mock(side_effect=failure) if stage == 'initialize' else Mock(return_value=vector)
        monkeypatch.setattr(vectors, 'get_vector_db', getter)
    with pytest.raises(HTTPException) as caught:
        await knowledge.get_vector_stats({'role': 'admin'}, database=index_lifecycle.db)
    assert caught.value.status_code == 503
    assert caught.value.__cause__ is failure
    assert '向量数据库' in caught.value.detail
    assert 'synthetic-private' not in str(caught.value.detail) + caplog.text
    assert type(failure).__name__ in caplog.text


async def test_empty_vector_statistics_remain_a_successful_business_result(monkeypatch):
    stats = {'total_documents': 0, 'index_size': 0, 'index_type': 'flat', 'bm25_corpus_size': 0}
    vector = SimpleNamespace(get_stats=Mock(return_value=stats))
    getter = Mock(return_value=vector)
    monkeypatch.setattr(vectors, 'get_vector_db', getter)
    assert await knowledge.get_vector_stats({'role': 'admin'}, database=index_lifecycle.db) == {'success': True, 'stats': stats}
    getter.assert_called_once_with()
    vector.get_stats.assert_called_once_with()


@pytest.mark.parametrize('stage', ['initialize', 'statistics'])
async def test_vector_stats_cancellation_is_not_converted_into_service_failure(monkeypatch, stage):
    failure = asyncio.CancelledError()
    vector = SimpleNamespace(get_stats=Mock(side_effect=failure))
    getter = Mock(side_effect=failure) if stage == 'initialize' else Mock(return_value=vector)
    monkeypatch.setattr(vectors, 'get_vector_db', getter)
    with pytest.raises(asyncio.CancelledError) as caught:
        await knowledge.get_vector_stats({'role': 'admin'}, database=index_lifecycle.db)
    assert caught.value is failure


@pytest.mark.parametrize('identity,expected', [(None, 401), ('member', 403), ('admin', 200), ('admin_failure', 503)])
async def test_vector_stats_http_permission_and_failure_contract(monkeypatch, identity, expected):
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(db=index_lifecycle.db, is_pg_mode=lambda: False, startup_env={})
    app.include_router(knowledge.router)
    if identity is not None:

        async def verified_user(request):
            return {'username': 'synthetic-vector-observer', 'role': 'user' if identity == 'member' else 'admin', 'auth_type': 'managed_api_key'}
        monkeypatch.setattr(dependencies, 'get_current_user', verified_user)
    stats = {'total_documents': 0, 'index_size': 0}
    getter = Mock(side_effect=OSError('synthetic-private-vector-location')) if identity == 'admin_failure' else Mock(return_value=SimpleNamespace(get_stats=Mock(return_value=stats)))
    monkeypatch.setattr(vectors, 'get_vector_db', getter)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://isolated.test') as client:
        response = await client.get('/api/vector/stats')
    assert response.status_code == expected
    assert 'synthetic-private' not in response.text
    if expected in {401, 403}:
        getter.assert_not_called()
    elif expected == 200:
        assert response.json() == {'success': True, 'stats': stats}
        getter.assert_called_once_with()
    else:
        assert 'success' not in response.json() and 'detail' in response.json()
        getter.assert_called_once_with()
