"""A settings patch cannot persist invalid generation numbers or silently skip them."""

from app.runtime import RuntimeContainer

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from api import config
from inference.generation_options import read_generation_options


class Database:
    def __init__(self):
        self.values = {"temperature": "old-invalid", "maxTokens": "old-invalid", "topP": "old-invalid"}
        self.writes = []
        self.reads = 0

    def update_config(self, update):
        self.writes.append(dict(update))
        self.values.update(update)

    @property
    def config(self):
        self.reads += 1
        return dict(self.values)


@pytest.fixture
def rig(monkeypatch):
    owned = Database()
    invalidate = Mock()
    monkeypatch.setattr(config, 'invalidate_config_cache', invalidate)
    return SimpleNamespace(owned=owned, invalidate=invalidate)


def request(body):
    return SimpleNamespace(json=AsyncMock(return_value=body))


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value', [('temperature', 'synthetic-private-invalid'), ('temperature', True), ('temperature', float('nan')), ('temperature', -1), ('maxTokens', 0), ('maxTokens', 64.5), ('maxTokens', None), ('maxTokens', '****'), ('topP', 0), ('topP', 1.1)])
async def test_invalid_numeric_patch_has_no_side_effect_or_private_error(rig, caplog, field, value):
    with pytest.raises(HTTPException) as caught:
        await config.update_config(request({field: value}), current_user={'role': 'admin'}, database=rig.owned)
    assert caught.value.status_code == 422 and isinstance(caught.value.__cause__, ValidationError)
    assert caught.value.detail['fields'] == [field]
    assert 'synthetic-private' not in str(caught.value.detail) + str(caught.value.__cause__) + caplog.text
    assert rig.owned.writes == [] and rig.owned.reads == 0
    rig.invalidate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value,expected', [('temperature', '0.2', 0.2), ('maxTokens', '1536', 1536), ('topP', '0.75', 0.75)])
async def test_only_modified_numeric_field_is_validated_and_normalized(rig, monkeypatch, field, value, expected):
    for name in ['VLLM_TEMPERATURE', 'VLLM_MAX_TOKENS', 'VLLM_TOP_P']:
        monkeypatch.setenv(name, 'synthetic-invalid-environment')
    original = dict(rig.owned.values)
    result = await config.update_config(request({field: value}), current_user={'role': 'admin'}, database=rig.owned)
    assert result['success'] and rig.owned.writes == [{field: expected}]
    assert result['config'][field] == expected and rig.owned.reads == 1
    assert all((rig.owned.values[key] == original[key] for key in original if key != field))
    rig.invalidate.assert_called_once()


@pytest.mark.asyncio
async def test_unrelated_patch_does_not_fill_defaults_or_validate_old_numeric_settings(rig, monkeypatch):
    for name in ['VLLM_TEMPERATURE', 'VLLM_MAX_TOKENS', 'VLLM_TOP_P']:
        monkeypatch.setenv(name, 'synthetic-invalid-environment')
    original = dict(rig.owned.values)
    result = await config.update_config(request({'replyDelay': 2}), current_user={'role': 'admin'}, database=rig.owned)
    assert result['success'] and rig.owned.writes == [{'replyDelay': 2}]
    assert all((rig.owned.values[key] == value for key, value in original.items()))


@pytest.mark.asyncio
async def test_http_error_names_field_and_does_not_echo_private_value(rig):
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(db=rig.owned, is_pg_mode=lambda: False, startup_env={})
    app.include_router(config.router)
    app.dependency_overrides[config.get_current_admin] = lambda: {'role': 'admin'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://unit') as client:
        response = await client.put('/api/config', json={'maxTokens': 'synthetic-private-invalid'})
    assert response.status_code == 422 and response.json()['detail']['fields'] == ['maxTokens']
    assert 'synthetic-private' not in response.text and rig.owned.writes == []


@pytest.mark.asyncio
async def test_saved_numbers_are_consumed_by_same_generation_contract(rig, monkeypatch, tmp_path):
    from db.database import SQLiteDB
    owned = SQLiteDB(tmp_path / 'canonical-settings.sqlite')
    try:
        body = {'temperature': '0.25', 'maxTokens': '1536', 'topP': '1.0'}
        result = await config.update_config(request(body), current_user={'role': 'admin'}, database=owned)
        assert result['success']
        resolved = read_generation_options(owned.config)
        assert resolved == {'temperature': 0.25, 'maxTokens': 1536, 'topP': 1.0}
        assert result['config']['maxTokens'] == 1536
    finally:
        owned.close_connection()
