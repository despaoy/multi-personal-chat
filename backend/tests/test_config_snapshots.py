"""Configuration snapshots never switch caches or refill old values after update."""

import concurrent.futures
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from api import config as config_api
from cache import config_cache
from db import adapter
from db.database import SQLiteDB
from inference import model_manager


@pytest.fixture(autouse=True)
def isolated_snapshot(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(config_cache, "_CONFIG_CACHE", None)
    monkeypatch.setattr(config_cache, "monotonic", lambda: clock[0])
    monkeypatch.setattr(config_cache, "_ttl_with_jitter", lambda ttl: ttl)

    return clock


def test_valid_snapshot_is_copied_and_expires_at_the_declared_ttl(isolated_snapshot):
    database = SimpleNamespace(config={"maxTokens": "733", "enabled": "false"})
    first, hit = config_cache.read_config(database)
    assert first == {"maxTokens": 733, "enabled": False} and not hit
    first["maxTokens"] = 1
    database.config["maxTokens"] = "377"
    assert config_cache.read_config(database) == ({"maxTokens": 733, "enabled": False}, True)
    isolated_snapshot[0] += 60
    assert config_cache.read_config(database) == ({"maxTokens": 377, "enabled": False}, False)


def test_expired_snapshot_never_masks_a_database_failure(isolated_snapshot):
    failure = OSError("fiction-config-unavailable")
    failed = False

    class Database:
        @property
        def config(self):
            if failed:
                raise failure
            return {"model": "stored-model"}

    database = Database()
    assert config_cache.read_config(database) == ({"model": "stored-model"}, False)
    failed = True
    isolated_snapshot[0] += 60
    for _ in range(2):
        with pytest.raises(OSError) as error:
            config_cache.read_config(database)
        assert error.value is failure


@pytest.mark.parametrize("stored", [None, [], "invalid"])
def test_malformed_database_result_cannot_be_cached(stored):
    database = SimpleNamespace(config=stored)
    with pytest.raises(TypeError, match="mapping"):
        config_cache.read_config(database)
    database.config = {"model": "valid-model"}
    assert config_cache.read_config(database) == ({"model": "valid-model"}, False)


def test_database_instances_cannot_share_each_others_values():
    first = SimpleNamespace(config={"model": "first-model"})
    second = SimpleNamespace(config={"model": "second-model"})
    assert config_cache.read_config(first)[0] == {"model": "first-model"}
    assert config_cache.read_config(second)[0] == {"model": "second-model"}
    assert config_cache.read_config(first)[0] == {"model": "first-model"}


def test_invalidation_waits_for_old_load_then_prevents_stale_refill():
    entered, release, invalidating = threading.Event(), threading.Event(), threading.Event()
    stored = {"model": "old-model"}

    class Database:
        def __init__(self):
            self.reads = 0

        @property
        def config(self):
            self.reads += 1
            captured = dict(stored)
            if self.reads == 1:
                entered.set()
                assert release.wait(2), "Test did not release the captured database read"
            return captured

    database = Database()

    def invalidate():
        invalidating.set()
        config_cache.invalidate_config_cache()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        loading = executor.submit(config_cache.read_config, database)
        assert entered.wait(2)
        stored["model"] = "new-model"
        clearing = executor.submit(invalidate)
        assert invalidating.wait(2)
        try:
            with pytest.raises(concurrent.futures.TimeoutError):
                clearing.result(timeout=.05)
        finally:
            release.set()
        assert loading.result(timeout=2) == ({"model": "old-model"}, False)
        assert clearing.result(timeout=2) is None
    assert config_cache.read_config(database) == ({"model": "new-model"}, False)
    assert database.reads == 2


@pytest.mark.asyncio
async def test_api_save_invalidates_actual_snapshot_used_by_the_model(monkeypatch, tmp_path):
    database = SQLiteDB(tmp_path / 'config.db')
    database.update_config({'openaiCompatBaseUrl': 'https://fiction.example', 'openaiCompatApiKey': 'fiction-private-key', 'openaiCompatModel': 'old-model'})
    monkeypatch.setattr(adapter, 'db', database)
    monkeypatch.setenv('MODEL_PROVIDER', 'openai_compat')
    first = await config_api.get_config(current_user={'isAdmin': True}, database=database)
    assert first['cached'] is False and 'fiction-private-key' not in str(first)
    assert (await config_api.get_config(current_user={'isAdmin': True}, database=database))['cached'] is True
    manager = model_manager.ModelManager(base_dir=tmp_path)
    monkeypatch.setattr(model_manager, 'get_model_manager', lambda *, database=None: manager)
    try:
        request = SimpleNamespace(json=AsyncMock(return_value={'openaiCompatModel': 'new-model'}))
        result = await config_api.update_config(request, current_user={'isAdmin': True}, database=database)
        assert result['success'] and manager.get_current_provider().model == 'new-model'
        assert model_manager._get_db_config()['openaiCompatModel'] == 'new-model'
        assert (await config_api.get_config(current_user={'isAdmin': True}, database=database))['config']['openaiCompatModel'] == 'new-model'
        assert first['config']['openaiCompatModel'] == 'old-model'
    finally:
        manager.shutdown()
        database.close_connection()


@pytest.mark.asyncio
async def test_config_api_read_failure_returns_error_without_empty_success(monkeypatch):
    failure = OSError('fiction-private-config-error')

    class Database:

        @property
        def config(self):
            raise failure
    database = Database()
    with pytest.raises(HTTPException) as error:
        await config_api.get_config(current_user={'isAdmin': True}, database=database)
    assert error.value.status_code == 503 and error.value.__cause__ is failure
    assert 'fiction-private' not in str(error.value.detail)
