"""Required startup resources either start together or release their allocations."""

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.parametrize(
    "dependency", ["resource_pool", "backup_manager", "access_control", "circuit_breaker", "encryption"]
)
def test_missing_required_dependency_preserves_import_error(dependency, tmp_path):
    source = """import builtins,importlib,json,logging,sys
logging.disable(logging.CRITICAL)
original=builtins.__import__
failure=ImportError('synthetic required startup dependency unavailable')
def blocked(name,*args,**kwargs):
 if name=='infra.'+sys.argv[1]:raise failure
 return original(name,*args,**kwargs)
builtins.__import__=blocked
try:importlib.import_module('app.main')
except ImportError as error:print(json.dumps({'original_error':error is failure}))
else:print(json.dumps({'original_error':False}))
"""
    env = dict(
        os.environ,
        ENVIRONMENT="test",
        USE_POSTGRESQL="false",
        DATABASE_PATH=str(tmp_path / "isolated.sqlite"),
        DATABASE_URL="postgresql+asyncpg://unused@127.0.0.1:1/unused",
        JWT_SECRET="owned-startup-synthetic-key-more-than-32-characters",
        ENCRYPTION_KEY=base64.urlsafe_b64encode(b"o" * 32).decode(),
    )
    result = subprocess.run(
        [sys.executable, "-c", source, dependency], env=env, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["original_error"] is True


@pytest.fixture
def lifecycle(monkeypatch, tmp_path):
    from app import main, config
    from app.runtime import RuntimeContainer
    from inference import model_manager

    events = []
    allocations = {}
    failure = RuntimeError("synthetic resource initialization failure")
    fail_at = [None]

    def allocate(kind, *args, **kwargs):
        if fail_at[0] == kind:
            raise failure
        resource = SimpleNamespace(args=args, kwargs=kwargs)

        async def close():
            events.append(kind)

        async def start():
            if fail_at[0] == "backup-start":
                raise failure

        resource.close = resource.stop_scheduled_backup = close
        resource.start_scheduled_backup = start
        allocations[kind] = resource
        return resource

    monkeypatch.setattr(main, "validate_or_raise_for_startup", lambda env: None)
    monkeypatch.setattr(main, "HttpClientPool", lambda *a, **k: allocate("http", *a, **k))
    monkeypatch.setattr(main, "ConnectionPool", lambda *a, **k: allocate("sqlite", *a, **k))
    monkeypatch.setattr(main, "BackupManager", lambda *a, **k: allocate("backup", *a, **k))
    monkeypatch.setattr(main, "AccessControlManager", lambda *a, **k: allocate("access", *a, **k))
    monkeypatch.setattr(model_manager, "_model_manager", None)
    getter = Mock(side_effect=AssertionError("Shutdown must not allocate a model manager"))
    monkeypatch.setattr(model_manager, "get_model_manager", getter)
    for module, name in [
        ("evaluation.runtime_runner", "shutdown_generation_evaluations"),
        ("training.task_manager", "shutdown_simple_lora_trainer"),
        ("api.knowledge", "shutdown_intent_tasks"),
        ("services.delivery_memory", "shutdown_delivery_memory"),
        ("services.turn_completion", "shutdown_turn_completions"),
        ("character.memory_llm", "shutdown_memory_enrichment"),
        ("api.generate", "close_vllm_client"),
        ("cache.redis_client", "close_async_redis"),
    ]:
        monkeypatch.setattr(module + "." + name, AsyncMock())
    monkeypatch.setattr("cache.redis_client.close_sync_redis", Mock())
    monkeypatch.setattr("cache.redis_client.health_check", lambda: True)
    monkeypatch.setattr("services.turn_completion.start_turn_completions", Mock())
    worker = Mock()
    monkeypatch.setattr("services.delivery_memory.ensure_delivery_memory_worker", worker)
    monkeypatch.setattr(config, "circuit_breaker_registry", SimpleNamespace(clear=AsyncMock()))

    def application(*, postgres=False, backup_dir="", db_path=True):
        db = SimpleNamespace(execute_sql=Mock(), close=AsyncMock())
        if db_path:
            db.db_path = tmp_path / "actual.sqlite"
        runtime = RuntimeContainer(
            db=db,
            is_pg_mode=lambda: postgres,
            startup_env={"ENVIRONMENT": "test", "MEMORY_LLM_ENABLED": "false", "BACKUP_DIR": backup_dir},
        )
        return main.create_app(runtime), db

    return SimpleNamespace(
        main=main,
        config=config,
        events=events,
        allocations=allocations,
        failure=failure,
        fail_at=fail_at,
        application=application,
        getter=getter,
        worker=worker,
    )


@pytest.mark.parametrize(
    ("fail_at", "closed"),
    [
        ("http", []),
        ("sqlite", ["http"]),
        ("backup", ["sqlite", "http"]),
        ("backup-start", ["sqlite", "http", "backup"]),
        ("access", ["sqlite", "http", "backup"]),
        ("worker", ["sqlite", "http", "backup"]),
    ],
)
async def test_startup_error_propagates_and_closes_preceding_allocations(lifecycle, fail_at, closed):
    rig = lifecycle
    rig.fail_at[0] = fail_at
    if fail_at == "worker":
        rig.worker.side_effect = rig.failure
    app, _ = rig.application()
    with pytest.raises(RuntimeError) as raised:
        async with rig.main.lifespan(app):
            pytest.fail("Required resource failure admitted startup")
    assert raised.value is rig.failure
    assert rig.events == closed
    assert all(getattr(rig.config, name) is None for name in rig.main._LIFESPAN_RESOURCE_NAMES)
    rig.getter.assert_not_called()


@pytest.mark.parametrize("backup_dir", ["", "scheduled", "/tmp/owned-startup-test-backups"])
async def test_sqlite_resources_share_selected_database_and_close_on_exit(lifecycle, backup_dir):
    rig = lifecycle
    app, db = rig.application(backup_dir=backup_dir)
    async with rig.main.lifespan(app):
        path = db.db_path.resolve()
        expected_backup = Path(backup_dir) if backup_dir else path.parent / "backups"
        if not expected_backup.is_absolute():
            expected_backup = path.parent / expected_backup
        assert rig.allocations["sqlite"].args == (str(path),)
        assert rig.allocations["backup"].args == (str(path),)
        assert rig.allocations["backup"].kwargs["backup_dir"] == str(expected_backup)
        assert rig.allocations["access"].args == (db,)
    assert rig.events == ["sqlite", "http", "backup"]
    rig.getter.assert_not_called()


async def test_postgresql_needs_no_sqlite_path_pool_or_backup(lifecycle):
    rig = lifecycle
    app, db = rig.application(postgres=True, db_path=False)
    async with rig.main.lifespan(app):
        assert set(rig.allocations) == {"http", "access"}
        assert rig.config.connection_pool is rig.config.backup_mgr is None
    assert rig.events == ["http"]
    db.close.assert_awaited_once()
    rig.getter.assert_not_called()


async def test_missing_sqlite_path_errors_before_allocating_another_database(lifecycle):
    rig = lifecycle
    app, _ = rig.application(db_path=False)
    with pytest.raises(AttributeError, match="db_path"):
        async with rig.main.lifespan(app):
            pytest.fail("Missing database path admitted startup")
    assert rig.allocations == {}


async def test_shutdown_closes_existing_model_and_preserves_request_failure(lifecycle, monkeypatch):
    from inference import model_manager

    rig = lifecycle
    model = SimpleNamespace(shutdown=AsyncMock())
    monkeypatch.setattr(model_manager, "_model_manager", model)
    app, _ = rig.application()
    with pytest.raises(RuntimeError) as raised:
        async with rig.main.lifespan(app):
            raise rig.failure
    assert raised.value is rig.failure
    model.shutdown.assert_awaited_once()
    rig.getter.assert_not_called()
    assert rig.events == ["sqlite", "http", "backup"]
