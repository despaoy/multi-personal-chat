"""Application resources stay owned across real authentication and inference."""

import asyncio
import gzip
import hashlib
import sqlite3
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from app import main
from app.runtime import RuntimeContainer
from db.database import SQLiteDB
from infra.access_control import AccessControlManager, Role
from infra.backup_manager import BackupManager
from infra.resource_pool import HttpClientPool

QUESTION = "完整合成资料：周二09:00开门、17:00闭馆，暴雨暂停；预约不能替代原件。请完整保留条件。"
ANSWER = "控制回复：周二09:00开门、17:00闭馆，暴雨暂停；预约不能替代原件。"


@pytest.fixture
async def applications(tmp_path, monkeypatch):
    from api import generate
    from services import chat_answer

    monkeypatch.setenv("MODEL_PROVIDER", "openai_compat")
    monkeypatch.setenv("VLLM_ENABLED", "false")
    monkeypatch.setenv("MODEL_STORAGE_PATH", str(tmp_path / "models"))
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "circuit_breaker_registry", None)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: asyncio.Semaphore(2))
    monkeypatch.setattr(chat_answer, "get_lora_system_prompt", lambda _: "仅依据完整题设，保留原始条件。")
    for module, name in [
        ("evaluation.runtime_runner", "shutdown_generation_evaluations"),
        ("training.task_manager", "shutdown_simple_lora_trainer"),
        ("api.knowledge", "shutdown_intent_tasks"),
        ("services.delivery_memory", "shutdown_delivery_memory"),
        ("services.turn_completion", "shutdown_turn_completions"),
        ("character.memory_llm", "shutdown_memory_enrichment"),
        ("inference.vllm_client", "close_shared_vllm_client"),
        ("cache.redis_client", "close_async_redis"),
    ]:
        monkeypatch.setattr(module + "." + name, AsyncMock())
    monkeypatch.setattr("services.turn_completion.start_turn_completions", Mock())
    monkeypatch.setattr("services.delivery_memory.ensure_delivery_memory_worker", Mock())
    pools, backups, managers, wire = [], [], [], []

    def pool_factory(**kwargs):
        index = len(pools)

        def respond(request):
            wire.append((index, request.url.host, json.loads(request.content)))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": ANSWER}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60},
                },
            )

        pool = HttpClientPool(**kwargs)
        pool._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        pools.append(pool)
        return pool

    def backup_factory(*args, **kwargs):
        backup = BackupManager(*args, **kwargs)
        backups.append(backup)
        return backup

    def access_factory(database):
        manager = AccessControlManager(database)
        managers.append(manager)
        return manager

    monkeypatch.setattr(main, "HttpClientPool", pool_factory)
    monkeypatch.setattr(main, "BackupManager", backup_factory)
    monkeypatch.setattr(main, "AccessControlManager", access_factory)
    databases = [SQLiteDB(tmp_path / f"application-{index}.sqlite") for index in range(2)]
    containers = []
    for index, database in enumerate(databases):
        database.update_config(
            {
                "temperature": 0.2 + index * 0.5,
                "maxTokens": 128,
                "topP": 1.0,
                "useKnowledgeBase": False,
                "openaiCompatBaseUrl": f"https://owner-{index}.invalid/v1",
                "openaiCompatModel": f"fixture-model-{index}",
                "openaiCompatApiKey": f"synthetic-owner-{index}-credential",
                "modelProvider": "openai_compat",
            }
        )
        runtime = SimpleNamespace(priority_for=lambda *args: 0, check_rate_limits=AsyncMock(), shutdown=AsyncMock())

        async def submit(job, **kwargs):
            return await job()

        runtime.submit = submit
        containers.append(
            RuntimeContainer(
                db=database,
                is_pg_mode=lambda: False,
                inference_runtime=runtime,
                startup_env={
                    "ENVIRONMENT": "test",
                    "MEMORY_LLM_ENABLED": "false",
                    "BACKUP_DIR": str(tmp_path / f"backup-{index}"),
                    "JWT_SECRET": "isolated-application-resource-key-at-least-32-characters",
                    "ENCRYPTION_KEY": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=",
                    "MODEL_PROVIDER": "openai_compat",
                    "OPENAI_COMPAT_API_KEY": "synthetic-application-resource-key",
                    "MULTIPERSONAL_BACKEND_URL": "http://127.0.0.1:8000",
                },
            )
        )
    apps = [main.create_app(container) for container in containers]
    keys = [None, None]

    @asynccontextmanager
    async def running(index):
        async with main.lifespan(apps[index]):
            keys[index] = managers[index].create_api_key(Role.ADMIN, description=f"synthetic-app-{index}-admin")[
                "api_key"
            ]
            yield apps[index]

    def client(index, key=None):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=apps[index], raise_app_exceptions=False),
            base_url="https://isolated",
            headers={"X-API-Key": keys[index] if key is None else key},
        )

    state = SimpleNamespace(
        databases=databases,
        containers=containers,
        apps=apps,
        keys=keys,
        pools=pools,
        backups=backups,
        managers=managers,
        wire=wire,
        running=running,
        client=client,
        root=tmp_path,
    )
    try:
        yield state
    finally:
        for backup in backups:
            if backup._running:
                await backup.stop_scheduled_backup()
        for pool in pools:
            if not pool.get_pool_stats()["closed"]:
                await pool.close()
        for database in databases:
            database.close_connection()


async def test_two_lifespans_keep_resources_and_close_their_own_tasks(applications):
    state = applications
    async with state.running(0):
        assert state.containers[0].http_client_pool is state.pools[0]
        async with state.running(1):
            assert state.containers[0].http_client_pool is state.pools[0]
            assert state.containers[1].http_client_pool is state.pools[1]
            assert state.containers[0].backup_mgr is state.backups[0]
            assert state.containers[1].access_control_mgr is state.managers[1]
        assert state.pools[1].get_pool_stats()["closed"] is True
        assert state.pools[0].get_pool_stats()["closed"] is False and state.backups[0]._running
        assert state.backups[1]._scheduled_task is None
    assert all(pool.get_pool_stats()["closed"] for pool in state.pools)
    assert all(backup._scheduled_task is None and not backup._running for backup in state.backups)
    assert all(
        container.http_client_pool is container.backup_mgr is container.access_control_mgr is None
        for container in state.containers
    )


async def test_real_managed_keys_and_same_key_ids_never_cross_application_databases(applications):
    state = applications
    async with state.running(0):
        async with state.running(1):
            async with state.client(0) as first, state.client(1) as second:
                assert (await first.get("/api/config")).status_code == 200
                assert (await second.get("/api/config")).status_code == 200
                for index, other in [(0, 1), (1, 0)]:
                    async with state.client(index, state.keys[other]) as crossed:
                        assert (await crossed.get("/api/config")).status_code == 401
                created = []
                for index, api in enumerate([first, second]):
                    response = await api.post(
                        "/api/enhanced/access-control/keys",
                        json={"role": "viewer", "description": f"application-{index}-viewer", "rate_limit": 60},
                    )
                    assert response.status_code == 200
                    created.append(response.json()["apiKey"]["api_key"])
                    listed = (await api.get("/api/enhanced/access-control/keys")).json()["keys"]
                    assert {row["description"] for row in listed} == {
                        f"synthetic-app-{index}-admin",
                        f"application-{index}-viewer",
                    }
                async with state.client(0, created[0]) as viewer:
                    assert (await viewer.get("/api/config")).status_code == 403
                assert (await first.delete("/api/enhanced/access-control/keys/2")).status_code == 200
                async with state.client(0, created[0]) as revoked:
                    assert (await revoked.get("/api/config")).status_code == 401
                async with state.client(1, created[1]) as unaffected:
                    assert (await unaffected.get("/api/config")).status_code == 403
        async with state.client(0) as first:
            assert (await first.get("/api/config")).status_code == 200


async def test_backup_api_uses_the_current_applications_directory_and_database(applications):
    state = applications
    async with state.running(0), state.running(1):
        for index in range(2):
            async with state.client(index) as api:
                response = await api.post("/api/enhanced/backups/create", params={"backup_type": "full"})
                assert response.status_code == 200 and response.json()["success"]
                path = Path(response.json()["path"]).resolve()
                assert path.is_relative_to((state.root / f"backup-{index}").resolve())
                assert path.is_file()
                body = gzip.decompress(path.read_bytes())
                assert hashlib.sha256(body).hexdigest() == response.json()["backup"]["sha256"]
                inspected = state.root / f"inspected-backup-{index}.sqlite"
                inspected.write_bytes(body)
                copied = sqlite3.connect(inspected.as_uri() + "?mode=ro", uri=True)
                try:
                    assert (
                        float(copied.execute("SELECT value FROM config WHERE key='temperature'").fetchone()[0])
                        == 0.2 + index * 0.5
                    )
                finally:
                    copied.close()
                listed = await api.get("/api/enhanced/backups")
                assert listed.status_code == 200 and any(
                    Path(row["path"]).resolve() == path for row in listed.json()["backups"]
                )


async def test_actual_chat_requests_borrow_only_their_application_pool(applications):
    state = applications

    async def generate(index, suffix):
        async with state.client(index) as api:
            response = await api.post(
                "/api/generate",
                json={"message": QUESTION, "loraName": "default", "traceId": f"synthetic-pool-{index}-{suffix}"},
            )
            assert response.status_code == 200 and response.json()["reply"] == ANSWER

    async with state.running(0):
        async with state.running(1):
            await generate(0, "first")
            await generate(1, "first")
        await generate(0, "second")
    assert [(pool, host) for pool, host, _ in state.wire] == [
        (0, "owner-0.invalid"),
        (1, "owner-1.invalid"),
        (0, "owner-0.invalid"),
    ]
    assert all(QUESTION in str(body["messages"]) for _, _, body in state.wire)
    assert [pool.get_pool_stats()["total_acquired"] for pool in state.pools] == [2, 1]


async def test_application_pool_failure_does_not_create_a_standalone_client(applications, monkeypatch):
    from inference.model_manager import get_model_manager
    from inference.request_context import resolve_request_context, use_request_context

    state = applications
    async with state.running(0):
        pool = state.pools[0]
        context = await resolve_request_context(get_model_manager(database=state.databases[0]), http_client_pool=pool)
        await pool.close()
        replacement = Mock(
            side_effect=AssertionError("An application pool failure must not create a replacement client")
        )
        monkeypatch.setattr(httpx, "AsyncClient", replacement)
        with use_request_context(context), pytest.raises(RuntimeError, match="已关闭"):
            await context.provider.async_complete(
                [{"role": "user", "content": QUESTION}], temperature=0.2, max_tokens=128
            )
        replacement.assert_not_called()
