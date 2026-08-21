"""Regression tests for unified sessions, caching, RAG, and inference admission."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _message(*, adapter: str, created_at: str) -> dict:
    return {
        "sessionType": "group",
        "conversationType": "group",
        "sessionId": "qq:group:room-1",
        "sessionName": f"Room via {adapter}",
        "platform": "qq",
        "adapter": adapter,
        "conversationId": "room-1",
        "senderId": "u1",
        "senderName": "Alice",
        "userId": "u1",
        "userName": "Alice",
        "message": f"message-{adapter}",
        "reply": "reply",
        "createdAt": created_at,
    }


def test_fastapi_app_attaches_runtime_and_business_routes():
    from app.main import app
    from app.runtime import RuntimeContainer, get_runtime_container

    assert isinstance(get_runtime_container(app), RuntimeContainer)
    paths = set(app.openapi()["paths"])
    assert "/api/generate" in paths
    assert "/api/auth/login" in paths
    assert "/api/integrations/astrbot/messages" in paths

@pytest.mark.asyncio
async def test_readiness_uses_runtime_container_database(monkeypatch):
    from app import main
    from app.readiness import ReadinessProbe
    from app.runtime import RuntimeContainer, get_runtime_container

    vector_db = ModuleType("knowledge.vector_db")
    vector_db.get_vector_db = lambda: object()
    monkeypatch.setitem(sys.modules, "knowledge.vector_db", vector_db)

    calls = []

    class Database:
        def execute_sql(self, query):
            calls.append(query)
            return [{"ok": 1}]

    original_container = main.app.state.runtime_container
    main.app.state.runtime_container = RuntimeContainer(
        db=Database(),
        is_pg_mode=lambda: False,
        startup_env={},
    )
    # 替换 readiness_probe 为 model_not_required，避免依赖 create_app() 时的环境变量
    original_probe = main.app.state.readiness_probe
    main.app.state.readiness_probe = ReadinessProbe(
        database_check=lambda: get_runtime_container(main.app).db.execute_sql("SELECT 1"),
        model_required=False,
    )
    try:
        result = await main.readiness_check(SimpleNamespace(app=main.app))
    finally:
        main.app.state.runtime_container = original_container
        main.app.state.readiness_probe = original_probe

    assert result["status"] == "ready"
    assert calls == ["SELECT 1"]

def test_session_identity_isolates_type_and_deduplicates_adapters(tmp_path):
    from db.database import SQLiteDB

    db = SQLiteDB(tmp_path / "sessions.db")
    db.add_message(_message(adapter="napcat-a", created_at="2026-01-01T00:00:00"))
    db.add_message(_message(adapter="napcat-b", created_at="2026-01-01T00:00:01"))

    summaries = db.get_session_summaries()
    assert len(summaries) == 1
    assert summaries[0]["conversationId"] == "room-1"
    assert summaries[0]["messageCount"] == 2
    assert summaries[0]["adapter"] == "napcat-b"

    db.set_session_bot_enabled("same", False, "qq", "same", "private")
    db.set_session_bot_enabled("same", True, "qq", "same", "group")
    assert db.is_session_bot_enabled("same", "qq", "same", "private") is False
    assert db.is_session_bot_enabled("same", "qq", "same", "group") is True


def test_legacy_session_switch_migrates_to_conversations(tmp_path):
    from db.database import SQLiteDB

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE session_settings ("
        "sessionId TEXT PRIMARY KEY, sessionType TEXT, sessionName TEXT, "
        "bot_enabled INTEGER, updated_at TEXT)"
    )
    conn.execute(
        "INSERT INTO session_settings VALUES (?, ?, ?, ?, ?)",
        ("legacy-room", "group", "Legacy Room", 0, "2026-01-01T00:00:00"),
    )
    conn.commit()
    conn.close()

    db = SQLiteDB(path)
    assert db.is_session_bot_enabled(
        "legacy-room", "qq", "legacy-room", "group"
    ) is False
    remaining = db.execute_sql(
        "SELECT COUNT(*) AS count FROM sqlite_master WHERE type='table' AND name='session_settings'"
    )
    assert remaining[0]["count"] == 0


def test_sqlite_normalizes_legacy_redundant_indexes(tmp_path):
    from db.database import SQLiteDB

    path = tmp_path / "indexes.db"
    db = SQLiteDB(path)
    db.execute_sql(
        "CREATE INDEX IF NOT EXISTS idx_messages_createdAt ON messages(createdAt)"
    )
    db.execute_sql(
        "CREATE INDEX IF NOT EXISTS idx_conversations_platform_conversation "
        "ON conversations(platform, conversationId, conversationType)"
    )
    db.close_connection()

    migrated = SQLiteDB(path)
    connection = migrated.get_connection()
    message_indexes = {
        row["name"] for row in connection.execute("PRAGMA index_list(messages)")
    }
    conversation_indexes = {
        row["name"]
        for row in connection.execute("PRAGMA index_list(conversations)")
    }

    assert "idx_messages_created_at" in message_indexes
    assert "idx_messages_createdAt" not in message_indexes
    assert "idx_conversations_platform_conversation" not in conversation_indexes

def test_orm_metadata_declares_runtime_indexes():
    from db.models import metadata

    expected = {
        "audit_logs": {"idx_audit_logs_timestamp"},
        "training_tasks": {
            "idx_training_tasks_task_id",
            "idx_training_tasks_lora_name",
            "idx_training_tasks_status",
        },
    }
    for table_name, index_names in expected.items():
        declared = {index.name for index in metadata.tables[table_name].indexes}
        assert index_names <= declared


def test_sqlite_runtime_schema_matches_orm_metadata(tmp_path):
    from db.database import SQLiteDB
    from db.models import metadata

    database = SQLiteDB(tmp_path / "schema-contract.db")
    connection = database.get_connection()

    for table_name, table in metadata.tables.items():
        actual_columns = {
            row["name"]
            for row in connection.execute(f'PRAGMA table_info("{table_name}")')
        }
        assert actual_columns == set(table.columns.keys()), table_name

    database.close_connection()


def test_sqlite_assigns_exactly_one_admin_under_concurrent_registration(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    from db.database import SQLiteDB

    database = SQLiteDB(tmp_path / "users.db")
    barrier = threading.Barrier(2)

    def create(username: str):
        barrier.wait(timeout=2)
        return database.add_user(username, "hash")

    with ThreadPoolExecutor(max_workers=2) as executor:
        users = list(executor.map(create, ("alice", "bob")))

    assert sorted(user["role"] for user in users) == ["admin", "user"]


def test_postgres_serializes_first_admin_with_transaction_lock():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "db" / "pg_database.py").read_text(
        encoding="utf-8"
    )
    method = source.split("async def add_user", 1)[1].split(
        "async def get_user", 1
    )[0]

    assert "pg_advisory_xact_lock" in method
    assert method.index("pg_advisory_xact_lock") < method.index("count_stmt")

def test_bounded_ttl_cache_expires_and_evicts(monkeypatch):
    from cache import ttl_value_cache

    now = [10.0]
    monkeypatch.setattr(ttl_value_cache.time, "monotonic", lambda: now[0])
    cache = ttl_value_cache.BoundedTTLCache[str, bool](ttl=5, max_size=2)
    cache.set("a", False)
    cache.set("b", True)
    assert cache.get("a") is False

    cache.set("c", True)
    assert cache.get("b") is None
    assert len(cache) == 2

    now[0] = 16.0
    assert cache.get("a") is None
    assert cache.get("c") is None


@pytest.mark.asyncio
async def test_inference_runtime_serializes_sessions_and_restarts(monkeypatch):
    from infra.concurrency_control import InferenceRuntime

    monkeypatch.setenv("INFERENCE_WORKERS", "2")
    monkeypatch.setenv("INFERENCE_QUEUE_MAX_SIZE", "8")
    runtime = InferenceRuntime()
    active = 0
    max_active = 0

    async def work(value: int) -> int:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return value

    values = await asyncio.gather(
        runtime.submit(lambda: work(1), session_id="same", priority=10),
        runtime.submit(lambda: work(2), session_id="same", priority=10),
    )
    assert values == [1, 2]
    assert max_active == 1

    await runtime.shutdown()
    assert await runtime.submit(lambda: work(3), session_id="same", priority=10) == 3
    await runtime.shutdown()


def test_inference_runtime_restarts_on_a_fresh_event_loop(monkeypatch):
    from infra.concurrency_control import InferenceRuntime

    monkeypatch.setenv("INFERENCE_WORKERS", "1")
    monkeypatch.setenv("INFERENCE_QUEUE_MAX_SIZE", "4")
    runtime = InferenceRuntime()

    async def first_lifecycle():
        assert await runtime.submit(
            lambda: asyncio.sleep(0, result="first"),
            session_id="same",
            priority=10,
        ) == "first"
        await asyncio.sleep(0.01)
        await runtime.shutdown()

    async def second_lifecycle():
        assert await runtime.submit(
            lambda: asyncio.sleep(0, result="second"),
            session_id="same",
            priority=10,
        ) == "second"
        await asyncio.sleep(0.01)
        assert await asyncio.wait_for(
            runtime.submit(
                lambda: asyncio.sleep(0, result="third"),
                session_id="same",
                priority=10,
            ),
            timeout=0.5,
        ) == "third"
        await runtime.shutdown()

    asyncio.run(first_lifecycle())
    asyncio.run(second_lifecycle())


@pytest.mark.asyncio
async def test_composite_rate_limit_refunds_earlier_scopes(monkeypatch):
    from infra.concurrency_control import InferenceRuntime, RateLimitExceeded

    monkeypatch.setenv("CHAT_GLOBAL_QPS", "0")
    monkeypatch.setenv("CHAT_GLOBAL_BURST", "1")
    monkeypatch.setenv("CHAT_CONVERSATION_QPS", "0")
    monkeypatch.setenv("CHAT_CONVERSATION_BURST", "1")
    monkeypatch.setenv("CHAT_SENDER_QPS", "0")
    monkeypatch.setenv("CHAT_SENDER_BURST", "1")
    runtime = InferenceRuntime()

    allowed, _ = await runtime.sender_limiter.acquire("qq:blocked")
    assert allowed is True
    with pytest.raises(RateLimitExceeded) as exc_info:
        await runtime.check_rate_limits("qq", "room", "blocked")
    assert exc_info.value.scope == "sender"

    await runtime.check_rate_limits("qq", "room", "other")


@pytest.mark.asyncio
async def test_vllm_generation_reuses_one_rag_result(monkeypatch):
    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector, rag_helper

    calls = 0
    captured = {}

    async def retrieve(query, top_k, filters):
        nonlocal calls
        calls += 1
        return {
            "results": [{"title": "source", "content": "evidence", "score": 0.9}],
            "citations": [{"source_title": "source"}],
            "confidence": 0.9,
            "abstained": False,
        }

    class Client:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            return "answer"

    monkeypatch.setattr(intent_detector, "needs_rag", lambda _: (True, "test", None))
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "system")
    monkeypatch.setattr(generate, "_vllm_client", Client())
    monkeypatch.setattr(
        rag_helper,
        "get_rag_helper",
        lambda: SimpleNamespace(
            format_context_results=lambda results: results[0]["content"]
        ),
    )

    request = MessageRequest(message="question", sessionId="session")
    reply, used_rag, meta = await generate._generate_with_vllm(
        request,
        None,
        runtime_config={"useKnowledgeBase": True},
    )

    assert reply == "answer"
    assert used_rag is True
    assert calls == 1
    assert meta["citations"] == [{"source_title": "source"}]
    assert "evidence" in captured["messages"][-1]["content"]
    assert '<retrieved_evidence trust="untrusted"' in captured["messages"][-1]["content"]
    assert "<user_query>" in captured["messages"][-1]["content"]
    assert "【事实与安全边界】" in captured["messages"][0]["content"]
    assert "【检索证据约束】" in captured["messages"][0]["content"]


@pytest.mark.asyncio
async def test_vllm_generation_omits_rag_policy_without_evidence(monkeypatch):
    from api import generate
    from db.schemas import MessageRequest

    captured = {}

    class Client:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            return "answer"

    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "persona")
    monkeypatch.setattr(generate, "_vllm_client", Client())

    reply, used_rag, meta = await generate._generate_with_vllm(
        MessageRequest(message="ordinary chat", sessionId="session"),
        None,
        runtime_config={"useKnowledgeBase": False},
    )

    assert reply == "answer"
    assert used_rag is False
    assert meta == {}
    assert "persona" in captured["messages"][0]["content"]
    assert "【事实与安全边界】" in captured["messages"][0]["content"]
    assert "【检索证据约束】" not in captured["messages"][0]["content"]
    # 对话者昵称进入用户消息的不可信 speaker_label 区，不再进系统提示词；
    # 无检索证据时用户消息只含称呼参考与问题本身，不含 RAG 证据区
    last_content = captured["messages"][-1]["content"]
    assert "ordinary chat" in last_content
    assert '<speaker_label trust="untrusted"' in last_content
    assert "【检索证据】" not in last_content

@pytest.mark.asyncio
async def test_vllm_rag_abstention_skips_model_generation(monkeypatch):
    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector

    async def retrieve(query, top_k, filters):
        return {
            "results": [],
            "citations": [],
            "confidence": 0.1,
            "abstained": True,
        }

    class Client:
        async def generate(self, **kwargs):
            raise AssertionError("abstained RAG requests must not call the model")

    monkeypatch.setattr(intent_detector, "needs_rag", lambda _: (True, "test", None))
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "system")
    monkeypatch.setattr(generate, "_vllm_client", Client())
    monkeypatch.setattr(generate, "_RAG_ABSTENTION_REPLY", "insufficient evidence")

    reply, used_rag, meta = await generate._generate_with_vllm(
        MessageRequest(message="unknown question", sessionId="session"),
        None,
        runtime_config={"useKnowledgeBase": True},
    )

    assert reply == "insufficient evidence"
    assert used_rag is True
    assert meta["abstained"] is True
    assert meta["modelInvoked"] is False

@pytest.mark.asyncio
async def test_inference_timeout_cancels_model_task_without_killing_worker(monkeypatch):
    from infra.concurrency_control import InferenceRuntime

    monkeypatch.setenv("INFERENCE_WORKERS", "1")
    runtime = InferenceRuntime()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_work():
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    request = asyncio.create_task(
        runtime.submit(slow_work, session_id="slow", priority=10, timeout=0.02)
    )
    await started.wait()
    with pytest.raises(asyncio.TimeoutError):
        await request
    await asyncio.wait_for(cancelled.wait(), timeout=1)

    assert await runtime.submit(
        lambda: asyncio.sleep(0, result="ok"),
        session_id="next",
        priority=10,
    ) == "ok"
    await runtime.shutdown()

@pytest.mark.asyncio
async def test_duplicate_integration_message_does_not_emit_rate_limit_reply(monkeypatch, tmp_path):
    from api import integrations
    from db.database import SQLiteDB

    class Request:
        async def body(self):
            return b"{}"

    class Runtime:
        async def check_rate_limits(self, *args):
            raise AssertionError("duplicates must be resolved before chat rate limiting")

    db = SQLiteDB(tmp_path / "integration.db")
    assert db.mark_integration_message_processed("qq", "napcat", "duplicate-1")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("ASTRBOT_ENABLED", "true")
    monkeypatch.setenv("ASTRBOT_INTEGRATION_TOKEN", "test-token")
    monkeypatch.setenv("INTEGRATION_SIGNATURE_REQUIRED", "false")
    monkeypatch.setattr(integrations, "db", db)
    monkeypatch.setattr(integrations, "inference_runtime", Runtime())

    payload = integrations.AstrBotMessageRequest(
        platform="qq",
        adapter="napcat",
        messageId="duplicate-1",
        conversationId="room",
        conversationType="private",
        senderId="sender",
        text="hello",
    )
    response = await integrations.receive_astrbot_message(
        payload,
        Request(),
        x_integration_token="test-token",
    )

    assert response.shouldReply is False
    assert response.model == "duplicate"

@pytest.mark.asyncio
async def test_enhanced_cache_endpoints_use_async_cache_contract(monkeypatch):
    from api import enhanced

    class Cache:
        removed_pattern = None

        @property
        def stats(self):
            return {"size": 2}

        async def invalidate(self, pattern=None):
            self.removed_pattern = pattern
            return 2

    async def no_vllm_stats():
        return None

    cache = Cache()
    monkeypatch.setattr(enhanced, "response_cache", cache)
    monkeypatch.setattr(enhanced, "_get_vllm_load_balancer_stats", no_vllm_stats)
    monkeypatch.setattr(enhanced, "connection_pool", lambda: None)
    monkeypatch.setattr(enhanced, "http_client_pool", lambda: None)
    monkeypatch.setattr(enhanced, "backup_mgr", lambda: None)
    monkeypatch.setattr(enhanced, "failover_mgr", lambda: None)
    monkeypatch.setattr(enhanced, "access_control_mgr", lambda: None)
    monkeypatch.setattr(enhanced, "circuit_breaker_registry", None)

    stats = await enhanced.get_enhanced_stats(current_user={})
    cleared = await enhanced.invalidate_cache("prefix", current_user={})

    assert stats["stats"]["responseCache"] == {"size": 2}
    assert cleared["removed"] == 2
    assert cache.removed_pattern == "prefix"
@pytest.mark.asyncio
async def test_close_resource_offloads_synchronous_cleanup():
    import threading

    from app import main

    event_loop_thread = threading.get_ident()
    close_threads: list[int] = []

    class Resource:
        def close(self):
            close_threads.append(threading.get_ident())

    await main._close_resource("test resource", Resource(), "close")

    assert close_threads
    assert close_threads[0] != event_loop_thread


def test_lifespan_cleanup_is_guarded_by_finally():
    import ast
    import inspect
    import textwrap

    from app import main

    tree = ast.parse(textwrap.dedent(inspect.getsource(main.lifespan)))
    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and node.finalbody
        and any(isinstance(child, (ast.Yield, ast.YieldFrom)) for child in ast.walk(node))
    ]
    assert guarded, "lifespan shutdown must run from a finally block"


def test_integration_tokens_are_environment_only(monkeypatch):
    from api import integrations
    from api.config import _mask_config

    monkeypatch.setenv("ASTRBOT_INTEGRATION_TOKEN", "active-token")
    monkeypatch.setenv("ASTRBOT_INTEGRATION_TOKENS", "previous-token, next-token")

    assert integrations._allowed_tokens() == [
        "active-token",
        "previous-token",
        "next-token",
    ]
    masked = _mask_config({
        "astrbotIntegrationToken": "legacy-secret",
        "astrbotIntegrationTokens": "legacy-secret-2",
        "ASTRBOTINTEGRATIONTOKEN": "case-variant",
        "temperature": 0.7,
    })
    assert masked == {"temperature": 0.7}


@pytest.mark.asyncio
async def test_config_api_rejects_integration_tokens():
    from api import config
    from fastapi import HTTPException

    class Request:
        async def json(self):
            return {"astrbotIntegrationToken": "must-not-enter-database"}

    with pytest.raises(HTTPException) as exc_info:
        await config.update_config(Request(), current_user={"role": "admin"})

    assert exc_info.value.status_code == 422
@pytest.mark.asyncio
async def test_config_api_rejects_non_object_payload():
    from api import config
    from fastapi import HTTPException

    class Request:
        async def json(self):
            return ["not", "an", "object"]

    with pytest.raises(HTTPException) as exc_info:
        await config.update_config(Request(), current_user={"role": "admin"})

    assert exc_info.value.status_code == 422

@pytest.mark.asyncio
async def test_vllm_initialization_retries_after_transient_failure(monkeypatch):
    from api import generate
    from app import config
    from inference import vllm_client

    attempts = 0
    expected_client = object()

    async def get_shared_client():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("vLLM is starting")
        return expected_client

    monkeypatch.setattr(config, "is_vllm_enabled", lambda: True)
    monkeypatch.setattr(vllm_client, "get_vllm_client", get_shared_client)
    monkeypatch.setattr(generate, "_vllm_client", None)
    monkeypatch.setattr(generate, "_vllm_initialized", False)
    monkeypatch.setattr(generate, "_vllm_init_lock", None)
    monkeypatch.setattr(generate, "_vllm_init_lock_loop", None)

    assert await generate._ensure_vllm() is False
    assert generate._vllm_initialized is False
    assert await generate._ensure_vllm() is True
    assert generate._vllm_client is expected_client
    assert attempts == 2


def test_generate_module_locks_restart_on_a_fresh_event_loop(monkeypatch):
    from api import generate

    monkeypatch.setattr(generate, "_vllm_init_lock", None)
    monkeypatch.setattr(generate, "_vllm_init_lock_loop", None)
    monkeypatch.setattr(generate, "_local_model_lock", None)
    monkeypatch.setattr(generate, "_local_model_lock_loop", None)

    async def use_locks():
        vllm_lock = generate._loop_local_lock("vllm")
        local_lock = generate._loop_local_lock("local_model")
        async with vllm_lock:
            pass
        async with local_lock:
            pass
        return vllm_lock, local_lock

    first_vllm, first_local = asyncio.run(use_locks())
    second_vllm, second_local = asyncio.run(use_locks())

    assert second_vllm is not first_vllm
    assert second_local is not first_local

def test_shared_vllm_client_restarts_on_a_fresh_event_loop(monkeypatch):
    from app import config
    from inference import vllm_client

    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            created.append(self)

        async def close(self):
            self.closed = True

    monkeypatch.setattr(config, "get_vllm_served_model_name", lambda: "test-model")
    monkeypatch.setattr(vllm_client, "VLLMClient", FakeClient)
    monkeypatch.setattr(vllm_client, "_shared_client", None)
    monkeypatch.setattr(vllm_client, "_shared_client_lock", None)
    monkeypatch.setattr(vllm_client, "_shared_client_lock_loop", None)

    async def lifecycle():
        client = await vllm_client.get_vllm_client()
        assert await vllm_client.get_vllm_client() is client
        await vllm_client.close_shared_vllm_client()
        assert client.closed is True
        return client

    first = asyncio.run(lifecycle())
    second = asyncio.run(lifecycle())

    assert second is not first
    assert len(created) == 2

def test_lora_status_lock_restarts_on_a_fresh_event_loop(monkeypatch):
    from api import loras

    monkeypatch.setattr(loras, "_lora_status_lock", None)
    monkeypatch.setattr(loras, "_lora_status_lock_loop", None)

    async def use_lock():
        lock = loras._get_lora_status_lock()
        async with lock:
            pass
        return lock

    first = asyncio.run(use_lock())
    second = asyncio.run(use_lock())

    assert second is not first

def test_app_config_async_controls_restart_on_a_fresh_event_loop(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "_llm_semaphore", None)
    monkeypatch.setattr(config, "_llm_semaphore_loop", None)
    monkeypatch.setattr(config, "_generation_state_lock", None)
    monkeypatch.setattr(config, "_generation_state_lock_loop", None)

    async def use_controls():
        semaphore = config.get_llm_semaphore()
        state_lock = config.get_generation_state_lock()
        async with semaphore:
            async with state_lock:
                pass
        return semaphore, state_lock

    first_semaphore, first_lock = asyncio.run(use_controls())
    second_semaphore, second_lock = asyncio.run(use_controls())

    assert second_semaphore is not first_semaphore
    assert second_lock is not first_lock

def test_response_cache_restarts_on_a_fresh_event_loop():
    from cache.response_cache import ResponseCache

    cache = ResponseCache(default_ttl=60)
    prompt_hash = cache.compute_prompt_hash("hello")
    cache_key = cache.build_cache_key(model_name="model")

    async def use_cache(write: bool):
        if write:
            await cache.set(prompt_hash, cache_key, "world")
        value = await cache.get(prompt_hash, cache_key)
        return value, cache._get_lock()

    first_value, first_lock = asyncio.run(use_cache(True))
    second_value, second_lock = asyncio.run(use_cache(False))

    assert first_value == "world"
    assert second_value == "world"
    assert second_lock is not first_lock
