"""Regression tests for unified sessions, caching, RAG, and inference admission."""

from __future__ import annotations

from app.runtime import RuntimeContainer
from infra.concurrency_control import inference_runtime

from infra.resource_pool import HttpClientPool

from services import chat_answer

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
    vector_db = ModuleType('knowledge.vector_db')
    vector_db.get_vector_db = lambda: object()
    monkeypatch.setitem(sys.modules, 'knowledge.vector_db', vector_db)
    calls = []

    class Database:

        def execute_sql(self, query):
            calls.append(query)
            return [{'ok': 1}]
    original_container = main.app.state.runtime_container
    main.app.state.runtime_container = RuntimeContainer(db=Database(), is_pg_mode=lambda: False, startup_env={}, http_client_pool=HttpClientPool(max_connections=1))
    original_probe = main.app.state.readiness_probe
    main.app.state.readiness_probe = ReadinessProbe(database_check=lambda: get_runtime_container(main.app).db.execute_sql('SELECT 1'), model_required=False)
    try:
        result = await main.readiness_check(SimpleNamespace(app=main.app))
    finally:
        main.app.state.runtime_container = original_container
        main.app.state.readiness_probe = original_probe
    assert result['status'] == 'ready'
    assert calls == ['SELECT 1']

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




def test_sqlite_assigns_exactly_one_admin_under_concurrent_registration(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from db.database import SQLiteDB

    database = SQLiteDB(tmp_path / "users.db")
    barrier = threading.Barrier(2)

    def create(username: str):
        barrier.wait(timeout=2)
        return database.add_user(username, "hash")

    with ThreadPoolExecutor(max_workers=2) as executor:
        users = list(executor.map(create, ("alice", "bob")))

    assert sorted(user["role"] for user in users) == ["admin", "user"]


@pytest.mark.asyncio
@pytest.mark.parametrize("existing,bootstrap_only", [(False, False), (True, False), (True, True)])
async def test_postgres_registration_executes_lock_before_decision_and_commit(monkeypatch, existing, bootstrap_only):
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/test")
    from db.errors import RegistrationClosedError
    from db.pg_database import PgDatabase

    events = []
    session = MagicMock()

    async def execute(statement):
        sql = str(statement)
        if "pg_advisory_xact_lock" in sql:
            assert not events
            events.append("lock")
            return None
        if sql.startswith("SELECT"):
            assert events == ["lock"]
            events.append("read")
            return SimpleNamespace(fetchone=lambda: {"id": 1} if existing else None)
        assert sql.startswith("INSERT") and events == ["lock", "read"]
        values = statement.compile().params
        assert values["username"] == "alice" and values["password_hash"] == "hash"
        assert values["role"] == ("user" if existing else "admin")
        events.append("write")
        return SimpleNamespace(inserted_primary_key=(2,))

    async def commit():
        assert events == ["lock", "read", "write"]
        events.append("commit")

    session.execute = AsyncMock(side_effect=execute)
    session.commit = AsyncMock(side_effect=commit)
    session.__aenter__.return_value = session
    database = object.__new__(PgDatabase)
    database.async_session = lambda: session
    if bootstrap_only and existing:
        with pytest.raises(RegistrationClosedError):
            await database.add_user("alice", "hash", bootstrap_only=True)
        assert events == ["lock", "read"]
        session.commit.assert_not_awaited()
    else:
        result = await database.add_user("alice", "hash", bootstrap_only=bootstrap_only)
        assert result["id"] == 2 and result["role"] == ("user" if existing else "admin")
        assert events == ["lock", "read", "write", "commit"]
    session.__aexit__.assert_awaited_once()


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
async def test_vllm_generation_reuses_one_rag_result(monkeypatch, semantic_plan, controlled_public_bundle):
    from db.schemas import MessageRequest
    calls = 0
    captured = {}

    async def retrieve(query, top_k, filters, *, database):
        nonlocal calls
        calls += 1
        return {'results': [{'id': 'controlled-source', 'title': 'source', 'content': 'evidence', 'score': 0.9}], 'citations': [{'source_id': 'controlled-source', 'source_title': 'source'}], 'confidence': 0.9, 'abstained': False, 'domains': ['tsukiyashiro_kisaki']}

    class Client:

        async def generate(self, **kwargs):
            if not captured:
                captured.update(kwargs)
                return 'answer'
            import json
            payload = json.loads(kwargs['messages'][-1]['content'])
            assert payload['answer'] == 'answer' and 'evidence' in payload['sources'][0]['body']
            return '{"citations": []}'
    monkeypatch.setattr(chat_answer, 'retrieve_chat_knowledge', retrieve)
    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: 'system')
    semantic_plan('question', {'public_knowledge': [0]})
    request = MessageRequest(message='question', sessionId='session')
    _chat_result_meta = await chat_answer.generate_chat_answer(request, None, runtime_config={'useKnowledgeBase': True}, model_generate=Client().generate)
    reply, used_rag, meta = (_chat_result_meta.reply, _chat_result_meta.used_rag, _chat_result_meta.metadata)
    assert reply == 'answer'
    assert used_rag is True
    assert calls == 1
    assert meta['citations'] == []
    assert meta['answerMode'] == 'grounded_answer'
    assert meta['domainId'] == 'tsukiyashiro_kisaki'
    assert 'evidence' in captured['messages'][-1]['content']
    assert '<retrieved_evidence trust="untrusted"' in captured['messages'][-1]['content']
    assert '<user_query>' in captured['messages'][-1]['content']
    assert '【事实与安全边界】' in captured['messages'][0]['content']
    assert '【检索证据约束】' in captured['messages'][0]['content']


@pytest.mark.asyncio
async def test_vllm_generation_omits_rag_policy_without_evidence(monkeypatch):
    from db.schemas import MessageRequest

    captured = {}

    class Client:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            return "answer"

    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: "persona")

    _chat_result_meta = await chat_answer.generate_chat_answer(
        MessageRequest(message="ordinary chat", sessionId="session"),
        None,
        runtime_config={"useKnowledgeBase": False},
        model_generate=Client().generate,
    )
    reply, used_rag, meta = _chat_result_meta.reply, _chat_result_meta.used_rag, _chat_result_meta.metadata

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
@pytest.mark.parametrize('outcome', ['character reply', '', 'error'])
async def test_vllm_rag_abstention_generates_character_uncertainty(monkeypatch, outcome, semantic_plan, controlled_public_bundle):
    from db.schemas import MessageRequest
    captured = {}

    async def retrieve(query, top_k, filters, *, database):
        return {'results': [{'content': 'UNRELIABLE_FACT'}], 'context_text': 'UNRELIABLE_FACT', 'citations': [{'id': 'unreliable-source'}], 'confidence': 0.1, 'abstained': True}

    class Client:

        async def generate(self, **kwargs):
            captured.update(kwargs)
            if outcome == 'error':
                raise RuntimeError('model unavailable')
            return outcome
    monkeypatch.setattr(chat_answer, 'retrieve_chat_knowledge', retrieve)
    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: 'system')
    semantic_plan('unknown question', {'public_knowledge': [0]})
    if outcome != 'character reply':
        expected = RuntimeError if outcome == 'error' else ValueError
        with pytest.raises(expected, match='model unavailable|non-empty text'):
            await chat_answer.generate_chat_answer(MessageRequest(message='unknown question', sessionId='session'), 'character-adapter', runtime_config={'useKnowledgeBase': True}, model_generate=Client().generate)
        assert captured
        return
    _chat_result_meta = await chat_answer.generate_chat_answer(MessageRequest(message='unknown question', sessionId='session'), 'character-adapter', runtime_config={'useKnowledgeBase': True}, model_generate=Client().generate)
    reply, used_rag, meta = (_chat_result_meta.reply, _chat_result_meta.used_rag, _chat_result_meta.metadata)
    assert reply == 'character reply'
    assert used_rag is True
    assert meta['abstained'] is True
    assert _chat_result_meta.model_invoked is True
    assert meta['answerMode'] == 'abstention'
    assert meta['citations'] == []
    assert captured['lora_name'] == 'character-adapter'
    assert 'system' in captured['messages'][0]['content']
    assert '【本轮证据不足】' in captured['messages'][0]['content']
    assert '不得猜测或补编' in captured['messages'][0]['content']
    assert 'UNRELIABLE_FACT' not in str(captured['messages'])
    assert 'unknown question' in captured['messages'][-1]['content']
    assert 'generationError' not in meta

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
            return b'{}'

    class Runtime:

        async def check_rate_limits(self, *args):
            raise AssertionError('duplicates must be resolved before chat rate limiting')
    db = SQLiteDB(tmp_path / 'integration.db')
    monkeypatch.setenv('ENVIRONMENT', 'development')
    monkeypatch.setenv('ASTRBOT_ENABLED', 'true')
    monkeypatch.setenv('ASTRBOT_INTEGRATION_TOKEN', 'test-token')
    monkeypatch.setenv('INTEGRATION_SIGNATURE_REQUIRED', 'false')
    from app.runtime import RuntimeContainer
    http_request = Request()
    http_request.app = SimpleNamespace(state=SimpleNamespace(runtime_container=RuntimeContainer(db=db, is_pg_mode=lambda: False, inference_runtime=Runtime(), http_client_pool=HttpClientPool(max_connections=1))))
    payload = integrations.AstrBotMessageRequest(platform='qq', adapter='napcat', messageId='duplicate-1', conversationId='room', conversationType='private', senderId='sender', text='hello')
    import hashlib
    import json
    key = hashlib.sha256(json.dumps(['qq', 'napcat', 'private', 'room', 'sender', 'duplicate-1'], ensure_ascii=False).encode()).hexdigest()
    db.integration_receipt('claim', key=key, owner='owner', now=0, expires_at=1)
    db.integration_receipt('finish', key=key, owner='owner', status='generated', response='{}')
    db.integration_receipt('delivery', key=key, owner='owner', status='delivered')
    response = await integrations.receive_astrbot_message(payload, http_request, x_integration_token='test-token')
    assert response.shouldReply is False
    assert response.model == 'duplicate'

@pytest.mark.asyncio
async def test_enhanced_cache_endpoints_use_async_cache_contract(monkeypatch):
    from api import enhanced

    class Cache:
        removed_pattern = None

        @property
        def stats(self):
            return {'size': 2}

        async def invalidate(self, pattern=None):
            self.removed_pattern = pattern
            return 2

    async def no_vllm_stats():
        return None
    cache = Cache()
    monkeypatch.setattr(enhanced, 'response_cache', cache)
    monkeypatch.setattr(enhanced, '_get_vllm_load_balancer_stats', no_vllm_stats)
    runtime = RuntimeContainer(db=None, is_pg_mode=lambda: False, inference_runtime=inference_runtime, http_client_pool=HttpClientPool(max_connections=1))
    runtime.http_client_pool = None
    runtime.backup_mgr = None
    runtime.access_control_mgr = None
    monkeypatch.setattr(enhanced, 'circuit_breaker_registry', None)
    stats = await enhanced.get_enhanced_stats(current_user={}, runtime=runtime)
    cleared = await enhanced.invalidate_cache('prefix', current_user={})
    assert stats['stats']['responseCache'] == {'size': 2}
    assert cleared['removed'] == 2
    assert cache.removed_pattern == 'prefix'
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
    from fastapi import HTTPException

    from api import config

    class Request:
        async def json(self):
            return {"astrbotIntegrationToken": "must-not-enter-database"}

    with pytest.raises(HTTPException) as exc_info:
        await config.update_config(Request(), current_user={"role": "admin"})

    assert exc_info.value.status_code == 422
@pytest.mark.asyncio
async def test_config_api_rejects_non_object_payload():
    from fastapi import HTTPException

    from api import config

    class Request:
        async def json(self):
            return ["not", "an", "object"]

    with pytest.raises(HTTPException) as exc_info:
        await config.update_config(Request(), current_user={"role": "admin"})

    assert exc_info.value.status_code == 422

@pytest.mark.asyncio
async def test_vllm_initialization_retries_after_transient_failure(monkeypatch):
    from fastapi import HTTPException

    from api import generate
    from app import config
    from inference import vllm_client

    attempts = 0
    expected_client = object()
    failure = ConnectionError("vLLM is starting")

    async def get_shared_client():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise failure
        return expected_client

    monkeypatch.setattr(config, "is_vllm_enabled", lambda: True)
    monkeypatch.setattr(vllm_client, "get_vllm_client", get_shared_client)

    with pytest.raises(HTTPException) as caught:
        await generate.get_vllm_client()
    assert caught.value.status_code == 503 and caught.value.__cause__ is failure
    assert await generate.get_vllm_client() is expected_client
    assert attempts == 2


def test_local_model_lock_restarts_on_a_fresh_event_loop(monkeypatch):
    from api import generate

    monkeypatch.setattr(generate, "_local_model_lock", None)
    monkeypatch.setattr(generate, "_local_model_lock_loop", None)

    async def use_lock():
        lock = generate._get_local_model_lock()
        async with lock:
            pass
        return lock

    first = asyncio.run(use_lock())
    second = asyncio.run(use_lock())
    assert second is not first


def test_shared_vllm_client_restarts_on_a_fresh_event_loop(monkeypatch):
    from app import config
    from inference import vllm_client
    from infra.circuit_breaker import CircuitBreaker

    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            self._circuit_breaker = CircuitBreaker("vllm")
            created.append(self)

        async def close(self):
            self.closed = True

    monkeypatch.setattr(config, "get_vllm_served_model_name", lambda: "test-model")
    monkeypatch.setattr(vllm_client, "VLLMClient", FakeClient)
    monkeypatch.setattr(vllm_client, "_shared_client", None)
    monkeypatch.setattr(vllm_client, "_shared_client_lock", None)
    monkeypatch.setattr(vllm_client, "_shared_client_lock_loop", None)

    async def lifecycle():
        clients = await asyncio.gather(*(vllm_client.get_vllm_client() for _ in range(5)))
        client = clients[0]
        assert all(item is client for item in clients)
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
    prompt_hash = "fixture-prompt"
    cache_key = "fixture-model"

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


@pytest.mark.asyncio
async def test_enhanced_status_does_not_advertise_retired_failover(monkeypatch):
    import httpx
    from fastapi import FastAPI

    from api import enhanced
    from app.dependencies import get_current_admin

    async def no_vllm_stats():
        return None

    monkeypatch.setattr(enhanced, '_get_vllm_load_balancer_stats', no_vllm_stats)
    from app.runtime import RuntimeContainer
    from infra.concurrency_control import inference_runtime
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(db=None, is_pg_mode=lambda: False, inference_runtime=inference_runtime)
    app.include_router(enhanced.router)
    app.dependency_overrides[get_current_admin] = lambda: {'role': 'admin'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        status = await client.get('/api/enhanced/status')
        assert status.status_code == 200
        assert 'failover' not in status.json()['enhancedFeatures']
        assert 'failover' not in status.json()['availableFeatures']
        retired = await client.get('/api/enhanced/failover/status')
        assert retired.status_code == 404
