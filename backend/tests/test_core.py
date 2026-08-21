"""
核心模块单元测试

测试不需要外部服务的核心逻辑：
- 语义缓存 L1 (进程内LRU)
- 熔断器状态机
- 消息队列数据模型
- 文本归一化
- 令牌桶限流器
"""

import asyncio
import time
import uuid
from pathlib import Path
import pytest

# ============================================
# 熔断器状态机测试
# ============================================

class TestCircuitBreaker:
    """熔断器三状态机测试"""

    def _make_cb(self, threshold=3, timeout=1.0):
        from infra.circuit_breaker import CircuitBreaker, DegradationMode
        return CircuitBreaker(
            name="test",
            failure_threshold=threshold,
            recovery_timeout=timeout,
            degradation_mode=DegradationMode.DEFAULT,
        )

    @pytest.mark.asyncio
    async def test_initial_state_closed(self):
        cb = self._make_cb()
        assert cb.state.value == "closed"

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        cb = self._make_cb(threshold=3)
        async def failing_func():
            raise RuntimeError("fail")

        for _ in range(3):
            try:
                await cb.call(failing_func)
            except RuntimeError:
                pass
        assert cb.state.value == "open"

    @pytest.mark.asyncio
    async def test_success_resets_failures(self):
        cb = self._make_cb(threshold=3)
        async def failing_func():
            raise RuntimeError("fail")
        async def success_func():
            return "ok"

        # 2 failures
        for _ in range(2):
            try:
                await cb.call(failing_func)
            except RuntimeError:
                pass
        # 1 success resets
        result = await cb.call(success_func)
        assert result == "ok"
        # Need 3 more failures to open
        for _ in range(3):
            try:
                await cb.call(failing_func)
            except RuntimeError:
                pass
        assert cb.state.value == "open"

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self):
        cb = self._make_cb(threshold=2, timeout=0.1)
        async def failing_func():
            raise RuntimeError("fail")

        for _ in range(2):
            try:
                await cb.call(failing_func)
            except RuntimeError:
                pass
        assert cb.state.value == "open"

        # Wait for recovery timeout
        await asyncio.sleep(0.15)
        assert cb.state.value == "half_open"

    @pytest.mark.asyncio
    async def test_degradation_returns_default(self):
        from infra.circuit_breaker import CircuitBreaker, DegradationMode
        cb = CircuitBreaker(
            name="test",
            failure_threshold=1,
            recovery_timeout=60,
            degradation_mode=DegradationMode.DEFAULT,
        )
        cb.set_default("fallback")
        async def failing_func():
            raise RuntimeError("fail")

        # Trigger open
        try:
            await cb.call(failing_func)
        except RuntimeError:
            pass
        # Should return default value instead of raising
        result = await cb.call(failing_func)
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_reset(self):
        cb = self._make_cb(threshold=1)
        async def failing_func():
            raise RuntimeError("fail")
        try:
            await cb.call(failing_func)
        except RuntimeError:
            pass
        assert cb.state.value == "open"
        await cb.reset()
        assert cb.state.value == "closed"

    @pytest.mark.asyncio
    async def test_stats(self):
        cb = self._make_cb()
        async def success_func():
            return "ok"
        await cb.call(success_func)
        stats = cb.get_stats()
        assert stats["success_calls"] == 1
        assert stats["state"] == "closed"


# ============================================
# Multi-platform integration tests
# ============================================

class TestDatabaseConfiguration:
    def test_sqlite_path_uses_database_path_env(self, monkeypatch):
        from db.database import BACKEND_DIR, _database_path_from_env

        custom_path = Path(".test_tmp") / "config" / f"custom_{uuid.uuid4().hex}.db"
        monkeypatch.setenv("DATABASE_PATH", str(custom_path))

        assert _database_path_from_env() == BACKEND_DIR / custom_path

class TestMultiPlatformStorage:
    def _make_db(self):
        from db.database import SQLiteDB
        db_path = Path(".test_tmp") / f"qqchat_test_{uuid.uuid4().hex}.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return SQLiteDB(db_path)

    def test_platform_fields_and_session_isolation(self):
        db = self._make_db()
        db.add_message({
            "sessionType": "group",
            "sessionId": "qq:group:10001",
            "sessionName": "QQ group",
            "platform": "qq",
            "adapter": "napcat",
            "conversationId": "10001",
            "senderId": "u1",
            "sourceMessageId": "m1",
            "userId": "u1",
            "userName": "Alice",
            "message": "hello",
            "reply": "hi",
        })
        db.add_message({
            "sessionType": "group",
            "sessionId": "telegram:group:10001",
            "sessionName": "TG group",
            "platform": "telegram",
            "adapter": "telegram",
            "conversationId": "10001",
            "senderId": "u2",
            "sourceMessageId": "m2",
            "userId": "u2",
            "userName": "Bob",
            "message": "hello",
            "reply": "hi",
        })

        qq_rows = db.get_messages_filtered(platform="qq")
        tg_rows = db.get_messages_filtered(platform="telegram")
        assert len(qq_rows) == 1
        assert len(tg_rows) == 1
        assert qq_rows[0]["sessionId"] != tg_rows[0]["sessionId"]

    def test_message_count_filtered_matches_platform_filters(self):
        db = self._make_db()
        for platform, session_id in (("qq", "qq:group:10001"), ("telegram", "telegram:group:10001")):
            db.add_message({
                "sessionType": "group",
                "sessionId": session_id,
                "sessionName": platform,
                "platform": platform,
                "adapter": platform,
                "conversationId": "10001",
                "senderId": "u1",
                "sourceMessageId": f"{platform}-m1",
                "userId": "u1",
                "userName": "Alice",
                "message": "same text",
                "reply": "reply",
            })

        assert db.get_message_count_filtered(platform="qq") == 1
        assert db.get_message_count_filtered(platform="telegram") == 1
        assert db.get_message_count_filtered(search="same text") == 2
        assert db.get_message_count_filtered(session_name="telegram") == 1
        rows = db.get_messages_filtered(session_name="telegram")
        assert len(rows) == 1
        assert rows[0]["platform"] == "telegram"

    def test_integration_dedup_uses_platform_adapter_message(self):
        db = self._make_db()
        assert db.mark_integration_message_processed("qq", "napcat", "same-id") is True
        assert db.mark_integration_message_processed("qq", "napcat", "same-id") is False
        assert db.mark_integration_message_processed("telegram", "telegram", "same-id") is True

    def test_session_toggle_accepts_platform_context(self):
        db = self._make_db()
        sid = "telegram:private:42"
        assert db.is_session_bot_enabled(sid, "telegram", "42") is True
        db.set_session_bot_enabled(sid, False, "telegram", "42")
        assert db.is_session_bot_enabled(sid, "telegram", "42") is False

    def test_message_request_preserves_platform_metadata(self):
        from db.schemas import MessageRequest

        req = MessageRequest(
            message="hello",
            sessionType="group",
            sessionId="telegram:group:10001",
            sessionName="TG group",
            platform="telegram",
            adapter="telegram",
            conversationId="10001",
            senderId="u2",
            sourceMessageId="m2",
            traceId="trace-1",
        )

        assert req.platform == "telegram"
        assert req.adapter == "telegram"
        assert req.conversationId == "10001"
        assert req.sourceMessageId == "m2"
        assert req.traceId == "trace-1"

    def test_message_write_upserts_conversation_fields(self):
        db = self._make_db()
        db.add_message({
            "sessionType": "group",
            "conversationType": "group",
            "sessionId": "telegram:group:room-1",
            "sessionName": "Room One",
            "platform": "telegram",
            "adapter": "telegram",
            "conversationId": "room-1",
            "senderId": "u1",
            "senderName": "Alice",
            "userId": "u1",
            "userName": "Alice",
            "message": "hello",
            "reply": "hi",
        })

        row = db.get_messages_filtered(platform="telegram")[0]
        conversation = db.get_conversation("telegram", "room-1", "group")

        assert row["conversationType"] == "group"
        assert row["senderName"] == "Alice"
        assert conversation["displayName"] == "Room One"
        assert conversation["botEnabled"] == 1

    def test_conversation_bot_enabled_controls_are_mirrored(self):
        db = self._make_db()
        session_id = "qq:private:42"
        db.set_session_bot_enabled(session_id, False, "qq", "42")

        conversation = db.get_conversation("qq", "42")

        assert conversation["botEnabled"] == 0
        assert db.is_session_bot_enabled(session_id, "qq", "42") is False

    def test_integration_events_and_model_invocations_are_recorded(self):
        db = self._make_db()
        db.add_integration_event({
            "platform": "qq",
            "adapter": "napcat",
            "sourceMessageId": "m1",
            "conversationId": "g1",
            "conversationType": "group",
            "senderId": "u1",
            "eventHash": "hash-1",
            "rawSummary": {"message_type": "group"},
            "traceId": "trace-1",
        })
        db.add_model_invocation({
            "traceId": "trace-1",
            "platform": "qq",
            "conversationId": "g1",
            "sessionId": "qq:group:g1",
            "modelName": "mock",
            "loraName": "default",
            "costTime": 0.2,
            "promptTokens": 3,
            "completionTokens": 4,
            "usedRag": True,
        })

        event_rows = db.execute_sql("SELECT * FROM integration_events WHERE traceId = ?", ("trace-1",))
        invocation_rows = db.execute_sql("SELECT * FROM model_invocations WHERE traceId = ?", ("trace-1",))

        assert len(event_rows) == 1
        assert event_rows[0]["conversationType"] == "group"
        assert len(invocation_rows) == 1
        assert invocation_rows[0]["totalTokens"] == 7
        assert invocation_rows[0]["usedRag"] == 1

    def test_trace_id_is_saved_with_message(self):
        db = self._make_db()
        db.add_message({
            "sessionType": "private",
            "sessionId": "qq:private:42",
            "sessionName": "QQ private",
            "platform": "qq",
            "adapter": "napcat",
            "conversationId": "42",
            "senderId": "42",
            "sourceMessageId": "m-trace",
            "traceId": "trace-xyz",
            "userId": "42",
            "userName": "Alice",
            "message": "hello",
            "reply": "hi",
        })

        rows = db.get_messages_filtered(platform="qq")
        assert rows[0]["traceId"] == "trace-xyz"

    def test_fallback_dedup_message_id_is_stable_for_same_event(self):
        from api.integrations import AstrBotMessageRequest, _dedup_message_id

        req = AstrBotMessageRequest(
            platform="telegram",
            adapter="telegram",
            conversationId="10001",
            conversationType="group",
            senderId="u2",
            text="hello",
            raw={"timestamp": 123456},
        )

        first = _dedup_message_id(req, "hello")
        second = _dedup_message_id(req, "hello")
        changed = _dedup_message_id(req, "hello again")
        assert first.startswith("fallback:")
        assert first == second
        assert first != changed

    def test_degraded_response_private_vs_group(self):
        from api.integrations import AstrBotMessageRequest, _degraded_response

        private_req = AstrBotMessageRequest(
            platform="qq",
            adapter="napcat",
            conversationId="42",
            conversationType="private",
            senderId="u1",
            text="hello",
        )
        group_req = AstrBotMessageRequest(
            platform="qq",
            adapter="napcat",
            conversationId="100",
            conversationType="group",
            senderId="u1",
            text="hello",
        )

        private_resp = _degraded_response(private_req, "trace-private", model="db-unavailable")
        group_resp = _degraded_response(group_req, "trace-group", model="db-unavailable")

        assert private_resp.shouldReply is True
        assert private_resp.replyText
        assert private_resp.traceId == "trace-private"
        assert group_resp.shouldReply is False
        assert group_resp.replyText == ""
        assert group_resp.traceId == "trace-group"

    @pytest.mark.asyncio
    async def test_astrbot_global_switch_can_disable_platforms(self, monkeypatch):
        from api.integrations import _is_platform_enabled

        monkeypatch.setenv("ASTRBOT_ENABLED", "false")
        assert await _is_platform_enabled("qq") is False

    @pytest.mark.asyncio
    async def test_astrbot_personal_wechat_can_be_disabled_by_env(self, monkeypatch):
        from api.integrations import _is_platform_enabled

        monkeypatch.setenv("ASTRBOT_ENABLED", "true")
        monkeypatch.setenv("ASTRBOT_WECHAT_PERSONAL_ENABLED", "false")
        assert await _is_platform_enabled("wechat_personal") is False

class TestIntegrationSecurity:
    def test_integration_signature_accepts_once_and_rejects_replay(self):
        from fastapi import HTTPException
        from infra import security_utils

        security_utils._seen_nonces.clear()
        body = b'{"text":"hello"}'
        signature = security_utils.integration_signature("secret-token", "1000", "nonce-1", body)

        security_utils.verify_integration_signature(
            token="secret-token",
            timestamp="1000",
            nonce="nonce-1",
            signature=signature,
            body=body,
            now=1000,
        )
        with pytest.raises(HTTPException) as exc:
            security_utils.verify_integration_signature(
                token="secret-token",
                timestamp="1000",
                nonce="nonce-1",
                signature=signature,
                body=body,
                now=1000,
            )
        assert exc.value.status_code == 409

    def test_security_helpers_sanitize_and_redact(self):
        from infra.security_utils import redact_sensitive, strip_control_chars

        assert strip_control_chars("hi\x00 there\x1f") == "hi there"
        redacted = redact_sensitive({
            "token": "abcdef123456",
            "phone": "13800138000",
            "nested": {"openid": "openid-abcdefghijklmn"},
        })
        assert redacted["token"] != "abcdef123456"
        assert redacted["phone"] != "13800138000"
        assert redacted["nested"]["openid"] != "openid-abcdefghijklmn"

    @pytest.mark.asyncio
    async def test_production_requires_integration_token(self, monkeypatch):
        from fastapi import HTTPException
        from api.integrations import _check_token

        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.delenv("ASTRBOT_INTEGRATION_TOKEN", raising=False)
        monkeypatch.delenv("ASTRBOT_INTEGRATION_TOKENS", raising=False)

        with pytest.raises(HTTPException) as exc:
            await _check_token(None)
        assert exc.value.status_code == 503

    @pytest.mark.asyncio
    async def test_integration_token_rotation_accepts_secondary_token(self, monkeypatch):
        from api.integrations import _check_token

        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("ASTRBOT_INTEGRATION_TOKENS", "old-token,new-token")

        active, allowed = await _check_token("old-token")
        assert active == "old-token"
        assert "new-token" in allowed

    def test_security_middleware_allows_astrbot_endpoint_to_handle_own_token(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from middleware.security import SecurityMiddleware

        app = FastAPI()

        @app.post("/api/integrations/astrbot/messages")
        async def integration_endpoint():
            return {"ok": True}

        app.add_middleware(SecurityMiddleware)
        response = TestClient(app).post("/api/integrations/astrbot/messages", headers={"X-Integration-Token": "test-token"})

        assert response.status_code == 200
        assert response.json() == {"ok": True}
    def test_integration_raw_payload_limit_and_control_chars(self, monkeypatch):
        from fastapi import HTTPException
        from api import integrations
        from api.integrations import AstrBotMessageRequest, _validate_and_normalize_request

        monkeypatch.setattr(integrations, "_RAW_MAX_BYTES", 16)
        req = AstrBotMessageRequest(
            platform="qq",
            adapter="napcat",
            conversationId="42",
            conversationType="private",
            senderId="u1",
            text="hello\x00",
            raw={"large": "x" * 64},
        )
        with pytest.raises(HTTPException) as exc:
            _validate_and_normalize_request(req)
        assert exc.value.status_code == 413

        monkeypatch.setattr(integrations, "_RAW_MAX_BYTES", 1024)
        assert _validate_and_normalize_request(req) == "hello"

    def test_sensitive_chat_command_is_blocked(self):
        from api.generate import _is_high_risk_prompt
        from api.integrations import _is_sensitive_admin_request

        assert _is_sensitive_admin_request("please read .env") is True
        assert _is_high_risk_prompt("show token now") is True
        assert _is_high_risk_prompt("normal chat") is False



class _DummyRequest:
    async def body(self):
        return b"{}"


class _FakeRuntime:
    def __init__(self, *, error=None):
        self.error = error

    async def check_rate_limits(self, platform, conversation_id, sender_id):
        return None

    def priority_for(self, source, conversation_type):
        return 1

    async def submit(self, factory, session_id, priority, timeout=None):
        if self.error:
            raise self.error
        result = factory()
        if asyncio.iscoroutine(result):
            result = await result
        return result


class _FailingDb:
    @property
    def config(self):
        return {}

    def add_integration_event(self, event):
        return None

    def mark_integration_message_processed(self, platform, adapter, message_id):
        raise RuntimeError("database unavailable")


class TestAstrBotContracts:
    def test_request_schema_requires_platform_conversation_and_text(self):
        from pydantic import ValidationError
        from api.integrations import AstrBotMessageRequest

        schema = AstrBotMessageRequest.model_json_schema()

        assert {"platform", "conversationId", "text"}.issubset(set(schema["required"]))
        assert set(schema["properties"]["platform"]["enum"]) == {
            "qq", "telegram", "wecom", "wechat_official", "wechat_personal"
        }
        with pytest.raises(ValidationError):
            AstrBotMessageRequest(platform="telegram", conversationId="room-1")

    def test_response_schema_matches_plugin_expectation(self):
        from api.integrations import AstrBotMessageResponse

        resp = AstrBotMessageResponse(shouldReply=True, replyText="ok", model="mock", costTime=0.1, traceId="trace")
        payload = resp.model_dump()

        assert set(payload) == {"shouldReply", "replyText", "model", "costTime", "traceId"}
        assert payload["shouldReply"] is True

    def test_main_app_exposes_astrbot_endpoint(self, monkeypatch):
        monkeypatch.setenv("SECURITY_MIDDLEWARE_ENABLED", "false")
        from fastapi.testclient import TestClient

        from app.main import app

        openapi = TestClient(app).get("/openapi.json").json()
        assert "/api/integrations/astrbot/messages" in openapi["paths"]


class TestAstrBotIntegrationFlow:
    async def _call_success(self, monkeypatch, platform, conversation_type, conversation_id):
        from api import integrations
        from db.schemas import GenerateResponse

        temp_db = TestMultiPlatformStorage()._make_db()
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("ASTRBOT_ENABLED", "true")
        monkeypatch.setenv("ASTRBOT_INTEGRATION_TOKEN", "test-token")
        monkeypatch.setenv("INTEGRATION_SIGNATURE_REQUIRED", "false")
        monkeypatch.setattr(integrations, "db", temp_db)
        monkeypatch.setattr(integrations, "inference_runtime", _FakeRuntime())

        async def fake_generate(message_request, current_user=None, **kwargs):
            assert message_request.platform == platform
            assert message_request.conversationId == conversation_id
            assert message_request.sessionId == f"{platform}:{conversation_type}:{conversation_id}"
            return GenerateResponse(reply=f"reply:{platform}", model="mock", costTime=0.01)

        monkeypatch.setattr(integrations, "generate_reply_core", fake_generate)
        req = integrations.AstrBotMessageRequest(
            platform=platform,
            adapter="telegram" if platform == "telegram" else "napcat" if platform == "qq" else "official_account",
            messageId=f"msg-{platform}-{conversation_id}",
            conversationId=conversation_id,
            conversationType=conversation_type,
            senderId="sender-1",
            senderName="Alice",
            text="hello",
            raw={"conversationName": "Room", "timestamp": 123456},
        )

        return await integrations.receive_astrbot_message(req, _DummyRequest(), x_integration_token="test-token")

    @pytest.mark.asyncio
    async def test_simulated_astrbot_qq_group_message(self, monkeypatch):
        resp = await self._call_success(monkeypatch, "qq", "group", "10001")
        assert resp.shouldReply is True
        assert resp.replyText == "reply:qq"

    @pytest.mark.asyncio
    async def test_simulated_telegram_private_message(self, monkeypatch):
        resp = await self._call_success(monkeypatch, "telegram", "private", "42")
        assert resp.shouldReply is True
        assert resp.replyText == "reply:telegram"

    @pytest.mark.asyncio
    async def test_simulated_wechat_message(self, monkeypatch):
        resp = await self._call_success(monkeypatch, "wechat_official", "private", "openid-1")
        assert resp.shouldReply is True
        assert resp.replyText == "reply:wechat_official"

    @pytest.mark.asyncio
    async def test_backend_timeout_degrades_private_message(self, monkeypatch):
        from api import integrations

        temp_db = TestMultiPlatformStorage()._make_db()
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("ASTRBOT_INTEGRATION_TOKEN", "test-token")
        monkeypatch.setenv("INTEGRATION_SIGNATURE_REQUIRED", "false")
        monkeypatch.setattr(integrations, "db", temp_db)
        monkeypatch.setattr(integrations, "inference_runtime", _FakeRuntime(error=asyncio.TimeoutError()))
        req = integrations.AstrBotMessageRequest(
            platform="telegram",
            adapter="telegram",
            messageId="timeout-1",
            conversationId="42",
            conversationType="private",
            senderId="sender-1",
            text="hello",
        )

        resp = await integrations.receive_astrbot_message(req, _DummyRequest(), x_integration_token="test-token")

        assert resp.shouldReply is True
        assert resp.model == "queue-timeout"
        assert resp.replyText

    @pytest.mark.asyncio
    async def test_model_failure_degrades_group_silently(self, monkeypatch):
        from api import integrations

        temp_db = TestMultiPlatformStorage()._make_db()
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("ASTRBOT_INTEGRATION_TOKEN", "test-token")
        monkeypatch.setenv("INTEGRATION_SIGNATURE_REQUIRED", "false")
        monkeypatch.setattr(integrations, "db", temp_db)
        monkeypatch.setattr(integrations, "inference_runtime", _FakeRuntime(error=RuntimeError("model failed")))
        req = integrations.AstrBotMessageRequest(
            platform="qq",
            adapter="napcat",
            messageId="model-fail-1",
            conversationId="10001",
            conversationType="group",
            senderId="sender-1",
            text="hello",
        )

        resp = await integrations.receive_astrbot_message(req, _DummyRequest(), x_integration_token="test-token")

        assert resp.shouldReply is False
        assert resp.model == "generation-failed"

    @pytest.mark.asyncio
    async def test_database_exception_degrades_private_message(self, monkeypatch):
        from api import integrations

        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("ASTRBOT_INTEGRATION_TOKEN", "test-token")
        monkeypatch.setenv("INTEGRATION_SIGNATURE_REQUIRED", "false")
        monkeypatch.setattr(integrations, "db", _FailingDb())
        monkeypatch.setattr(integrations, "inference_runtime", _FakeRuntime())
        req = integrations.AstrBotMessageRequest(
            platform="telegram",
            adapter="telegram",
            messageId="db-fail-1",
            conversationId="42",
            conversationType="private",
            senderId="sender-1",
            text="hello",
        )

        resp = await integrations.receive_astrbot_message(req, _DummyRequest(), x_integration_token="test-token")

        assert resp.shouldReply is True
        assert resp.model == "db-unavailable"


class TestDeploymentValidation:
    def test_production_requires_explicit_critical_environment(self):
        from infra.deployment import validate_deployment_environment

        result = validate_deployment_environment({"ENVIRONMENT": "production", "LOG_LEVEL": "INFO"})

        assert result.ok is False
        assert any("ASTRBOT_INTEGRATION_TOKEN" in item for item in result.errors)
        assert any("DATABASE_URL" in item for item in result.errors)
        assert any("JWT_SECRET" in item for item in result.errors)
        assert any("ENCRYPTION_KEY" in item for item in result.errors)

    def test_production_accepts_complete_environment(self):
        from infra.deployment import validate_deployment_environment

        result = validate_deployment_environment({
            "ENVIRONMENT": "production",
            "ASTRBOT_INTEGRATION_TOKEN": "astrbot-token-value-that-is-long-enough",
            "QQCHAT_BACKEND_URL": "http://backend:8000",
            "DATABASE_URL": "postgresql+asyncpg://user:strongpass@postgres:5432/qqassistant",
            "VLLM_BASE_URL": "http://vllm:8001",
            "JWT_SECRET": "jwt-secret-value-that-is-at-least-32-chars",
            "ENCRYPTION_KEY": "ZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWU=",
            "ALLOWED_ORIGINS": "https://admin.example.com",
            "LOG_LEVEL": "INFO",
        })

        assert result.ok is True
        assert result.errors == []


class TestServiceStatus:
    @pytest.mark.asyncio
    async def test_services_includes_astrbot_gateway_degraded(self, monkeypatch):
        from api import stats

        monkeypatch.setenv("ASTRBOT_ENABLED", "true")
        monkeypatch.setattr(stats, "_check_service", lambda port: False)

        payload = await stats.get_services()
        astrbot = next(item for item in payload["services"] if item["name"] == "AstrBot Gateway")

        assert astrbot["status"] == "degraded"
        assert astrbot["port"] == 6185

class TestObservability:
    def test_metrics_payload_uses_model_invocations_for_percentiles_and_failures(self, monkeypatch):
        from datetime import datetime
        from api import stats

        db = TestMultiPlatformStorage()._make_db()
        now = datetime.now().isoformat()
        for idx, cost in enumerate([1.0, 2.0, 10.0], start=1):
            db.add_message({
                "sessionType": "private",
                "sessionId": f"qq:private:{idx}",
                "platform": "qq",
                "conversationId": str(idx),
                "message": "hello",
                "reply": "hi",
                "costTime": cost,
                "createdAt": now,
            })
        db.add_model_invocation({"traceId": "ok", "platform": "qq", "costTime": 2.0, "createdAt": now})
        db.add_model_invocation({"traceId": "bad", "platform": "qq", "costTime": 10.0, "errorType": "TimeoutError", "createdAt": now})

        monkeypatch.setattr(stats, "db", db)
        monkeypatch.setattr(stats, "_check_service", lambda port: True)

        metrics = stats._metrics_payload()

        assert metrics["todayMessages"] == 3
        assert metrics["todayReplies"] == 3
        assert metrics["p95ResponseTime"] == 10.0
        assert metrics["p99ResponseTime"] == 10.0
        assert metrics["modelFailureRate"] == 0.5
        assert metrics["queueLength"] >= 0

    def test_alerts_include_queue_slow_auth_and_model_failure(self):
        from api import stats
        from infra import observability

        observability._COUNTERS.clear()
        observability._RECENT.clear()
        observability._CONSECUTIVE.clear()
        for _ in range(3):
            observability.set_consecutive("model_failure", False)
        for _ in range(10):
            observability.increment("integration_auth_failures")

        metrics = {
            "queueLength": 90,
            "queueMaxSize": 100,
            "p95ResponseTime": 99,
            "astrBotGateway": {"status": "running"},
        }
        alerts = stats._alerts_from_metrics(metrics)
        alert_types = {item["type"] for item in alerts}

        assert "model_consecutive_failures" in alert_types
        assert "message_backlog" in alert_types
        assert "slow_response" in alert_types
        assert "frequent_auth_failures" in alert_types

    @pytest.mark.asyncio
    async def test_inference_runtime_reports_active_workers(self):
        from infra.concurrency_control import InferenceRuntime

        runtime = InferenceRuntime()
        runtime.worker_count = 1
        observed_active = 0

        async def task():
            nonlocal observed_active
            observed_active = runtime.stats()["active"]
            await asyncio.sleep(0.01)
            return "ok"

        result = await runtime.submit(task, session_id="obs", priority=1, timeout=1)

        assert result == "ok"
        assert observed_active == 1

# ============================================
# Concurrency control tests
# ============================================

class TestConcurrencyControl:
    @pytest.mark.asyncio
    async def test_token_bucket_rejects_over_capacity(self):
        from infra.concurrency_control import TokenBucketLimiter

        limiter = TokenBucketLimiter(rate=0.1, capacity=1)
        allowed, _ = await limiter.acquire("sender:u1")
        rejected, retry_after = await limiter.acquire("sender:u1")
        assert allowed is True
        assert rejected is False
        assert retry_after > 0

    @pytest.mark.asyncio
    async def test_same_session_is_serialized_across_workers(self):
        from infra.concurrency_control import InferenceRuntime

        runtime = InferenceRuntime()
        runtime.worker_count = 2
        active = 0
        max_active = 0

        async def task():
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return "ok"

        results = await asyncio.gather(
            runtime.submit(task, session_id="same-session", priority=10, timeout=1),
            runtime.submit(task, session_id="same-session", priority=10, timeout=1),
        )
        assert results == ["ok", "ok"]
        assert max_active == 1
