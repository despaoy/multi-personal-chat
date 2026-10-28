"""Gateway retries must reuse generation and expose only delivered history."""

from infra.resource_pool import HttpClientPool

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import FastAPI
from app.runtime import RuntimeContainer

from api import integrations
from db.database import SQLiteDB
from db.schemas import GenerateResponse


class Request:
    headers = {"X-Integration-Token": "test-token"}


class Runtime:
    async def check_rate_limits(self, *args):
        pass

    def priority_for(self, *args):
        return 0

    async def submit(self, factory, **kwargs):
        return await factory()


@pytest.fixture
def gateway(monkeypatch, tmp_path):
    database = SQLiteDB(tmp_path / 'delivery.db')
    monkeypatch.setenv('ENVIRONMENT', 'development')
    monkeypatch.setenv('ASTRBOT_ENABLED', 'true')
    monkeypatch.setenv('INTEGRATION_SIGNATURE_REQUIRED', 'false')
    monkeypatch.setenv('ASTRBOT_INTEGRATION_TOKEN', 'test-token')
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(db=database, is_pg_mode=lambda: False, inference_runtime=Runtime(), startup_env={}, http_client_pool=HttpClientPool(max_connections=1))
    monkeypatch.setattr(Request, 'app', app, raising=False)
    calls = []

    async def generate(msg, **kwargs):
        from api.generate import _save_message
        calls.append(msg)
        assert await _save_message(msg, 'reply', 'model', 'default', 0.1, database=database)
        kwargs['delivery_context'].update(message_saved=True, character_id=None)
        return GenerateResponse(reply='reply', model='model', costTime=0.1)
    monkeypatch.setattr(integrations, 'generate_reply_core', generate)
    payload = integrations.AstrBotMessageRequest(platform='qq', adapter='napcat', messageId='m1', conversationId='room', senderId='sender', text='hello')
    return (database, calls, payload)


async def receive(payload):
    return await integrations.receive_astrbot_message(payload, Request(), x_integration_token="test-token")


async def ack(response, status="delivered"):
    return await integrations.acknowledge_delivery(
        integrations.DeliveryAcknowledgement(
            receiptId=response.receiptId,
            deliveryToken=response.deliveryToken,
            status=status,
        ),
        Request(),
    )


@pytest.mark.asyncio
async def test_failed_send_reuses_reply_and_ack_exposes_history(gateway):
    database, calls, payload = gateway
    first = await receive(payload)
    assert first.receiptId and first.shouldReply
    history = lambda: database.list_conversation_history("qq", "napcat", "sender", "private", "room")
    assert history() == []
    await ack(first, "delivery_failed")
    retry = await receive(payload)
    assert retry.model_dump() == first.model_dump()
    assert len(calls) == 1
    assert (await ack(first))["changed"]
    assert [m["content"] for m in history()] == ["hello", "reply"]
    assert not (await ack(first))["changed"]
    assert not (await receive(payload)).shouldReply
    # Late failure reports cannot reverse successful delivery.
    assert not (await ack(first, "delivery_failed"))["changed"]


@pytest.mark.asyncio
async def test_generation_failure_and_timeout_are_retryable(gateway, monkeypatch):
    database, calls, payload = gateway
    original = integrations.generate_reply_core

    async def fail(*args, **kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(integrations, "generate_reply_core", fail)
    with pytest.raises(integrations.HTTPException) as error:
        await receive(payload)
    assert error.value.status_code == 503 and isinstance(error.value.__cause__, RuntimeError)
    assert database.execute_sql("SELECT status FROM integration_receipts") == [{"status": "failed"}]
    cancelled = asyncio.Event()

    async def slow(*args, **kwargs):
        try:
            await asyncio.sleep(1)
        finally:
            cancelled.set()

    monkeypatch.setattr(integrations, "generate_reply_core", slow)
    monkeypatch.setattr(integrations, "_MODEL_TIMEOUT", 0.05)
    with pytest.raises(integrations.HTTPException) as error:
        await receive(payload)
    assert error.value.status_code == 504 and isinstance(error.value.__cause__, TimeoutError)
    assert database.execute_sql("SELECT status FROM integration_receipts WHERE receipt_key NOT LIKE 'attempt:%'") == [{"status": "failed"}]
    assert cancelled.is_set()
    monkeypatch.setattr(integrations, "_MODEL_TIMEOUT", 180)
    monkeypatch.setattr(integrations, "generate_reply_core", original)
    assert (await receive(payload)).receiptId
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_concurrent_duplicate_does_not_generate_twice(gateway, monkeypatch):
    _, calls, payload = gateway
    original = integrations.generate_reply_core
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(integrations, "generate_reply_core", slow)
    task = asyncio.create_task(receive(payload))
    await entered.wait()
    duplicate = await receive(payload)
    assert duplicate.model == "processing" and duplicate.retryable
    release.set()
    await task
    assert len(calls) == 1


def test_receipt_lease_and_owner_fencing(gateway):
    database, _, _ = gateway
    assert database.integration_receipt("claim", key="key", owner="old", now=0, expires_at=5)
    assert not database.integration_receipt("claim", key="key", owner="new", now=1, expires_at=10)
    assert database.integration_receipt("claim", key="key", owner="new", now=6, expires_at=10)
    assert not database.integration_receipt("finish", key="key", owner="old", status="generated", response="old")
    assert database.integration_receipt("finish", key="key", owner="new", status="generated", response="new")
    archived = database.integration_receipt("get", key="attempt:old")
    assert archived["owner"] == "old" and archived["status"] == "superseded"


@pytest.mark.asyncio
async def test_ack_requires_matching_delivery_token(gateway):
    _, _, payload = gateway
    result = await receive(payload)
    result.deliveryToken = "0" * 32
    with pytest.raises(integrations.HTTPException) as error:
        await ack(result)
    assert error.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"X-Integration-Token": "wrong-token"}])
async def test_ack_rejects_missing_or_invalid_integration_token(gateway, headers):
    payload = integrations.DeliveryAcknowledgement(
        receiptId="a" * 64, deliveryToken="b" * 32, status="delivered"
    )
    with pytest.raises(integrations.HTTPException) as error:
        await integrations.acknowledge_delivery(payload, SimpleNamespace(headers=headers))
    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_ack_rejects_missing_required_signature(gateway, monkeypatch):
    monkeypatch.setenv("INTEGRATION_SIGNATURE_REQUIRED", "true")

    async def body():
        return b"{}"

    request = SimpleNamespace(headers={"X-Integration-Token": "test-token"}, body=body)
    payload = integrations.DeliveryAcknowledgement(
        receiptId="a" * 64, deliveryToken="b" * 32, status="delivered"
    )
    with pytest.raises(integrations.HTTPException) as error:
        await integrations.acknowledge_delivery(payload, request)
    assert error.value.status_code == 401


def test_receipt_migration_preserves_existing_data(tmp_path, monkeypatch):
    import sqlalchemy as sa

    # Execute the migration's actual SQL on SQLite without requiring the
    # deployment-only Alembic CLI in this lightweight test environment.
    alembic = ModuleType("alembic")
    alembic.op = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "alembic", alembic)
    path = Path(__file__).resolve().parents[1] / "alembic/versions/008_integration_receipts.py"
    spec = importlib.util.spec_from_file_location("receipt_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = sa.create_engine("sqlite:///" + (tmp_path / "migration.db").as_posix())
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE existing_data (value TEXT)")
        connection.exec_driver_sql("INSERT INTO existing_data VALUES ('keep')")
        migration.op = SimpleNamespace(execute=connection.exec_driver_sql)
        migration.upgrade()
        migration.upgrade()  # SQLite startup may have created the table already.
        assert connection.exec_driver_sql("SELECT value FROM existing_data").scalar() == "keep"
        assert "integration_receipts" in sa.inspect(connection).get_table_names()
    engine.dispose()


@pytest.fixture
def plugin(monkeypatch):
    # The gateway can be tested without an AstrBot installation or real sends.
    for name in ("astrbot", "astrbot.api", "astrbot.api.event", "astrbot.api.star"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["astrbot.api"].logger = logging.getLogger("gateway-test")
    sys.modules["astrbot.api.event"].AstrMessageEvent = object
    sys.modules["astrbot.api.event"].filter = SimpleNamespace(
        EventMessageType=SimpleNamespace(ALL="all"),
        event_message_type=lambda _: lambda fn: fn,
    )
    star = sys.modules["astrbot.api.star"]
    star.Context, star.Star = object, object
    star.register = lambda *args: lambda cls: cls
    path = Path(__file__).resolve().parents[2] / "astrbot_plugins/multipersonal_gateway/main.py"
    spec = importlib.util.spec_from_file_location("gateway_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = object.__new__(module.MultiPersonalGatewayPlugin)
    instance._seen_events, instance._sent_receipts, instance._inflight = {}, {}, set()
    instance.dedup_ttl = 300
    instance._should_forward = lambda *args: True
    payload = {
        "platform": "qq",
        "adapter": "napcat",
        "messageId": "m1",
        "conversationId": "room",
        "conversationType": "private",
        "senderId": "sender",
        "text": "hello",
    }
    instance._build_payload = lambda *args: payload
    return instance


@pytest.mark.asyncio
async def test_plugin_marks_seen_only_after_successful_send(plugin):
    order = []
    response = {"shouldReply": True, "replyText": "reply", "receiptId": "r", "deliveryToken": "t"}
    plugin._post_message = lambda _: response

    async def acknowledge(response, status):
        order.append(status)
        return True

    plugin._acknowledge = acknowledge

    class Event:
        message_str = "hello"
        fail = True

        def stop_event(self):
            pass

        def plain_result(self, text):
            return text

        async def send(self, text):
            order.append("send")
            if self.fail:
                raise RuntimeError("send failed")

    event = Event()
    assert [item async for item in plugin.on_message(event)] == []
    assert plugin._seen_events == {}
    event.fail = False
    assert [item async for item in plugin.on_message(event)] == []
    assert order == ["send", "delivery_failed", "send", "delivered"]
    assert [item async for item in plugin.on_message(event)] == []
    assert len(order) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation_type", ["private", "group"])
@pytest.mark.parametrize("failure,status", [
    (integrations.InferenceQueueFull("full"), 503),
    (TimeoutError("slow"), 504),
    (RuntimeError("model unavailable"), 503),
    (integrations.HTTPException(422, "invalid generation", headers={"X-Test": "retained"}), 422),
])
async def test_gateway_failure_is_an_error_and_releases_receipt(gateway, monkeypatch, conversation_type, failure, status):
    database, calls, payload = gateway
    payload.conversationType = conversation_type

    async def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(Request.app.state.runtime_container.inference_runtime, "submit", fail)
    with pytest.raises(integrations.HTTPException) as error:
        await receive(payload)
    assert error.value.status_code == status and calls == []
    if isinstance(failure, integrations.HTTPException):
        assert error.value is failure and error.value.headers == {"X-Test": "retained"}
    else:
        assert type(error.value.__cause__) is type(failure)
    rows = database.execute_sql("SELECT status,response FROM integration_receipts")
    assert rows == [{"status": "failed", "response": ""}]
    assert database.list_conversation_history("qq", "napcat", "sender", conversation_type, "room") == []


@pytest.mark.asyncio
async def test_gateway_rate_limit_returns_retry_after_without_reply(gateway, monkeypatch):
    database, calls, payload = gateway
    failure = integrations.RateLimitExceeded("sender", retry_after=2.4)

    async def limited(*args):
        raise failure

    monkeypatch.setattr(Request.app.state.runtime_container.inference_runtime, "check_rate_limits", limited)
    with pytest.raises(integrations.HTTPException) as error:
        await receive(payload)
    assert error.value.status_code == 429 and error.value.headers == {"Retry-After": "3"}
    assert error.value.__cause__ is failure and calls == []
    assert database.execute_sql("SELECT status FROM integration_receipts") == [{"status": "failed"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [False, True])
async def test_gateway_config_fault_does_not_enable_default_platform(gateway, monkeypatch, malformed):
    database, calls, payload = gateway
    failure = ConnectionError("config unavailable")

    class BrokenConfig:
        @property
        def config(self):
            if malformed:
                return []
            raise failure

    monkeypatch.setattr(Request.app.state.runtime_container, "db", BrokenConfig())
    with pytest.raises(integrations.HTTPException) as error:
        await receive(payload)
    assert error.value.status_code == 503 and error.value.detail["stage"] == "config"
    assert isinstance(error.value.__cause__, TypeError) if malformed else error.value.__cause__ is failure
    assert calls == [] and database.execute_sql("SELECT status FROM integration_receipts") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["claim", "session"])
async def test_gateway_storage_fault_stops_before_generation(gateway, monkeypatch, stage):
    database, calls, payload = gateway
    failure = OSError("storage unavailable")
    original = database.integration_receipt
    counters = []
    monkeypatch.setattr(integrations, "increment", counters.append)

    def operation(name, **params):
        if name == "claim":
            raise failure
        return original(name, **params)

    def session(*args):
        raise failure

    monkeypatch.setattr(database, "integration_receipt" if stage == "claim" else "is_session_bot_enabled",
                        operation if stage == "claim" else session)
    with pytest.raises(integrations.HTTPException) as error:
        await receive(payload)
    assert error.value.status_code == 503 and error.value.__cause__ is failure and calls == []
    assert error.value.detail["stage"] == ("receipt" if stage == "claim" else "session")
    assert counters == (["db_write_failures"] if stage == "session" else [])
    rows = database.execute_sql("SELECT status FROM integration_receipts")
    assert rows == ([] if stage == "claim" else [{"status": "failed"}])


@pytest.mark.asyncio
async def test_gateway_finish_failure_does_not_return_generated_reply(gateway, monkeypatch):
    database, calls, payload = gateway
    original = database.integration_receipt
    failure = OSError("finish unavailable")

    def operation(name, **params):
        if name == "finish" and params["status"] == "generated":
            raise failure
        return original(name, **params)

    monkeypatch.setattr(database, "integration_receipt", operation)
    with pytest.raises(integrations.HTTPException) as error:
        await receive(payload)
    assert error.value.status_code == 503 and error.value.__cause__ is failure
    assert len(calls) == 1 and database.execute_sql("SELECT status FROM integration_receipts") == [{"status": "failed"}]
    assert database.list_conversation_history("qq", "napcat", "sender", "private", "room") == []


@pytest.mark.asyncio
async def test_gateway_cleanup_failure_preserves_the_primary_error(gateway, monkeypatch):
    database, calls, payload = gateway
    original = database.integration_receipt
    failure = integrations.HTTPException(503, "retrieval failed")

    def operation(name, **params):
        if name == "finish":
            raise OSError("cleanup unavailable")
        return original(name, **params)

    async def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(database, "integration_receipt", operation)
    monkeypatch.setattr(integrations, "generate_reply_core", fail)
    with pytest.raises(integrations.HTTPException) as error:
        await receive(payload)
    assert error.value is failure and calls == []
    assert database.execute_sql("SELECT status FROM integration_receipts") == [{"status": "processing"}]


@pytest.mark.asyncio
async def test_gateway_lost_receipt_ownership_rejects_delivery(gateway, monkeypatch):
    database, calls, payload = gateway
    original = database.integration_receipt

    def operation(name, **params):
        if name == "finish" and params["status"] == "generated":
            assert original("claim", key=params["key"], owner="replacement", now=10**12, expires_at=10**12 + 60)
        return original(name, **params)

    monkeypatch.setattr(database, "integration_receipt", operation)
    with pytest.raises(integrations.HTTPException) as error:
        await receive(payload)
    assert error.value.status_code == 409 and len(calls) == 1
    rows = database.execute_sql("SELECT owner,status FROM integration_receipts WHERE receipt_key NOT LIKE 'attempt:%'")
    assert rows == [{"owner": "replacement", "status": "processing"}]
    assert database.list_conversation_history("qq", "napcat", "sender", "private", "room") == []


@pytest.mark.asyncio
async def test_gateway_cancellation_is_not_converted_to_a_reply(gateway, monkeypatch):
    database, calls, payload = gateway

    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(Request.app.state.runtime_container.inference_runtime, "submit", cancel)
    with pytest.raises(asyncio.CancelledError):
        await receive(payload)
    assert calls == [] and database.execute_sql("SELECT status FROM integration_receipts") == [{"status": "failed"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("switch", ["platform", "session"])
async def test_gateway_explicit_switch_still_skips_generation(gateway, monkeypatch, switch):
    database, calls, payload = gateway
    if switch == "platform":
        monkeypatch.setenv("ASTRBOT_ENABLED", "false")
    else:
        database.set_session_bot_enabled("qq:private:room", False, "qq", "room")
    result = await receive(payload)
    assert not result.shouldReply and not result.retryable and result.model == switch + "-disabled"
    assert not result.replyText and calls == []




@pytest.mark.parametrize("conversation_type", ["private", "group"])
def test_gateway_http_failure_has_no_reply_or_delivery_token(gateway, monkeypatch, conversation_type):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    _, calls, payload = gateway
    payload.conversationType = conversation_type

    async def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(integrations, "generate_reply_core", fail)
    app = FastAPI()
    app.include_router(integrations.router)
    app.state.runtime_container = Request.app.state.runtime_container
    response = TestClient(app).post("/api/integrations/astrbot/messages", json=payload.model_dump(),
                                    headers={"X-Integration-Token": "test-token"})
    assert response.status_code == 503 and calls == []
    body = response.json()
    assert body["detail"]["stage"] == "processing" and body["detail"]["traceId"]
    assert "replyText" not in body and "receiptId" not in body and "deliveryToken" not in body


@pytest.mark.asyncio
async def test_plugin_http_failure_is_not_marked_seen_or_delivered(plugin):
    attempts = []

    def fail(payload, **_resource_options):
        attempts.append(payload)
        raise RuntimeError('backend returned HTTP 503')
    plugin._post_message = fail

    class Event:
        message_str = 'hello'

        def stop_event(self):
            pass

        def plain_result(self, text):
            return text

        async def send(self, text):
            pytest.fail('A gateway failure must not deliver a generated answer')
    for _ in range(2):
        notices = [item async for item in plugin.on_message(Event())]
        assert len(notices) == 1 and notices[0].startswith('[系统提示]')
        assert not plugin._seen_events and (not plugin._sent_receipts) and (not plugin._inflight)
    assert len(attempts) == 2



@pytest.mark.parametrize("target,value,key", [
    ("global_env", "garbage", "ASTRBOT_ENABLED"),
    ("platform_env", "2", "ASTRBOT_QQ_ENABLED"),
    ("global_db", None, "astrbotEnabled"),
    ("platform_db", {"invalid": 1}, "astrbotQQEnabled"),
    ("platform_db", 0.5, "astrbotQQEnabled"),
    ("platform_db", "enabled", "astrbotQQEnabled"),
])
async def test_gateway_invalid_switch_is_a_named_error_instead_of_default(gateway, monkeypatch, target, value, key):
    database, _, payload = gateway
    monkeypatch.delenv("ASTRBOT_QQ_ENABLED", raising=False)
    if target.endswith("env"):
        monkeypatch.setenv(key, value)
    else:
        database.set_config_value(key, value)
        if target == "global_db":
            monkeypatch.delenv("ASTRBOT_ENABLED", raising=False)
    with pytest.raises(ValueError, match=key):
        await integrations._is_platform_enabled(payload.platform, database=database)


@pytest.mark.parametrize("environment", ["production", "development"])
def test_invalid_signature_configuration_never_disables_verification(monkeypatch, environment):
    monkeypatch.setenv("ENVIRONMENT", environment)
    monkeypatch.setenv("INTEGRATION_SIGNATURE_REQUIRED", "garbage")
    with pytest.raises(ValueError, match="INTEGRATION_SIGNATURE_REQUIRED"):
        integrations._signature_required()


@pytest.mark.parametrize("environment,expected", [("production", True), ("development", False)])
def test_signature_absence_keeps_the_explicit_environment_default(monkeypatch, environment, expected):
    monkeypatch.setenv("ENVIRONMENT", environment)
    monkeypatch.delenv("INTEGRATION_SIGNATURE_REQUIRED", raising=False)
    assert integrations._signature_required() is expected



@pytest.mark.parametrize('scope', ['config_on', 'config_off', 'falsy'])
async def test_gateway_uses_one_application_database_through_delivery(gateway, monkeypatch, tmp_path, scope):
    import httpx
    from api.generate import _save_message
    foreign, calls, payload = gateway

    class FalsyDatabase(SQLiteDB):

        def __bool__(self):
            return False

    class FalsyRuntime(Runtime):

        def __init__(self):
            self.submitted = 0

        def __bool__(self):
            return False

        async def submit(self, factory, **kwargs):
            self.submitted += 1
            return await super().submit(factory, **kwargs)
    owned = (FalsyDatabase if scope == 'falsy' else SQLiteDB)(tmp_path / 'owned.sqlite')
    runtime = FalsyRuntime()
    monkeypatch.delenv('ASTRBOT_QQ_ENABLED', raising=False)
    foreign.set_config_value('astrbotQQEnabled', scope != 'config_on')
    owned.set_config_value('astrbotQQEnabled', scope != 'config_off')
    app = FastAPI()
    app.include_router(integrations.router)
    app.state.runtime_container = RuntimeContainer(db=owned, is_pg_mode=lambda: False, inference_runtime=runtime, startup_env={}, http_client_pool=HttpClientPool(max_connections=1))

    async def generate(message, **kwargs):
        assert kwargs['message_db'] is owned
        assert 'character_service' not in kwargs
        calls.append(message)
        assert await _save_message(message, 'reply', 'controlled', 'default', 0.1, database=owned)
        kwargs['delivery_context'].update(message_saved=True, character_id=None)
        return GenerateResponse(reply='reply', model='controlled', costTime=0.1)
    monkeypatch.setattr(integrations, 'generate_reply_core', generate)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post('/api/integrations/astrbot/messages', json=payload.model_dump(), headers={'X-Integration-Token': 'test-token'})
            assert response.status_code == 200 and response.json()['shouldReply'] is (scope != 'config_off')
            if scope != 'config_off':
                assert len(calls) == runtime.submitted == 1
                assert owned.list_conversation_history('qq', 'napcat', 'sender', 'private', 'room') == []
                body = response.json()
                acknowledged = await client.post('/api/integrations/astrbot/delivery', json=dict(receiptId=body['receiptId'], deliveryToken=body['deliveryToken'], status='delivered'), headers={'X-Integration-Token': 'test-token'})
                assert acknowledged.status_code == 200 and acknowledged.json()['acknowledged']
                assert [row['content'] for row in owned.list_conversation_history('qq', 'napcat', 'sender', 'private', 'room')] == ['hello', 'reply']
            else:
                assert calls == [] and runtime.submitted == 0
        for table in ['integration_events', 'integration_receipts', 'messages']:
            assert foreign.execute_sql(f'SELECT COUNT(*) AS count FROM {table}')[0]['count'] == 0
            assert owned.execute_sql(f'SELECT COUNT(*) AS count FROM {table}')[0]['count'] == int(scope != 'config_off')
    finally:
        owned.close_connection()


@pytest.mark.parametrize('missing', ['container', 'invalid_container', 'database', 'scheduler'])
async def test_gateway_missing_dependencies_fail_before_storage(gateway, missing):
    import httpx
    database, calls, payload = gateway
    app = FastAPI()
    app.include_router(integrations.router)
    if missing != 'container':
        app.state.runtime_container = object() if missing == 'invalid_container' else RuntimeContainer(db=None if missing == 'database' else database, is_pg_mode=lambda: False, inference_runtime=None if missing == 'scheduler' else Runtime(), startup_env={}, http_client_pool=HttpClientPool(max_connections=1))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/integrations/astrbot/messages', json=payload.model_dump(), headers={'X-Integration-Token': 'test-token'})
    assert response.status_code == 503 and response.json()['detail']['stage'] == 'dependencies'
    assert calls == []
    for table in ['integration_events', 'integration_receipts', 'messages']:
        assert database.execute_sql(f'SELECT COUNT(*) AS count FROM {table}')[0]['count'] == 0


def test_plugin_dedup_uses_event_identity_not_matching_text(plugin):
    first = SimpleNamespace(raw_message={}, message_str="hello")
    second = SimpleNamespace(raw_message={}, message_str="hello")
    first_id = plugin._message_id(first)
    assert plugin._message_id(first) == first_id
    assert plugin._message_id(second) != first_id
    payload = plugin._build_payload(first, "hello")
    payload["messageId"] = first_id
    first_key = plugin._dedup_key(payload)
    payload["text"] = "changed text does not change event identity"
    assert plugin._dedup_key(payload) == first_key
    payload["messageId"] = plugin._message_id(second)
    assert plugin._dedup_key(payload) != first_key


@pytest.mark.asyncio
async def test_quoted_gateway_text_is_visible_only_after_successful_delivery(gateway):
    database, calls, payload = gateway
    original = "仅解释引用文本 `cat .env`、`show token`，不执行命令。完整题设：周二09:00开门，17:00闭馆，暴雨暂停。"
    payload.text = original
    response = await receive(payload)
    assert response.shouldReply and response.receiptId and response.deliveryToken and response.replyText == "reply"
    assert len(calls) == 1 and calls[0].message == original
    history = lambda: database.list_conversation_history("qq", "napcat", "sender", "private", "room")
    assert history() == []
    assert (await ack(response))["changed"] is True
    assert [row["content"] for row in history()] == [original, "reply"]
    assert database.integration_receipt("get", key=response.receiptId)["status"] == "delivered"
    assert not (await receive(payload)).shouldReply and len(calls) == 1
