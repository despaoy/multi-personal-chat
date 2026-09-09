"""Gateway retries must reuse generation and expose only delivered history."""

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

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
    database = SQLiteDB(tmp_path / "delivery.db")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("ASTRBOT_ENABLED", "true")
    monkeypatch.setenv("INTEGRATION_SIGNATURE_REQUIRED", "false")
    monkeypatch.setenv("ASTRBOT_INTEGRATION_TOKEN", "test-token")
    monkeypatch.setattr(integrations, "db", database)
    monkeypatch.setattr(integrations, "inference_runtime", Runtime())
    calls = []

    async def generate(msg, **kwargs):
        from api.generate import _save_message

        calls.append(msg)
        assert await _save_message(msg, "reply", "model", "default", 0.1, database=database)
        kwargs["delivery_context"].update(message_saved=True, character_id=None)
        return GenerateResponse(reply="reply", costTime=0.1)

    monkeypatch.setattr(integrations, "generate_reply_core", generate)
    payload = integrations.AstrBotMessageRequest(
        platform="qq",
        adapter="napcat",
        messageId="m1",
        conversationId="room",
        senderId="sender",
        text="hello",
    )
    return database, calls, payload


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
    assert (await receive(payload)).retryable
    cancelled = asyncio.Event()

    async def slow(*args, **kwargs):
        try:
            await asyncio.sleep(1)
        finally:
            cancelled.set()

    monkeypatch.setattr(integrations, "generate_reply_core", slow)
    monkeypatch.setattr(integrations, "_MODEL_TIMEOUT", 0.05)
    assert (await receive(payload)).retryable
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
async def test_character_completion_waits_for_ack_and_runs_once(gateway, monkeypatch):
    _, _, payload = gateway
    original = integrations.generate_reply_core
    completed = []

    async def generate(*args, **kwargs):
        result = await original(*args, **kwargs)
        kwargs["delivery_context"]["character_id"] = "character"
        return result

    async def prepare(*args, **kwargs):
        return object()

    async def complete(*args, **kwargs):
        completed.append(args)

    monkeypatch.setattr(integrations, "generate_reply_core", generate)
    monkeypatch.setattr(integrations, "_prepare_character_turn", prepare)
    monkeypatch.setattr(integrations, "_complete_character_turn", complete)
    response = await receive(payload)
    assert completed == []
    await ack(response, "delivery_failed")
    assert completed == []
    await ack(response)
    await ack(response)
    assert len(completed) == 1


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
