"""Atomic new source admission, handoff, deletion and failed-generation retry."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from db.database import SQLiteDB
from db.memory_source import source_identity, source_scope
from db.schemas import MessageRequest

FIELDS = ('role', 'web', 'web-character', 'alice', 'private', 'alice')
WHEN = datetime(2026, 10, 1, tzinfo=timezone.utc)
BODY = '我的朋友收到合成测试课程预约确认，回执TK-287，尚未参加或出发；不是我本人的预约。'


@pytest.fixture
def database(tmp_path):
    return SQLiteDB(tmp_path / 'reservation.sqlite')


def reserve(database, body=BODY, when=WHEN, source_id='new-identity', fields=FIELDS):
    return database.reserve_memory_source(*fields, source_message_id=source_id, body=body, observed_at=when)


def row(database):
    key = source_identity(source_scope(*FIELDS), 'new-identity')['source_key']
    return dict(database._get_connection().execute('SELECT * FROM memory_sources WHERE source_key=?', (key,)).fetchone())


def test_two_threads_bind_only_one_complete_text(database):
    barrier = Barrier(2)

    def bind(body):
        barrier.wait(timeout=3)
        return body, reserve(database, body)

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(bind, body) for body in [BODY, '另一条完整合成陈述：朋友预约另一门课程，回执TX-663。']]
        results = [job.result(timeout=5) for job in jobs]
    assert sorted(receipt['status'] for _, receipt in results) == ['conflict', 'pending']
    winner = next(body for body, receipt in results if receipt['status'] == 'pending')
    assert database.memory_source_admission(*FIELDS, source_message_id='new-identity', body=winner) == 'pending'
    assert row(database)['body'] is None and row(database)['state'] == 'pending'
    assert database.list_memory_sources(*FIELDS) == []


def test_same_text_retry_after_generation_failure_reuses_receipt(database):
    first = reserve(database)
    assert reserve(database, when=WHEN + timedelta(seconds=20)) == first
    assert row(database)['observed_at'] == WHEN.isoformat(timespec='microseconds')


def test_completion_uses_exact_reserved_clock_and_removes_digest(database):
    receipt = reserve(database)
    stamp = datetime.fromisoformat(receipt['observed_at'])
    assert database.capture_memory_source(*FIELDS, source_message_id='new-identity', body=BODY, observed_at=stamp) == 'recorded'
    assert row(database)['body_digest'] is None and row(database)['body'] == BODY


@pytest.mark.parametrize('changed', ['body', 'clock'])
def test_writer_cannot_replace_pending_binding(database, changed):
    reserve(database)
    body = '不同的完整陈述。' if changed == 'body' else BODY
    when = WHEN + timedelta(seconds=1) if changed == 'clock' else WHEN
    assert database.capture_memory_source(*FIELDS, source_message_id='new-identity', body=body, observed_at=when) == 'conflict'
    assert row(database)['state'] == 'pending' and row(database)['body'] is None


def test_source_only_erasure_clears_pending_digest_and_blocks_retry(database):
    reserve(database)
    assert database.erase_unlinked_memory_sources(*FIELDS, source_message_ids=['new-identity']) == 1
    assert row(database)['body_digest'] is None and row(database)['observed_at'] is None
    assert reserve(database, when=WHEN + timedelta(days=1))['status'] == 'revoked'


def test_owner_fence_blocks_old_pending_capture_even_with_new_worker_clock(database):
    reserve(database)
    owner = source_scope(*FIELDS)['owner_key']
    database._get_connection().execute('UPDATE memory_source_fences SET revoked_before=? WHERE owner_key=?', ((WHEN + timedelta(seconds=1)).isoformat(timespec='microseconds'), owner))
    database._get_connection().commit()
    assert reserve(database, when=WHEN + timedelta(seconds=3))['status'] == 'stale'
    assert database.capture_memory_source(*FIELDS, source_message_id='new-identity', body=BODY, observed_at=WHEN + timedelta(seconds=3)) == 'stale'


def test_exact_other_owner_can_bind_same_external_id(database):
    reserve(database)
    assert reserve(database, body='另一个账户的完整合成陈述。', fields=('role', 'web', 'web-character', 'bob', 'private', 'bob'))['status'] == 'pending'


def test_claim_link_legacy_pending_anchor_can_be_bound(database):
    identity = source_identity(source_scope(*FIELDS), 'new-identity')
    database._get_connection().execute("INSERT INTO memory_sources(source_key,owner_key,scope_key,source_message_id,state) VALUES(?,?,?,?, 'pending')", tuple(identity[k] for k in ['source_key', 'owner_key', 'scope_key', 'source_message_id']))
    database._get_connection().commit()
    assert reserve(database)['status'] == 'pending' and row(database)['body_digest']


@pytest.mark.asyncio
@pytest.mark.parametrize('message', ['不要记住，我喜欢素描。', '请彻底忘掉我的课程预约记忆。'])
async def test_opt_out_and_erasure_do_not_create_bindings(monkeypatch, message):
    from api import generate

    writer = Mock(side_effect=AssertionError('must not bind'))
    db = SimpleNamespace(memory_source_admission=Mock(return_value='new'), reserve_memory_source=writer)
    request = MessageRequest(message=message, characterId='role', platform='web', adapter='web-character', sourceMessageId='receipt')
    await generate._validate_web_source_identity(request, {'id': 'alice'}, database=db)
    writer.assert_not_called()
    assert request._source_received_at is None


@pytest.mark.asyncio
async def test_source_reservation_failure_prevents_model(monkeypatch):
    from api import generate

    handler = AsyncMock()
    monkeypatch.setattr(generate, '_generate_reply_impl', handler)
    db = SimpleNamespace(memory_source_admission=Mock(return_value='new'), reserve_memory_source=Mock(side_effect=RuntimeError('failed')))
    chat = generate._build_chat_generation_service(None, message_db=db)
    request = MessageRequest(message=BODY, characterId='role', platform='web', adapter='web-character', sourceMessageId='receipt')
    with pytest.raises(HTTPException) as error:
        await chat.generate(request, {'id': 'alice'})
    assert error.value.status_code == 503
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_reserved_receipt_is_private_and_reaches_actual_preparation(monkeypatch):
    from api import generate

    request = MessageRequest(message=BODY, characterId='role', platform='web', adapter='web-character', sourceMessageId='receipt', _source_received_at='2099-01-01T00:00:00+00:00')
    assert request._source_received_at is None
    db = SimpleNamespace(memory_source_admission=Mock(return_value='new'), reserve_memory_source=Mock(return_value={'status': 'pending', 'observed_at': WHEN.isoformat()}))
    await generate._validate_web_source_identity(request, {'id': 'alice'}, database=db)
    prepare = AsyncMock(return_value='prepared')
    service = SimpleNamespace(prepare_turn=prepare)
    assert await generate._prepare_character_turn(request, 'role', character_service=service) == 'prepared'
    assert prepare.call_args.args[0].received_at == WHEN
    assert '_source_received_at' not in request.model_dump()


def test_existing_database_additive_upgrade_keeps_original_source(database):
    assert database.capture_memory_source(*FIELDS, source_message_id='new-identity', body=BODY, observed_at=WHEN) == 'recorded'
    original = row(database)
    connection = database._get_connection()
    connection.execute('ALTER TABLE memory_sources DROP COLUMN body_digest')
    connection.commit()
    reopened = SQLiteDB(database.db_path)
    upgraded = row(reopened)
    assert upgraded == original and upgraded['body_digest'] is None


def test_scope_clear_erases_pending_digest_without_resurrection(database):
    from db.memory_source import lock_owner, revoke_plan, run_sqlite

    reserve(database)
    connection = database._get_connection()
    cursor = connection.cursor()
    cursor.execute('BEGIN IMMEDIATE')
    scope = source_scope(*FIELDS)
    run_sqlite(cursor, lock_owner(scope))
    run_sqlite(cursor, revoke_plan(scope, (), clear=True))
    connection.commit()
    assert row(database)['state'] == 'revoked' and row(database)['body_digest'] is None
    assert reserve(database, when=WHEN + timedelta(days=3))['status'] == 'revoked'
