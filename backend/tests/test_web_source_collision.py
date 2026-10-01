from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from db.schemas import MessageRequest


def request(**kwargs):
    return MessageRequest(message='新完整陈述', characterId='tsukiyashiro_kisaki', platform='web', adapter='web-character',
                          senderId='1', userId='1', conversationId='1', sourceMessageId='existing', **kwargs)


def service(monkeypatch, rows=(), *, read_error=None):
    from api import generate

    handler = AsyncMock(return_value='allowed')
    monkeypatch.setattr(generate, '_generate_reply_impl', handler)
    reader = Mock(return_value=list(rows), side_effect=read_error)
    database = SimpleNamespace(list_memory_sources=reader)
    return generate._build_chat_generation_service(None, message_db=database), handler, reader


@pytest.mark.asyncio
async def test_known_source_identity_conflict_rejects_before_handler(monkeypatch):
    chat, handler, reader = service(monkeypatch, [{'body': '原来的另一条完整陈述'}])
    with pytest.raises(HTTPException) as error:
        await chat.generate(request(), {'id': '1'})
    assert error.value.status_code == 409 and error.value.detail['code'] == 'source_identity_conflict'
    handler.assert_not_awaited()
    assert reader.call_args.kwargs == {'source_message_ids': ('existing',), 'limit': 1}


@pytest.mark.asyncio
async def test_same_content_identity_retry_is_allowed(monkeypatch):
    chat, handler, _ = service(monkeypatch, [{'body': '新完整陈述'}])
    assert await chat.generate(request(), {'id': '1'}) == 'allowed'
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_new_source_identity_is_allowed(monkeypatch):
    chat, handler, _ = service(monkeypatch)
    assert await chat.generate(request(), {'id': '1'}) == 'allowed'
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_authenticated_scope_and_injected_database_own_the_lookup(monkeypatch):
    chat, _, reader = service(monkeypatch)
    assert await chat.generate(request(), {'user_id': '2'}) == 'allowed'
    assert reader.call_args.args == ('tsukiyashiro_kisaki', 'web', 'web-character', '2', 'private', '2')


@pytest.mark.asyncio
async def test_source_read_failure_rejects_before_handler(monkeypatch):
    chat, handler, _ = service(monkeypatch, read_error=RuntimeError('database unavailable'))
    with pytest.raises(HTTPException) as error:
        await chat.generate(request(), {'id': '1'})
    assert error.value.status_code == 503
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_stateless_management_does_not_require_source_repository(monkeypatch):
    chat, handler, reader = service(monkeypatch, read_error=AssertionError('must not read'))
    assert await chat.generate(MessageRequest(message='管理台问题'), {'id': '1'}) == 'allowed'
    reader.assert_not_called()
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_branch_owned_path_keeps_its_existing_authority(monkeypatch):
    chat, handler, reader = service(monkeypatch, read_error=AssertionError('must not read'))
    assert await chat.generate(request(branchId='branch-owned'), {'id': '1'}) == 'allowed'
    reader.assert_not_called()
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_source_comparison_uses_same_sanitized_text_as_generation(monkeypatch):
    chat, handler, _ = service(monkeypatch, [{'body': '新完整陈述'}])
    incoming = request()
    incoming.message = '\x00新完整陈述 \n'
    assert await chat.generate(incoming, {'id': '1'}) == 'allowed'
    assert handler.call_args.args[0].message == '新完整陈述'
