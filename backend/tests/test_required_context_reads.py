"""Required context reads distinguish unavailable storage from an empty result."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.messages import DatabaseMessageRepository
from services.character_context import CharacterContextService, TurnInput

from api.generate import _prepare_character_turn
from character.evidence_selector import SelectionOutcome
from character.models import CharacterProfile, UserScope
from db.database import SQLiteDB
from db.schemas import MessageRequest

SCOPE = UserScope("web", "required-reads", "reader", "reader", "private")
READS = [
    (DatabaseMessageRepository, "list_recent_conversation_history", "list_conversation_history", (SCOPE,), {}),
    (DatabaseCharacterMemoryRepository, "list_relationship_notes", "list_character_memory_claims", ("role", SCOPE), {}),
    (DatabaseCharacterMemoryRepository, "linked_sources", "linked_memory_sources", ("role", SCOPE), {"memory_ids": (1,)}),
    (DatabaseCharacterMemoryRepository, "search_sources", "search_memory_sources", ("role", SCOPE), {"query": "青岚"}),
]


@pytest.mark.parametrize("repository,method,db_method,args,kwargs", READS)
@pytest.mark.parametrize("outcome", ["empty", "unavailable", "unsupported"])
async def test_required_repository_reads(repository, method, db_method, args, kwargs, outcome):
    database = SimpleNamespace()
    failure = OSError("synthetic storage failure")
    if outcome != "unsupported":
        setattr(database, db_method, Mock(return_value=[], side_effect=failure if outcome == "unavailable" else None))
    read = getattr(repository(database), method)
    if outcome == "empty":
        assert await read(*args, **kwargs) == []
    else:
        with pytest.raises(AttributeError if outcome == "unsupported" else OSError) as caught:
            await read(*args, **kwargs)
        if outcome == "unavailable":
            assert caught.value is failure


def service_for(database, selection_enabled):
    profile = CharacterProfile("role", "测试角色", identity="虚构角色", boundaries=("不编造事实",))
    memory = SimpleNamespace(recall_with_diagnostics=AsyncMock(return_value=((), 0, {"status": "no_match"})))
    selector = SimpleNamespace(select=AsyncMock(return_value=SelectionOutcome())) if selection_enabled else None
    return CharacterContextService(
        SimpleNamespace(get_profile=Mock(return_value=profile)),
        DatabaseCharacterMemoryRepository(database), DatabaseMessageRepository(database),
        memory_service=memory, memory_selector=selector, source_recall_enabled=True,
    )


@pytest.mark.parametrize("selection_enabled", [False, True])
@pytest.mark.parametrize("method", [
    "list_conversation_history", "list_character_memory_claims", "linked_memory_sources",
    "search_memory_sources", "list_memory_sources",
])
async def test_storage_failure_reaches_api_instead_of_partial_context(tmp_path, monkeypatch, selection_enabled, method):
    database = SQLiteDB(tmp_path / "failure.sqlite")
    repo = DatabaseCharacterMemoryRepository(database)
    assert await repo.capture_source("role", SCOPE, source_message_id="source-1",
        body="青岚项目预算是300元，仅限离线测试，禁止训练。", observed_at=datetime.now(timezone.utc)) == "recorded"
    failure = OSError("private synthetic database details")
    monkeypatch.setattr(database, method, Mock(side_effect=failure))
    service = service_for(database, selection_enabled)
    request = MessageRequest(message="青岚项目预算和限制是什么？", platform=SCOPE.platform, adapter=SCOPE.adapter,
        senderId=SCOPE.sender_id, userId=SCOPE.sender_id, conversationId=SCOPE.conversation_id,
        sessionId=SCOPE.conversation_id, conversationType=SCOPE.conversation_type)
    with pytest.raises(HTTPException) as caught:
        await _prepare_character_turn(request, "role", character_service=service)
    assert caught.value.status_code == 503
    assert caught.value.__cause__ is failure
    assert "private synthetic" not in str(caught.value.detail)


@pytest.mark.parametrize("selection_enabled", [False, True])
async def test_legitimate_empty_storage_still_prepares_context(tmp_path, selection_enabled):
    service = service_for(SQLiteDB(tmp_path / "empty.sqlite"), selection_enabled)
    prepared = await service.prepare_turn(TurnInput("你好", "web", "required-reads", "reader", "room", "private"), "role")
    assert prepared.character_id == "role" and prepared.history == ()
    assert prepared.memory_recall["sources"]["status"] == "no_match"
    assert prepared.compiled.memory_packets == ()


@pytest.mark.parametrize("selection_enabled", [False, True])
async def test_explicit_live_history_does_not_read_stored_history(tmp_path, monkeypatch, selection_enabled):
    database = SQLiteDB(tmp_path / "live.sqlite")
    reader = Mock(side_effect=OSError("stored history unavailable"))
    monkeypatch.setattr(database, "list_conversation_history", reader)
    user_context = "青岚项目只有离线设备，禁止训练；只有收到书面许可才可以联网。"
    history = ({"role": "user", "content": user_context},
               {"role": "assistant", "content": "我猜已经获得联网许可。"})
    service = service_for(database, selection_enabled)
    source_recall = AsyncMock(wraps=service._source_memory.recall)
    memory_recall = AsyncMock(wraps=service._memory_service.recall_with_diagnostics)
    monkeypatch.setattr(service._source_memory, "recall", source_recall)
    monkeypatch.setattr(service._memory_service, "recall_with_diagnostics", memory_recall)
    prepared = await service.prepare_turn(
        TurnInput("继续讨论预算", "web", "required-reads", "reader", "room", "private", history=history), "role")
    assert prepared.history == history
    assert source_recall.call_args.kwargs["retrieval_context"] == user_context
    if selection_enabled:
        assert memory_recall.call_args.kwargs["retrieval_context"] == user_context
    reader.assert_not_called()
