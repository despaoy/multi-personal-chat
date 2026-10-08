"""Complete synthetic claims exercise required reads through the API boundary."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from fastapi import HTTPException
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.messages import DatabaseMessageRepository
from services.character_context import CharacterContextService

from api.generate import _prepare_character_turn
from character.memory_service import CharacterMemoryService
from character.models import CharacterProfile, MemoryItem, UserScope
from db.database import SQLiteDB
from db.schemas import MessageRequest

SCOPE = UserScope("web", "strict-memory", "reader", "reader", "private")
QUERY = "我的大学专业是什么？"


@pytest.fixture
async def repository(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "memory.sqlite"))
    source = "我的大学专业是环境工程，这是我本人的专业。"
    stamp = datetime.now(timezone.utc) - timedelta(seconds=1)
    await repo.capture_source("role", SCOPE, source_message_id="source-1", body=source, observed_at=stamp)
    await repo.append_claim("role", SCOPE, MemoryItem("", "user_fact", "大学专业是环境工程", 0.9),
        memory_key="user_major", confidence=0.98, source_message_id="source-1", evidence=(source,),
        observed_at=stamp.isoformat(), metadata={"temporal_provenance": {
            "version": 1, "producer": "semantic_memory", "validity_authority": "unverified"}})
    return repo


@pytest.mark.parametrize("method", ["list_memory_records", "linked_source_revisions", "linked_source_receipts"])
async def test_required_read_failure_stops_api_without_retry(repository, monkeypatch, method):
    failure = TypeError("unexpected keyword include_inactive; synthetic private details")
    reader = AsyncMock(side_effect=failure)
    monkeypatch.setattr(repository, method, reader)
    profile = CharacterProfile("role", "虚构角色", identity="测试角色", boundaries=("不编造事实",))
    service = CharacterContextService(SimpleNamespace(get_profile=Mock(return_value=profile)), repository,
        DatabaseMessageRepository(repository._database),
        memory_service=CharacterMemoryService(repository, semantic_enabled=False), source_recall_enabled=False)
    request = MessageRequest(message=QUERY, platform=SCOPE.platform, adapter=SCOPE.adapter,
        userId=SCOPE.sender_id, sessionId=SCOPE.conversation_id, conversationType="private")
    with pytest.raises(HTTPException) as caught:
        await _prepare_character_turn(request, "role", character_service=service)
    assert caught.value.status_code == 503 and caught.value.__cause__ is failure
    assert "synthetic private" not in str(caught.value.detail)
    reader.assert_awaited_once()


@pytest.mark.parametrize("missing", ["linked_source_revisions", "linked_source_receipts"])
async def test_missing_exact_reader_never_uses_broad_source_fallback(repository, missing):
    methods = {name: getattr(repository, name) for name in (
        "list_memory_records", "linked_source_revisions", "linked_source_receipts") if name != missing}
    broad_reader = AsyncMock(return_value=[])
    adapter = SimpleNamespace(**methods, linked_sources=broad_reader)
    with pytest.raises(AttributeError, match=missing):
        await CharacterMemoryService(adapter, semantic_enabled=False).recall_with_diagnostics("role", SCOPE, QUERY)
    broad_reader.assert_not_awaited()


async def test_embedding_failure_does_not_reuse_cached_success(repository):
    provider = SimpleNamespace(embed_texts=Mock(side_effect=lambda texts: np.ones((len(texts), 2), dtype=np.float32)))
    service = CharacterMemoryService(repository, semantic_enabled=True, embedding_provider=provider)
    items, count, trace = await service.recall_with_diagnostics("role", SCOPE, QUERY)
    assert count == 1 and items and trace["semantic_status"] == "available"
    assert trace["claim_revision_status"] == trace["temporal_source_status"] == "available"
    failure = TimeoutError("synthetic embedding timeout")
    provider.embed_texts.side_effect = failure
    with pytest.raises(TimeoutError, match="synthetic embedding timeout"):
        await service.recall_with_diagnostics("role", SCOPE, QUERY)
    assert provider.embed_texts.call_count == 2


@pytest.mark.parametrize("shape", ["zero", "empty", "wrong_rows"])
async def test_invalid_embedding_output_cannot_become_successful_recall(repository, shape):
    def encode(texts):
        if shape == "empty":
            return np.zeros((len(texts), 0), dtype=np.float32)
        if shape == "wrong_rows":
            return np.ones((len(texts) + 1, 2), dtype=np.float32)
        return np.zeros((len(texts), 2), dtype=np.float32)
    service = CharacterMemoryService(repository, semantic_enabled=True,
        embedding_provider=SimpleNamespace(embed_texts=encode))
    with pytest.raises(ValueError, match="zero norm|形状不正确"):
        await service.recall_with_diagnostics("role", SCOPE, QUERY)
