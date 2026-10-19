"""A model scope request is either authorized as requested or explicitly rejected."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, MemoryResponseError, MemoryScopeAuthorization, parse_llm_proposals, build_memory_llm_messages
from character.models import UserScope
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository


def entry(**extra):
    return dict(kind="name", value="清舟", evidence="我叫清舟", confidence=.98, operation="ADD", attributed_to="user", **extra)


def parse(item, source="我叫清舟。", authorization=None):
    return parse_llm_proposals(json.dumps(dict(memories=[item]), ensure_ascii=False), source_message=source, scope_authorization=authorization)


@pytest.mark.parametrize("scope,source,allowed", [
    ("conversation", "我叫清舟。", ["conversation"]),
    ("conversation", "请记住我叫清舟，这个姓名一直没变。", ["conversation"]),
    ("user_character", "请记住我叫清舟，以后你记住这个姓名。", ["conversation", "user_character"]),
    ("user_global", "我叫清舟，请所有角色记住这个姓名。", ["conversation", "user_global"]),
])
def test_authorized_scope_is_kept(scope, source, allowed):
    authorization = MemoryScopeAuthorization(source, tuple(allowed))
    proposal, = parse(entry(scope_level=scope), source, authorization)
    assert proposal.scope_level == scope
    messages = build_memory_llm_messages(source, (), (), (), 2000, .85, scope_authorization=authorization)
    assert json.loads(messages[1]["content"])["proposal_constraints"]["scope_levels"] == allowed


def test_omitted_optional_scope_is_conversation():
    proposal, = parse(entry())
    assert proposal.scope_level == "conversation"


@pytest.mark.parametrize("scope", ["unknown", None, False, "user_character", "user_global"])
def test_invalid_or_unauthorized_scope_cannot_be_saved_as_conversation(scope):
    with pytest.raises(MemoryResponseError) as caught:
        parse(entry(scope_level=scope))
    assert caught.value.field == "scope_level"


@pytest.mark.asyncio
async def test_scope_failure_rejects_whole_batch_without_partial_claims(tmp_path):
    database=SQLiteDB(tmp_path / "scope.sqlite")
    repository=DatabaseCharacterMemoryRepository(database)
    scope=UserScope("web", "scope-contract", "synthetic-owner", "synthetic-owner", "private")
    payload=json.dumps(dict(memories=[entry(), entry(scope_level="user_global")]))
    completion=SimpleNamespace(complete=AsyncMock(return_value=payload), close=AsyncMock())
    worker=MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "fixture"), completion=completion)
    try:
        receipt=await worker.schedule_and_wait(repository=repository, character_id="role", user_scope=scope,
            message="我叫清舟。", rule_hints=(), source_message_id="source", immediate=True)
        assert receipt["status"] == "failed" and receipt["stage"] == "proposal_validation"
        assert receipt["field"] == "scope_level" and receipt["reason"] == "invalid_model_response"
        assert receipt["accepted"] == receipt["persisted"] == 0
        assert await repository.list_memory_records("role", scope) == []
        assert len(await repository.list_sources("role", scope)) == 1
        completion.complete.assert_awaited_once()
    finally:
        await worker.shutdown(timeout=1)
        database.close_connection()
