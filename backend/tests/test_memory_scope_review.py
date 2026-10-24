"""Scope review narrows candidates; the writer cannot override its decision."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from character.memory_llm import (
    MemoryEnrichmentScheduler, MemoryLlmConfig, MemoryResponseError,
    MemoryScopeAuthorization, build_memory_llm_messages, parse_llm_proposals,
    review_memory_scopes,
)
from character.models import UserScope
from db.database import SQLiteDB
from inference.context_budget import InputBudgetError
from repositories.character_memory import DatabaseCharacterMemoryRepository


LOCAL = "请记住我叫海岚，只在当前会话使用；不要让所有角色记住这个名字。"
ROLE = "请记住我叫海岚，以后这个角色也记住；不要让所有角色记住。"
QUOTED = "我叫海岚。“请所有角色记住这个名字”是我引用别人的一句话，不是我的授权；只在当前会话使用。"
GLOBAL = "我叫海岚，请所有角色记住这个名字。"


def decision(scope="user_global", authorized=False, evidence=LOCAL):
    return dict(scope_level=scope, authorized=authorized, evidence=evidence)


def response(decisions):
    return json.dumps(dict(decisions=decisions), ensure_ascii=False)


def proposal(scope):
    return json.dumps(dict(memories=[dict(kind="name", value="海岚", evidence="我叫海岚",
        confidence=.99, operation="ADD", attributed_to="user", scope_level=scope)]), ensure_ascii=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("source,decisions,expected", [
    (LOCAL, [decision()], ("conversation",)),
    (ROLE, [decision("user_character", True, "以后这个角色也记住"),
            decision(evidence="不要让所有角色记住")], ("conversation", "user_character")),
    (QUOTED, [decision(evidence="不是我的授权；只在当前会话使用")], ("conversation",)),
    (GLOBAL, [decision(authorized=True, evidence="请所有角色记住这个名字")], ("conversation", "user_global")),
])
async def test_one_review_shared_by_request_and_write_validation(source, decisions, expected):
    completion = SimpleNamespace(complete=AsyncMock(return_value=response(decisions)))
    authorization = await review_memory_scopes(completion, source, requested_scopes=tuple(d["scope_level"] for d in decisions))
    assert authorization.scope_levels == expected
    completion.complete.assert_awaited_once()
    review = json.loads(completion.complete.call_args.args[0][1]["content"])
    assert review["current_user_message"] == source
    messages = build_memory_llm_messages(source, (), (), (), 2000, .85, scope_authorization=authorization)
    assert json.loads(messages[1]["content"])["proposal_constraints"]["scope_levels"] == list(expected)
    accepted, = parse_llm_proposals(proposal(expected[-1]), source_message=source, scope_authorization=authorization)
    assert accepted.scope_level == expected[-1]
    if "user_global" not in expected:
        with pytest.raises(MemoryResponseError):
            parse_llm_proposals(proposal("user_global"), source_message=source, scope_authorization=authorization)


@pytest.mark.asyncio
async def test_no_cross_scope_proposal_needs_no_scope_model_call():
    completion = SimpleNamespace(complete=AsyncMock())
    authorization = await review_memory_scopes(completion, "请记住我叫海岚，这个名字一直没变。", requested_scopes=())
    assert authorization.scope_levels == ("conversation",)
    completion.complete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [
    "invalid json", response([]), response([decision(authorized=None)]),
    response([decision(evidence="不存在的引文")]),
    response([decision("user_character")]), response([decision(), decision()]),
    '{"decisions":[],"decisions":[]}',
])
async def test_incomplete_uncertain_or_invalid_review_is_an_error(raw):
    completion = SimpleNamespace(complete=AsyncMock(return_value=raw))
    with pytest.raises(MemoryResponseError):
        await review_memory_scopes(completion, LOCAL, requested_scopes=("user_global",))
    completion.complete.assert_awaited_once()


def test_missing_review_and_review_from_different_source_cannot_authorize():
    for authorization in (None, MemoryScopeAuthorization(GLOBAL + "这是另一条消息。", (("user_global", "请所有角色记住这个名字"),))):
        with pytest.raises(MemoryResponseError):
            parse_llm_proposals(proposal("user_global"), source_message=GLOBAL, scope_authorization=authorization)
        if authorization is not None:
            with pytest.raises(MemoryResponseError):
                build_memory_llm_messages(GLOBAL, (), (), (), 2000, .85, scope_authorization=authorization)
    constraints = json.loads(build_memory_llm_messages(GLOBAL, (), (), (), 2000, .85)[1]["content"])["proposal_constraints"]
    assert constraints["scope_levels"] == ["conversation", "user_character", "user_global"]
    assert constraints["scope_authorizations"] == []


@pytest.mark.asyncio
async def test_scope_review_checks_its_actual_model_budget_before_calling():
    completion = SimpleNamespace(complete=AsyncMock())
    with pytest.raises(InputBudgetError):
        await review_memory_scopes(completion, GLOBAL, context_window_tokens=1500, count_tokens=lambda text: 1000)
    with pytest.raises(InputBudgetError):
        await review_memory_scopes(completion, GLOBAL, max_input_chars=2)
    completion.complete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("granted,writer_scope,expected_status,expected_stage,expected_calls", [
    (False, "conversation", "saved", "persistence", 1),
    (True, "user_global", "saved", "persistence", 2),
    (False, "user_global", "failed", "proposal_validation", 2),
    (None, "user_global", "failed", "scope_review", 2),
])
async def test_review_and_writer_success_and_failure_are_truthful(tmp_path, granted, writer_scope,
        expected_status, expected_stage, expected_calls):
    source = GLOBAL if granted else LOCAL
    completion = SimpleNamespace(complete=AsyncMock(side_effect=[
        proposal(writer_scope), response([decision(authorized=granted, evidence=source)]),
    ]), close=AsyncMock())
    database = SQLiteDB(tmp_path / "scope-review.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("web", "scope-review", "synthetic-owner", "synthetic-owner", "private")
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "fixture"), completion=completion)
    try:
        receipt = await worker.schedule_and_wait(repository=repository, character_id="role", user_scope=scope,
            message=source, rule_hints=(), source_message_id="source", immediate=True)
        assert receipt["status"] == expected_status and receipt["stage"] == expected_stage
        assert completion.complete.await_count == expected_calls
        sources = await repository.list_sources("role", scope)
        assert len(sources) == 1 and sources[0]["body"] == source
        records = await repository.list_memory_records("role", scope)
        if expected_status == "saved":
            assert receipt["persisted"] == 1 and len(records) == 1
            assert records[0]["scope_level"] == writer_scope
        else:
            assert receipt["accepted"] == receipt["persisted"] == 0 and records == []
            assert receipt["retryable"] is False and receipt["reason"] == "invalid_model_response"
    finally:
        await worker.shutdown(timeout=1)
        database.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope,source,decisions", [
    ("user_global", "请记住：我叫海岚，请所有角色记住我的名字；我的专业是工业设计，这项专业信息只在当前会话中使用，不要跨会话或与其他角色共享。",
     [decision(authorized=True, evidence="请所有角色记住我的名字")]),
    ("user_character", "请记住：我叫海岚，以后这个角色也记住我的名字；我的专业是工业设计，这项专业只在当前会话中使用，不要让所有角色记住。",
     [decision("user_character", True, "以后这个角色也记住我的名字"), decision(evidence="不要让所有角色记住")]),
])
async def test_authorized_fact_quote_reaches_writer_without_expanding_other_facts(scope, source, decisions):
    completion = SimpleNamespace(complete=AsyncMock(return_value=response(decisions)))
    authorization = await review_memory_scopes(completion, source, requested_scopes=tuple(d["scope_level"] for d in decisions))
    messages = build_memory_llm_messages(source, (), (), (), 2000, .85, scope_authorization=authorization)
    constraints = json.loads(messages[1]["content"])["proposal_constraints"]
    assert constraints["scope_levels"] == ["conversation", scope]
    assert constraints["scope_authorizations"] == [dict(scope_level=scope, evidence=decisions[0]["evidence"])]
    items = json.loads(proposal(scope))["memories"]
    items.append(dict(kind="major", value="工业设计", evidence="我的专业是工业设计", confidence=.99,
                      operation="ADD", attributed_to="user", scope_level="conversation"))
    parsed = parse_llm_proposals(json.dumps(dict(memories=items), ensure_ascii=False),
                                 source_message=source, scope_authorization=authorization)
    assert {item.memory.memory_key: item.scope_level for item in parsed} == {"user_name": scope, "user_major": "conversation"}
    completion.complete.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("target_scope,source", [
    ("user_character", "我叫海岚；授权今后在与林栖的每段谈话里沿用这个名字。"),
    ("user_global", "我叫海岚；授权今后在每个人物的对话里沿用这个名字。"),
])
async def test_proposed_scope_is_reviewed_without_keyword_nomination(tmp_path, target_scope, source):
    seen = []

    async def complete(messages):
        payload = json.loads(messages[-1]["content"])
        seen.append(payload)
        assert payload["current_user_message"] == source
        if len(seen) == 1:
            assert payload["proposal_constraints"]["scope_levels"] == ["conversation", "user_character", "user_global"]
            return proposal(target_scope)
        assert payload["candidate_scopes"] == [target_scope]
        return response([decision(target_scope, True, source)])

    database = SQLiteDB(tmp_path / "semantic-scope.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("web", "semantic-scope", "synthetic-owner", "first", "group")
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "fixture"),
        completion=SimpleNamespace(complete=complete, close=AsyncMock()))
    try:
        receipt = await worker.schedule_and_wait(repository=repository, character_id="role", user_scope=scope,
            message=source, rule_hints=(), source_message_id="source", immediate=True)
        assert receipt["status"] == "saved" and receipt["persisted"] == 1 and len(seen) == 2
        other_scope = UserScope("web", "semantic-scope", "synthetic-owner", "second", "group")
        records = await repository.list_memory_records("role", other_scope)
        assert len(records) == 1 and records[0]["scope_level"] == target_scope
        assert "海岚" in records[0]["content"]
    finally:
        await worker.shutdown(timeout=1)
        database.close_connection()
