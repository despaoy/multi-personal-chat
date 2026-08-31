"""Focused contracts for semantic memory writes and lifecycle scheduling."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from character.memory_llm import (
    MemoryEnrichmentScheduler,
    MemoryLlmConfig,
    build_memory_llm_messages,
    parse_llm_proposals,
)
from character.models import CompiledCharacterContext, MemoryItem, RelationshipState, UserScope
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import CharacterContextService, PreparedCharacterTurn, TurnInput


def _response(**memory) -> str:
    return json.dumps({"memories": [memory]}, ensure_ascii=False)


def _record(
    memory_id: int = 7,
    *,
    key: str = "preference_咖啡",
    content: str = "用户说喜欢咖啡",
) -> dict:
    return {
        "id": memory_id,
        "memory_key": key,
        "memory_type": "user_fact",
        "content": content,
        "status": "active",
    }


def _scope() -> UserScope:
    return UserScope(
        platform="qq",
        adapter="onebot",
        sender_id="user-1",
        conversation_id="user-1",
        conversation_type="private",
    )


def test_conditional_preference_is_kept_as_coexisting_claim():
    source = "不是讨厌咖啡，只是不喜欢太苦的咖啡。"
    result = parse_llm_proposals(
        _response(
            kind="dislike",
            value="咖啡",
            content="用户只在咖啡太苦时不喜欢咖啡",
            evidence=source,
            confidence=0.96,
            operation="COEXIST",
            target_memory_id="7",
            target_memory_key="preference_咖啡",
            attributed_to="user",
            qualifiers={"condition": "太苦"},
        ),
        source_message=source,
        existing_memories=(_record(),),
    )

    assert len(result) == 1
    assert result[0].operation == "COEXIST"
    assert result[0].target_memory_id == "7"
    assert dict(result[0].qualifiers) == {"condition": "太苦"}
    assert result[0].memory is not None
    assert "太苦" in result[0].memory.content


def test_uncertain_future_fact_is_pending_instead_of_becoming_active():
    source = "我可能明年换工作"
    result = parse_llm_proposals(
        _response(
            kind="goal",
            value="明年换工作",
            content="用户可能明年换工作",
            evidence=source,
            confidence=0.92,
            operation="PENDING",
            attributed_to="user",
            qualifiers={"certainty": "可能"},
            valid_from="2027-01-01",
        ),
        source_message=source,
    )

    assert len(result) == 1
    assert result[0].operation == "PENDING"
    assert result[0].valid_from.startswith("2027-01-01")
    assert result[0].memory is not None
    assert result[0].memory.content.startswith("待确认：")


def test_pending_uses_relation_confidence_and_reordered_value_stays_grounded():
    pending_source = "我好像更喜欢喝茶，不过还不确定。"
    pending = parse_llm_proposals(
        _response(
            kind="like",
            value="茶",
            content="用户喜欢茶",
            evidence=pending_source,
            confidence=0.70,
            operation="PENDING",
            attributed_to="user",
            qualifiers={"certainty": "不确定"},
        ),
        source_message=pending_source,
        confidence_threshold=0.85,
    )
    reordered_source = "请记住，以后给我推荐饮料时要避开含咖啡因的。"
    reordered = parse_llm_proposals(
        _response(
            kind="dislike",
            value="含咖啡因的饮料",
            content="用户不喜欢含咖啡因的饮料",
            evidence=reordered_source,
            confidence=0.95,
            operation="ADD",
            attributed_to="user",
        ),
        source_message=reordered_source,
    )

    assert [item.operation for item in pending] == ["PENDING"]
    assert [item.operation for item in reordered] == ["ADD"]


def test_explicit_erasure_phrase_and_duplicate_merge_are_normalized_safely():
    erase_source = "请把你记住的我的住址彻底删掉。"
    erased = parse_llm_proposals(
        _response(
            kind="erasure",
            value="用户要求彻底删除住址信息",
            content="用户要求彻底删除住址信息",
            evidence=erase_source,
            confidence=1.0,
            operation="ERASE",
            target_memory_id="7",
            target_memory_key="user_location",
            attributed_to="user",
        ),
        source_message=erase_source,
        existing_memories=(_record(key="user_location", content="用户说自己来自或居住在杭州"),),
    )
    duplicate_source = "我还是在准备保研。"
    duplicate = parse_llm_proposals(
        _response(
            kind="goal",
            value="准备保研",
            content="用户正在进行或准备：保研",
            evidence=duplicate_source,
            confidence=0.95,
            operation="MERGE",
            target_memory_id="7",
            target_memory_key="goal_保研",
            attributed_to="user",
        ),
        source_message=duplicate_source,
        existing_memories=(_record(key="goal_保研", content="用户正在进行或准备：保研"),),
    )

    assert [item.operation for item in erased] == ["ERASE"]
    assert [item.operation for item in duplicate] == ["NOOP"]


def test_model_relation_format_mistakes_are_normalized_with_evidence_and_whitelist():
    conditional_source = "不是完全讨厌咖啡，只是不喜欢太苦的。"
    conditional = parse_llm_proposals(
        _response(
            kind="like",
            value="不喜欢太苦的咖啡",
            content="用户不喜欢太苦的咖啡",
            evidence=conditional_source,
            confidence=0.95,
            operation="MERGE",
            target_memory_id="7",
            target_memory_key="preference_咖啡",
            attributed_to="user",
        ),
        source_message=conditional_source,
        existing_memories=(_record(),),
    )
    contrast_source = "准确地说，我喝无咖啡因咖啡，普通咖啡才不喝。"
    contrast = parse_llm_proposals(
        _response(
            kind="dislike",
            value="含咖啡因的饮料",
            content="用户不喜欢含咖啡因的饮料",
            evidence=contrast_source,
            confidence=0.95,
            operation="MERGE",
            target_memory_id="7",
            target_memory_key="preference_咖啡",
            attributed_to="user",
        ),
        source_message=contrast_source,
        existing_memories=(_record(content="用户说不喜欢咖啡"),),
    )

    assert [item.operation for item in conditional] == ["COEXIST"]
    assert [item.operation for item in contrast] == ["COEXIST"]
    assert contrast[0].memory is not None
    assert "普通咖啡" in contrast[0].memory.content
    assert "含咖啡因的饮料" not in contrast[0].memory.content


def test_ellipsis_history_is_context_not_evidence_and_add_can_link_same_slot():
    ellipsis_source = "还是上次那个方向。"
    ellipsis = parse_llm_proposals(
        _response(
            kind="other_user_fact",
            value="点云补全",
            content="用户开头的第三人称事实",
            evidence="我最近一直在做点云补全。",
            confidence=1.0,
            operation="ADD",
            attributed_to="user",
        ),
        source_message=ellipsis_source,
        history=({"role": "user", "content": "我最近一直在做点云补全。"},),
    )
    pet_source = "我又养了一只叫团子的猫。"
    pet = parse_llm_proposals(
        _response(
            kind="other_user_fact",
            value="团子",
            content="用户养了一只叫团子的猫",
            evidence=pet_source,
            confidence=1.0,
            operation="ADD",
            attributed_to="user",
        ),
        source_message=pet_source,
        existing_memories=(_record(key="fact_养猫", content="用户养了一只叫年糕的猫"),),
    )

    assert [item.operation for item in ellipsis] == ["ADD"]
    assert ellipsis[0].evidence == ellipsis_source
    assert [item.operation for item in pet] == ["MERGE"]
    assert pet[0].target_memory_id == "7"


def test_temporary_location_links_stable_location_and_moves_time_fields():
    source = "这周我在北京出差"
    result = parse_llm_proposals(
        _response(
            kind="location",
            value="北京",
            content="用户这周在北京出差",
            evidence=source,
            confidence=0.95,
            operation="ADD",
            attributed_to="user",
            qualifiers={
                "valid_from": "2026-08-24T00:00:00+00:00",
                "valid_to": "2026-08-31T00:00:00+00:00",
            },
        ),
        source_message="这周我在北京出差，下周就回杭州。",
        existing_memories=(_record(key="user_location", content="用户说自己来自或居住在杭州"),),
    )

    assert [item.operation for item in result] == ["COEXIST"]
    assert result[0].target_memory_id == "7"
    assert result[0].valid_from.startswith("2026-08-24")
    assert result[0].valid_to.startswith("2026-08-31")


def test_deictic_correction_must_target_a_memory_used_in_last_reply():
    source = "刚才那条我说错了，改成做点云识别"
    existing = (
        _record(memory_id=7, key="goal_代码补全", content="用户正在进行或准备：代码补全"),
        _record(memory_id=8, key="goal_论文阅读", content="用户正在进行或准备：论文阅读"),
    )
    response = _response(
        kind="goal",
        value="做点云识别",
        content="用户正在进行或准备：做点云识别",
        evidence=source,
        confidence=0.98,
        operation="SUPERSEDE",
        target_memory_id="7",
        target_memory_key="goal_代码补全",
        attributed_to="user",
    )

    accepted = parse_llm_proposals(
        response,
        source_message=source,
        existing_memories=existing,
        feedback_target_ids=("7",),
    )
    rejected = parse_llm_proposals(
        response,
        source_message=source,
        existing_memories=existing,
        feedback_target_ids=("8",),
    )

    assert [item.operation for item in accepted] == ["SUPERSEDE"]
    assert rejected == []


def test_invalid_subject_time_and_external_source_are_rejected():
    third_party = _response(
        kind="like",
        value="咖啡",
        content="用户喜欢咖啡",
        evidence="我室友喜欢咖啡",
        confidence=0.99,
        operation="ADD",
        attributed_to="user",
    )
    bad_time = _response(
        kind="goal",
        value="保研",
        content="用户正在准备保研",
        evidence="我正在准备保研",
        confidence=0.99,
        operation="ADD",
        attributed_to="user",
        valid_from="2028-01-01",
        valid_to="2027-01-01",
    )

    assert parse_llm_proposals(third_party, source_message="我室友喜欢咖啡") == []
    assert parse_llm_proposals(bad_time, source_message="我正在准备保研") == []
    assert (
        parse_llm_proposals(
            _response(
                kind="like",
                value="咖啡",
                content="用户喜欢咖啡",
                evidence="我喜欢咖啡",
                confidence=0.99,
                operation="ADD",
                attributed_to="user",
            ),
            source_message="我喜欢咖啡",
            source_type="rag",
        )
        == []
    )


def test_prompt_exposes_feedback_targets_but_filters_tool_and_system_history():
    messages = build_memory_llm_messages(
        "刚才那条说错了",
        (),
        (
            {"role": "system", "content": "系统里说用户喜欢红茶"},
            {"role": "tool", "content": "RAG 说用户住在北京"},
            {"role": "assistant", "content": "你是指咖啡吗"},
            {"role": "user", "content": "对，是咖啡"},
        ),
        (_record(),),
        2000,
        0.85,
        feedback_target_ids=("7", "999"),
        write_mode="hot",
    )
    payload = json.loads(messages[1]["content"])

    assert payload["feedback_target_ids"] == ["7"]
    assert payload["existing_memories"][0]["was_injected_in_last_reply"] is True
    assert [item["role"] for item in payload["recent_history"]] == ["assistant", "user"]
    assert all(item["eligible_as_memory_evidence"] is False for item in payload["recent_history"])


class _Completion:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[dict[str, str]]] = []
        self.closed = False

    async def complete(self, messages):
        self.calls.append(messages)
        return self.response

    async def close(self):
        self.closed = True


class _ClaimRepository:
    def __init__(self, records: list[dict] | None = None) -> None:
        self.records = records or []
        self.claims: list[tuple[tuple, dict]] = []
        self.erases: list[dict] = []

    async def list_memory_records(self, character_id, user_scope, limit=30):
        return self.records[:limit]

    async def append_claim(self, *args, **kwargs):
        self.claims.append((args, kwargs))
        return {"id": 101, "persisted": True}

    async def erase_memory(self, character_id, user_scope, **kwargs):
        self.erases.append(kwargs)
        return 1


@pytest.mark.asyncio
async def test_idle_write_waits_for_flush_and_uses_append_claim_contract():
    source = "我可能明年换工作"
    completion = _Completion(
        _response(
            kind="goal",
            value="明年换工作",
            content="用户可能明年换工作",
            evidence=source,
            confidence=0.95,
            operation="PENDING",
            attributed_to="user",
            qualifiers={"certainty": "可能"},
        )
    )
    repository = _ClaimRepository()
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(
            enabled=True,
            base_url="http://127.0.0.1:8001",
            model="test",
            idle_seconds=360,
            batch_size=4,
        ),
        completion=completion,
    )

    assert scheduler.schedule(
        repository=repository,
        character_id="kisaki",
        user_scope=_scope(),
        message=source,
        rule_hints=[],
        source_message_id="msg-1",
    )
    assert scheduler.status.buffered == 1
    assert completion.calls == []

    assert await scheduler.flush_memory(timeout=1.0)
    assert len(repository.claims) == 1
    kwargs = repository.claims[0][1]
    assert kwargs["relation_type"] == "PENDING"
    assert kwargs["status"] == "pending"
    assert kwargs["evidence"] == (source,)
    assert kwargs["source_message_ids"] == ("msg-1",)
    assert scheduler.status.saved == 1
    await scheduler.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_merge_promotes_one_canonical_claim_without_losing_old_pet(tmp_path):
    """MERGE 必须先聚合旧内容，再 supersede 旧版本。"""

    repository = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "merge-memory.db"))
    scope = _scope()
    old = await repository.append_claim(
        "kisaki",
        scope,
        MemoryItem("", "user_fact", "用户养猫咪小白", 0.7),
        memory_key="fact_宠物",
    )
    source = "我又养了一只团子猫"
    completion = _Completion(
        _response(
            kind="other_user_fact",
            value="团子猫",
            content="用户又养了一只团子猫",
            evidence=source,
            confidence=0.97,
            operation="MERGE",
            target_memory_id=str(old["id"]),
            target_memory_key="fact_宠物",
            attributed_to="user",
        )
    )
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(
            enabled=True,
            base_url="http://127.0.0.1",
            model="test",
            idle_seconds=360,
        ),
        completion=completion,
    )

    assert scheduler.schedule(
        repository=repository,
        character_id="kisaki",
        user_scope=scope,
        message=source,
        rule_hints=[],
        source_message_id="msg-pet-2",
    )
    assert await scheduler.flush_memory(timeout=2.0)

    active = await repository.list_memory_records("kisaki", scope, limit=10)
    lineage = await repository.list_memory_records("kisaki", scope, limit=10, include_inactive=True)
    assert len(active) == 1
    assert "小白" in active[0]["content"]
    assert "团子" in active[0]["content"]
    assert active[0]["relation_type"] == "MERGE"
    assert active[0]["parent_memory_id"] == old["id"]
    assert active[0]["supersedes_memory_id"] == old["id"]
    assert {item["status"] for item in lineage} == {"active", "superseded"}
    await scheduler.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_explicit_erase_uses_hot_path_and_physical_delete():
    source = "请忘掉刚才那条记忆"
    completion = _Completion(
        _response(
            kind="goal",
            value="",
            content="",
            evidence=source,
            confidence=0.99,
            operation="ERASE",
            target_memory_id="7",
            target_memory_key="goal_代码补全",
            attributed_to="user",
        )
    )
    repository = _ClaimRepository([_record(memory_id=7, key="goal_代码补全", content="用户正在进行或准备：代码补全")])
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(enabled=True, base_url="http://127.0.0.1", model="test"),
        completion=completion,
    )

    assert scheduler.schedule(
        repository=repository,
        character_id="kisaki",
        user_scope=_scope(),
        message=source,
        rule_hints=[],
        feedback_target_ids=("7",),
    )
    assert scheduler.status.last_outcome == "queued_hot"
    assert await scheduler.flush_memory(timeout=1.0)
    assert repository.erases == [{"memory_id": 7, "memory_key": "goal_代码补全", "scope_level": "conversation"}]
    assert scheduler.status.erased == 1
    await scheduler.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_complete_turn_passes_only_actual_injected_memory_ids(monkeypatch):
    class _Scheduler:
        enabled = True
        status = SimpleNamespace(last_outcome="idle")

        def __init__(self):
            self.kwargs = None

        def schedule(self, **kwargs):
            self.kwargs = kwargs
            return True

    class _Repository:
        async def increment_interaction(self, character_id, user_scope):
            return 1

    scheduler = _Scheduler()
    monkeypatch.setattr("character.memory_llm.get_memory_enrichment_scheduler", lambda: scheduler)
    service = CharacterContextService(object(), _Repository(), object())
    prepared = PreparedCharacterTurn(
        character_id="kisaki",
        user_scope=_scope(),
        compiled=CompiledCharacterContext("", "", "", ("7", "11")),
        history=({"role": "user", "content": "之前的用户消息"},),
        relationship=RelationshipState(),
        memory_candidates=2,
        interaction_count=0,
        reply_guard=None,  # type: ignore[arg-type]
    )
    turn = TurnInput(
        message="刚才那条我说错了，改成做点云识别",
        platform="qq",
        adapter="onebot",
        sender_id="user-1",
        conversation_id="user-1",
        conversation_type="private",
    )

    outcome = await service.complete_turn(prepared, turn, "assistant reply", source_message_id="msg-2")

    assert outcome.memory_enrichment_mode == "hot"
    assert outcome.memory_enrichment_status == "queued_hot"
    assert scheduler.kwargs["feedback_target_ids"] == ("7", "11")
    assert "assistant reply" not in json.dumps(scheduler.kwargs, ensure_ascii=False, default=str)
