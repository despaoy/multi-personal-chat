"""Stored zero importance survives an authorized erasure proposal."""
import json

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import MemoryResponseError, parse_llm_proposals
from character.models import MemoryItem, UserScope
from db.database import SQLiteDB


@pytest.mark.parametrize("importance",[0.0,0.5,1.0])
async def test_erasure_preserves_stored_fields(tmp_path,importance):
    from legacy_memory_fixtures import seed_legacy_memory

    repo=DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path/'memory.sqlite'))
    scope=UserScope('web','fixture','fiction','fiction','private')
    content='用户说自己来自或居住在杭州'
    seed_legacy_memory(repo,'role',scope,MemoryItem('0','user_fact',content,importance),memory_key='user_location',source_message_id='source')
    records=await repo.list_memory_records('role',scope)
    assert len(records)==1 and records[0]['importance']==importance
    source='请把你记住的我的住址彻底删掉。'
    payload=dict(operation='ERASE',kind='location',value='用户要求彻底删除住址信息',evidence=source,confidence=1.0,attributed_to='user',target_memory_id=str(records[0]['id']),target_memory_key='user_location')
    result=parse_llm_proposals(json.dumps(dict(memories=[payload]),ensure_ascii=False),source_message=source,existing_memories=records)
    assert len(result)==1 and result[0].operation=='ERASE'
    assert result[0].memory.importance==importance
    assert result[0].memory.memory_type=='user_fact'
    assert result[0].memory.content==content
    records[0]["importance"]=None
    with pytest.raises(TypeError):
        parse_llm_proposals(json.dumps(dict(memories=[payload]),ensure_ascii=False),source_message=source,existing_memories=records)


def target_record(identifier, content="用户的猫叫年糕", **changes):
    return dict(id=str(identifier), memory_key="fact_pet", memory_type="user_fact",
                content=content, **{"status": "active", **changes})


def target_response(**changes):
    return {**dict(operation="MERGE", kind="other_user_fact", value="年糕", content="用户的猫叫年糕，现在三岁",
                evidence="我家的猫叫年糕，它现在三岁。", confidence=.99, attributed_to="user",
                target_memory_id="8", target_memory_key="fact_pet"), **changes}


def parse_target(item, records):
    return parse_llm_proposals(json.dumps({"memories": [item]}, ensure_ascii=False),
        source_message=item["evidence"], existing_memories=tuple(records))


@pytest.mark.parametrize("operation", ["MERGE", "SUPERSEDE", "COEXIST", "RETRACT", "ERASE"])
def test_key_only_cannot_guess_a_target_for_any_mutation(operation):
    source = "请把你记住的我的宠物信息彻底删掉。" if operation == "ERASE" else "我刚才说错了，我家的猫叫年糕。"
    with pytest.raises(MemoryResponseError) as caught:
        parse_target(target_response(operation=operation, target_memory_id="", evidence=source),
                     [target_record(7, "用户的狗叫豆包"), target_record(8)])
    assert caught.value.field == "target_memory_id"


@pytest.mark.parametrize("change,records,field", [
    ({"target_memory_key": ""}, [target_record(8)], "target_memory_key"),
    ({"target_memory_id": "unknown"}, [target_record(8)], "target_memory_id"),
    ({"target_memory_key": "fact_dog"}, [target_record(8)], "target_memory_key"),
    ({}, [target_record(8, status="superseded")], "target_memory_id"),
    ({}, [target_record(8), target_record(8)], "target_memory_id"),
])
def test_invalid_target_is_a_specific_protocol_failure(change, records, field):
    with pytest.raises(MemoryResponseError) as caught:
        parse_target(target_response(**change), records)
    assert caught.value.field == field


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_id_keeps_same_key_records_unambiguous(reverse):
    records = [target_record(7, "用户的狗叫豆包"), target_record(8)]
    (proposal,) = parse_target(target_response(), records[::-1] if reverse else records)
    assert proposal.target_memory_id == "8" and proposal.target_memory_key == "fact_pet"
    assert "年糕" in proposal.memory.content and "豆包" not in proposal.memory.content


@pytest.mark.parametrize("target_error", ["", "missing_id", "incompatible_kind"])
async def test_actual_worker_rejects_a_bad_target_before_any_fact_write(tmp_path, target_error):
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig

    database = SQLiteDB(tmp_path / "targets.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("test", "target-contract", "fiction", "fiction", "private")
    observed = datetime(2026, 10, 25, tzinfo=timezone.utc) - timedelta(days=1)
    for index, body in enumerate(["我的狗叫豆包。", "我的猫叫年糕。"]):
        await repository.capture_source("role", scope, source_message_id=str(index), body=body, observed_at=observed)
        await repository.append_claim("role", scope, MemoryItem("", "user_fact", body.replace("我", "用户", 1), .8),
            memory_key="fact_pet", relation_type="COEXIST", source_message_id=str(index),
            evidence=(body,), observed_at=observed.isoformat())
    before = await repository.list_memory_records("role", scope, include_inactive=True)
    cat = next(row for row in before if "年糕" in row["content"])
    source = "我的专业是材料工程。我家的猫叫年糕，它现在三岁。"
    new = dict(operation="ADD", kind="major", value="材料工程", content="", evidence="我的专业是材料工程。",
               confidence=.99, attributed_to="user")
    selected = target_response(target_memory_id="" if target_error == "missing_id" else str(cat["id"]))
    if target_error == "incompatible_kind":
        selected.update(kind="major", value="材料工程", content="用户的专业是材料工程", evidence=new["evidence"])
    completion = SimpleNamespace(complete=AsyncMock(return_value=json.dumps({"memories": [new, selected]}, ensure_ascii=False)), close=AsyncMock())
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "fixture"), completion=completion,
        embedding_provider=SimpleNamespace(embed_texts=lambda texts: [[1., 0.] for _ in texts]))
    try:
        result = await worker.schedule_and_wait(repository=repository, character_id="role", user_scope=scope,
            message=source, rule_hints=(), source_message_id="new", feedback_target_ids=tuple(str(row["id"]) for row in before))
        rows = await repository.list_memory_records("role", scope, include_inactive=True)
        assert result["source_capture"] == "recorded"
        if target_error:
            assert result["status"] == "failed" and result["stage"] == "proposal_validation"
            expected_field = "target_memory_id" if target_error == "missing_id" else "kind"
            assert result["field"] == expected_field and result["accepted"] == result["persisted"] == 0
            assert result["reason"] == "invalid_model_response" and not result["retryable"]
            assert rows == before
        else:
            assert result["status"] == "saved" and result["persisted"] == 2
            merged = next(row for row in rows if row["source_message_id"] == "new" and row["memory_key"] == "fact_pet")
            assert merged["parent_memory_id"] == cat["id"] and merged["supersedes_memory_id"] == cat["id"]
            assert "三岁" in merged["content"] and "豆包" not in merged["content"]
            assert next(row for row in rows if row["id"] == cat["id"])["status"] == "superseded"
            assert next(row for row in rows if "豆包" in row["content"])["status"] == "active"
        assert next(row for row in await repository.list_sources("role", scope) if row["source_message_id"] == "new")["body"] == source
        completion.complete.assert_awaited_once()
    finally:
        await worker.shutdown(timeout=1)
        database.close_connection()


@pytest.mark.parametrize("kind,value,source,right_key,wrong_key", [
    ("major", "考古学", "我的专业是考古学。", "user_major", "user_name"),
    ("like", "红茶", "我喜欢红茶。", "preference_咖啡", "user_major"),
    ("residence", "丽水", "我目前住在丽水。", "user_residence", "user_origin"),
    ("origin", "宣城", "我的老家在宣城。", "user_origin", "user_residence"),
])
@pytest.mark.parametrize("wrong", [False, True])
def test_target_predicate_must_match_the_declared_fact(kind, value, source, right_key, wrong_key, wrong):
    key = wrong_key if wrong else right_key
    record = dict(id="101", memory_key=key, memory_type="user_fact", content="用户此前保存的记录", status="active")
    item = dict(kind=kind, value=value, evidence=source, content="", operation="SUPERSEDE",
                confidence=.99, attributed_to="user", target_memory_id="101", target_memory_key=key)
    if wrong:
        with pytest.raises(MemoryResponseError) as caught:
            parse_target(item, [record])
        assert caught.value.field == "kind"
    else:
        proposal, = parse_target(item, [record])
        assert proposal.operation == "SUPERSEDE" and proposal.target_memory_id == "101"
        assert proposal.target_memory_key == proposal.memory.memory_key == right_key
        assert value in proposal.memory.content and proposal.evidence == source


def withdrawal_case(operation="RETRACT", model_kind=None, metadata=None):
    target = dict(
        id="7",
        memory_key="preference_纸艺",
        memory_type="user_fact",
        content="用户讨厌纸艺",
        status="active",
        importance=0.5,
        scope_level="conversation",
        metadata={"fact_kind": "dislike"} if metadata is None else metadata,
    )
    source = (
        "我刚才说错了，我撤回之前那条关于纸艺的偏好；其他记忆保持不变。"
        if operation == "RETRACT"
        else "请从记忆中彻底删除之前那条纸艺偏好；其他记忆保持不变。"
    )
    raw = dict(
        operation=operation,
        target_memory_id="7",
        target_memory_key="preference_纸艺",
        evidence=source,
        confidence=0.99,
        attributed_to="user",
    )
    if model_kind is not None:
        raw["kind"] = model_kind
    return target, source, raw


@pytest.mark.parametrize("operation", ["RETRACT", "ERASE"])
@pytest.mark.parametrize("model_kind", [None, "like"])
def test_target_operation_preserves_stored_kind_without_reading_display(operation, model_kind):
    target, source, raw = withdrawal_case(operation, model_kind)
    result = parse_llm_proposals(
        json.dumps({"memories": [raw]}, ensure_ascii=False),
        source_message=source,
        existing_memories=(target,),
    )
    assert len(result) == 1
    proposal = result[0]
    assert proposal.operation == operation and proposal.target_memory_id == "7"
    assert proposal.fact_kind == "dislike"
    assert proposal.memory.content == target["content"]


@pytest.mark.parametrize("operation", ["RETRACT", "ERASE"])
def test_target_operation_keeps_legacy_unknown_type_unknown(operation):
    target, source, raw = withdrawal_case(operation, "like", metadata={})
    (proposal,) = parse_llm_proposals(
        json.dumps({"memories": [raw]}, ensure_ascii=False),
        source_message=source,
        existing_memories=(target,),
    )
    assert proposal.fact_kind == ""
    assert proposal.memory.memory_type == target["memory_type"]
    assert proposal.memory.content == target["content"]


@pytest.mark.asyncio
async def test_retraction_persists_authoritative_type_and_preserves_other_fact(
    tmp_path,
):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
    from db.database import SQLiteDB

    database = SQLiteDB(tmp_path / "typed-withdrawal.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("web", "typed-withdrawal", "owner", "owner", "private")
    observed = datetime(2026, 11, 6, tzinfo=timezone.utc)
    target, source, raw = withdrawal_case()
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unit-only", "unit-only"), completion=None)
    try:
        await repository.capture_source(
            "role",
            scope,
            source_message_id="old-dislike",
            body="我讨厌纸艺。",
            observed_at=observed,
        )
        await repository.append_claim(
            "role",
            scope,
            MemoryItem("", "user_fact", target["content"], 0.5),
            memory_key=target["memory_key"],
            evidence=("我讨厌纸艺。",),
            source_message_id="old-dislike",
            observed_at=observed.isoformat(),
            metadata={"fact_kind": "dislike"},
        )
        await repository.append_claim(
            "role",
            scope,
            MemoryItem("", "user_fact", "用户喜欢陶笛", 0.6),
            memory_key="preference_陶笛",
            evidence=("我喜欢陶笛。",),
            metadata={"fact_kind": "like"},
            observed_at=observed.isoformat(),
        )
        before = await repository.list_memory_records("role", scope, limit=None, include_inactive=True)
        old = next(r for r in before if r["memory_key"] == target["memory_key"])
        other = next(r for r in before if r["memory_key"] == "preference_陶笛")
        raw["target_memory_id"] = str(old["id"])
        (proposal,) = parse_llm_proposals(
            json.dumps({"memories": [raw]}, ensure_ascii=False),
            source_message=source,
            existing_memories=tuple(before),
        )
        await repository.capture_source(
            "role",
            scope,
            source_message_id="withdrawal",
            body=source,
            observed_at=observed,
        )
        job = SimpleNamespace(
            repository=repository,
            character_id="role",
            user_scope=scope,
            message=source,
            source_message_id="withdrawal",
            observed_at=observed,
            write_mode="hot",
            feedback_target_ids=(),
        )
        assert await worker._persist_proposal(job, proposal) == "saved"
        after = await repository.list_memory_records("role", scope, limit=None, include_inactive=True)
        withdrawn = next(r for r in after if r["relation_type"] == "RETRACT")
        assert withdrawn["metadata"]["fact_kind"] == "dislike"
        assert next(r for r in after if r["id"] == old["id"])["status"] == "retracted"
        assert next(r for r in after if r["id"] == other["id"]) == other
        sources = await repository.list_sources("role", scope)
        assert any(r["source_message_id"] == "withdrawal" and r["body"] == source for r in sources)
    finally:
        database.close_connection()
