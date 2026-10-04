"""New synthetic component control; normal rule writing and scoped recall.

Main provider is a spy. No API/auth/vector or cloud success is claimed.
"""

from datetime import datetime, timezone
from html import unescape

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import PreparedCharacterTurn

from character.context_builder import compile_reference_context
from character.memory_extractor import extract_memories
from character.memory_service import CharacterMemoryService
from character.models import CompiledCharacterContext, RelationshipState, UserScope
from character.rule_memory_writer import write_rule_memory
from db.database import SQLiteDB
from inference.generation_request import GenerationRequest, generate_character_response
from inference.private_context_authority import make_private_context_revalidator

BODY = "我喜欢乌龙茶。"
QUERY = "我保存的乌龙茶偏好是什么？只根据当前可核验的本人记录回答，不确认现实审核结果，缺少记录就保持未知。"


async def exercise(tmp_path, *, erase):
    db = SQLiteDB(tmp_path / "structured-authority.sqlite")
    repo = DatabaseCharacterMemoryRepository(db)
    scope = UserScope("web", "structured-authority", "fiction-owner", "fiction-owner", "private")
    now = datetime.now(timezone.utc)
    assert (
        await repo.capture_source(
            "fiction-role", scope, source_message_id="fiction-structured-original", body=BODY, observed_at=now
        )
        == "recorded"
    )
    extracted = extract_memories(BODY, reference_time=now)
    assert len(extracted) == 1 and extracted[0].evidence
    assert await write_rule_memory(
        repo, "fiction-role", scope, extracted[0], "fiction-structured-original", observed_at=now
    )
    service = CharacterMemoryService(repo, semantic_enabled=False)
    memories, count = await service.load_relevant_memories("fiction-role", scope, QUERY, reference_time=now)
    assert count == 1 and len(memories) == 1 and memories[0].evidence and memories[0].source_message_ids
    reference, ids = compile_reference_context(memories, complete_evidence=True, max_chars=None)
    assert BODY.rstrip("。") in reference and ids == (memories[0].memory_id,)
    context = CompiledCharacterContext(
        "虚构组件角色画像", "", reference, memory_status="available", used_memory_ids=ids, memory_packets=memories
    )
    assert not context.episodic_reference_context and not context.source_candidate_context
    prepared = PreparedCharacterTurn(
        character_id="fiction-role",
        user_scope=scope,
        compiled=context,
        history=(),
        relationship=RelationshipState(),
        memory_candidates=count,
        interaction_count=0,
        reply_guard=None,
    )
    request = GenerationRequest(
        message=QUERY,
        persona_prompt="虚构组件角色规则",
        character_context=context,
        context_window_tokens=65536,
        private_context_revalidator=make_private_context_revalidator(prepared, db),
    )
    assert callable(request.private_context_revalidator)
    if erase:
        assert await repo.erase_memory("fiction-role", scope, memory_id=int(memories[0].memory_id)) == 1
        assert await repo.get_memory_record(int(memories[0].memory_id), "fiction-role", scope) is None
    seen = []

    async def model(**kwargs):
        seen.append(unescape("\n".join(message["content"] for message in kwargs["messages"])))
        return "仅依据当前可核验的记录回答。"

    result = await generate_character_response(request, model)
    assert len(seen) == 1 and result.reply and QUERY in seen[0]
    assert (BODY.rstrip("。") in seen[0]) is (not erase)
    assert (memories[0].content in seen[0]) is (not erase)
    assert bool(result.plan.character_context.used_memory_ids) is (not erase)


@pytest.mark.asyncio
async def test_unchanged_structured_memory_keeps_normal_scoped_evidence(tmp_path):
    await exercise(tmp_path, erase=False)


@pytest.mark.asyncio
async def test_erased_structured_memory_without_speech_packet_is_absent_from_final_input(tmp_path):
    await exercise(tmp_path, erase=True)


async def structured_request(tmp_path, *, projected=False, historical=False):
    import json
    from datetime import timedelta

    from inference.generation_request import RetrievalResult

    db = SQLiteDB(tmp_path / "more-authority.sqlite")
    repo = DatabaseCharacterMemoryRepository(db)
    scope = UserScope("web", "structured-authority", "fiction-owner", "fiction-owner", "private")
    now = datetime.now(timezone.utc) - timedelta(days=2)
    for index, body in enumerate((BODY, "我喜欢豆浆。")):
        assert (
            await repo.capture_source(
                "fiction-role", scope, source_message_id=f"fiction-structured-{index}", body=body, observed_at=now
            )
            == "recorded"
        )
        items = extract_memories(body, reference_time=now)
        assert len(items) == 1
        assert await write_rule_memory(
            repo, "fiction-role", scope, items[0], f"fiction-structured-{index}", observed_at=now
        )
    original = next(
        row for row in await repo.list_memory_records("fiction-role", scope, limit=None) if "乌龙茶" in row["content"]
    )
    if projected:
        metadata = dict(
            original["metadata"],
            temporal_provenance=dict(version=1, producer="semantic_memory", validity_authority="unverified"),
        )
        connection = db._get_connection()
        connection.execute(
            "UPDATE character_memories SET content=?,metadata_json=? WHERE id=?",
            ("语义提供者的展示摘要，非原始断言", json.dumps(metadata), original["id"]),
        )
        connection.commit()
    if historical:
        body = "我不喜欢乌龙茶。"
        stamp = now + timedelta(days=1)
        assert (
            await repo.capture_source(
                "fiction-role", scope, source_message_id="fiction-correction", body=body, observed_at=stamp
            )
            == "recorded"
        )
        items = extract_memories(body, reference_time=stamp)
        assert len(items) == 1 and await write_rule_memory(
            repo, "fiction-role", scope, items[0], "fiction-correction", observed_at=stamp
        )
    query = (
        "我过去保存的乌龙茶和豆浆偏好分别是什么？只依据本人历史记录，不证明现实审核完成。"
        if historical
        else "我保存的乌龙茶和豆浆偏好分别是什么？只依据当前可核验的本人记录，不证明现实审核完成。"
    )
    service = CharacterMemoryService(repo, semantic_enabled=False)
    memories, count = await service.load_relevant_memories("fiction-role", scope, query)
    assert count >= 2 and all(item.storage_versions for item in memories)
    target = next(item for item in memories if item.memory_id == str(original["id"]))
    assert target.historical is historical
    if projected:
        assert target.content != (await repo.get_memory_record(original["id"], "fiction-role", scope))["content"]
        assert target.source_versions
    reference, ids = compile_reference_context(memories, complete_evidence=True, max_chars=None)
    context = CompiledCharacterContext(
        "虚构角色画像", "", reference, memory_status="available", used_memory_ids=ids, memory_packets=memories
    )
    prepared = PreparedCharacterTurn(
        character_id="fiction-role",
        user_scope=scope,
        compiled=context,
        history=(),
        relationship=RelationshipState(),
        memory_candidates=count,
        interaction_count=0,
        reply_guard=None,
    )
    public = "独立虚构公共说明：书页登记收费13元。"
    request = GenerationRequest(
        message=query,
        persona_prompt="虚构角色规则",
        character_context=context,
        context_window_tokens=65536,
        retrieval=RetrievalResult(status="ok", evidence=public),
        private_context_revalidator=make_private_context_revalidator(prepared, db),
    )
    return request, prepared, repo, db, scope, target, public


async def wire_response(request):
    seen = []

    async def model(**kwargs):
        seen.append(unescape("\n".join(message["content"] for message in kwargs["messages"])))
        return "仅依据本轮仍可核验的独立资料回答。"

    result = await generate_character_response(request, model)
    assert len(seen) == 1
    return seen[0], result


@pytest.mark.asyncio
async def test_selected_claim_erasure_keeps_independent_full_claim_and_public_evidence(tmp_path):
    request, _prepared, repo, _db, scope, target, public = await structured_request(tmp_path)
    assert await repo.erase_memory("fiction-role", scope, memory_id=int(target.memory_id)) == 1
    wire, result = await wire_response(request)
    assert target.content not in wire and BODY.rstrip("。") not in wire
    assert "我喜欢豆浆" in wire and public in wire
    assert len(result.plan.character_context.memory_packets) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["status", "metadata_json", "valid_to", "evidence_json"])
async def test_storage_change_invalidates_reviewed_claim_without_substituting_new_data(tmp_path, field):
    import json

    request, _prepared, repo, db, scope, target, public = await structured_request(tmp_path)
    row = await repo.get_memory_record(int(target.memory_id), "fiction-role", scope)
    values = dict(
        status="retracted",
        metadata_json=json.dumps(dict(row["metadata"], qualifiers=dict(condition="新增未经本轮审核的条件"))),
        valid_to="2025-01-01T00:00:00+00:00",
        evidence_json=json.dumps(["新增未经本轮审核的完整替代说明"]),
    )
    connection = db._get_connection()
    assert field in values
    connection.execute(
        "UPDATE character_memories SET " + field + "=? WHERE id=?", (values[field], int(target.memory_id))
    )
    connection.commit()
    wire, _result = await wire_response(request)
    assert target.content not in wire and BODY.rstrip("。") not in wire
    assert "新增未经本轮审核" not in wire and "我喜欢豆浆" in wire and public in wire


@pytest.mark.asyncio
async def test_scoped_read_failure_removes_only_affected_packet(tmp_path, monkeypatch):
    from inference.structured_context_authority import revalidate_private_memories

    request, _prepared, repo, _db, scope, target, public = await structured_request(tmp_path)
    read = repo.get_memory_record

    async def fail_target(memory_id, *args):
        if str(memory_id) == target.memory_id:
            raise RuntimeError("explicit isolated read failure")
        return await read(memory_id, *args)

    monkeypatch.setattr(repo, "get_memory_record", fail_target)
    refreshed = await revalidate_private_memories(request, repo, "fiction-role", scope)
    wire, _result = await wire_response(refreshed)
    assert target.content not in wire and "我喜欢豆浆" in wire and public in wire


@pytest.mark.asyncio
async def test_unchanged_historical_versions_remain_authorized(tmp_path):
    request, _prepared, _repo, _db, _scope, target, public = await structured_request(tmp_path, historical=True)
    wire, result = await wire_response(request)
    assert target.content in wire and BODY.rstrip("。") in wire and public in wire
    assert target.memory_id in result.plan.character_context.used_memory_ids


@pytest.mark.asyncio
async def test_legitimate_semantic_projection_keeps_original_stored_version_binding(tmp_path):
    request, _prepared, _repo, _db, _scope, target, public = await structured_request(tmp_path, projected=True)
    wire, result = await wire_response(request)
    assert target.content in wire and BODY.rstrip("。") in wire and public in wire
    assert target.memory_id in result.plan.character_context.used_memory_ids
    assert "语义提供者的展示摘要" not in wire


@pytest.mark.asyncio
async def test_changed_linked_source_invalidates_projection_even_if_stored_claim_is_unchanged(tmp_path):
    from character.memory_read_authority import record_version

    request, _prepared, repo, db, scope, target, public = await structured_request(tmp_path, projected=True)
    before = await repo.get_memory_record(int(target.memory_id), "fiction-role", scope)
    connection = db._get_connection()
    connection.execute(
        "UPDATE memory_sources SET body=? WHERE source_message_id=?",
        ("完整新的未经本轮审核的原话。", "fiction-structured-0"),
    )
    connection.commit()
    assert record_version(before) == record_version(
        await repo.get_memory_record(int(target.memory_id), "fiction-role", scope)
    )
    wire, _result = await wire_response(request)
    assert target.content not in wire and BODY.rstrip("。") not in wire and "完整新的未经本轮审核" not in wire
    assert "我喜欢豆浆" in wire and public in wire


@pytest.mark.asyncio
async def test_guard_retry_rechecks_structured_memory_after_first_answer(tmp_path):
    from dataclasses import replace

    from character.output_guard import ReplyGuard

    request, _prepared, repo, _db, scope, target, public = await structured_request(tmp_path)
    request = replace(request, reply_guard=ReplyGuard(forbid_laughter=True), reply_guard_mode="strict")
    seen = []

    async def model(**kwargs):
        seen.append(unescape("\n".join(message["content"] for message in kwargs["messages"])))
        if len(seen) == 1:
            assert await repo.erase_memory("fiction-role", scope, memory_id=int(target.memory_id)) == 1
            return "哈哈，已核对。"
        return "仅依据仍可核验的资料回答。"

    result = await generate_character_response(request, model)
    assert result.guard_retried and not result.guard_fallback and len(seen) == 2
    assert target.content in seen[0] and target.content not in seen[1]
    assert BODY.rstrip("。") not in seen[1] and "我喜欢豆浆" in seen[1] and public in seen[1]


@pytest.mark.asyncio
async def test_internal_storage_versions_never_enter_model_or_frozen_completion(tmp_path):
    import json

    from services.delivery_memory import freeze_completion

    from character.evidence_selector import selection_messages

    request, prepared, _repo, _db, _scope, target, _public = await structured_request(tmp_path)
    wire, _result = await wire_response(request)
    selection = json.dumps(selection_messages(request.message, request.character_context.memory_packets))
    frozen = json.dumps(freeze_completion(prepared))
    for text in (wire, selection, frozen):
        assert "storage_versions" not in text and "source_versions" not in text
        assert all(version not in text for _key, version in target.storage_versions)
