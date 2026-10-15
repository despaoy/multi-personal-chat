"""Malformed stored origin scopes fail, valid exclusions stay empty.

Sources, claim links, rule claims, relationships, scoped revision SQL, recall,
context preparation and final authority checks use real SQLite/repositories.
Only the fixed fictional profile is supplied at its normal registry boundary.
All model-dependent features are explicitly disabled and guarded.
"""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.messages import DatabaseMessageRepository
from services.character_context import CharacterContextService, TurnInput

from api.generate import _prepare_character_turn
from character.memory_extractor import extract_memories
from character.memory_service import CharacterMemoryService
from character.models import CharacterProfile, MemoryItem, RelationshipState, UserScope
from character.rule_memory_writer import write_rule_memory
from db.database import SQLiteDB
from db.memory_source import source_identity, source_scope
from db.schemas import MessageRequest
from inference.generation_request import GenerationRequest
from inference.private_context_authority import make_private_context_revalidator

ROLE = "fiction-role"
SCOPE = UserScope("web", "revision-scope", "fiction-owner", "fiction-room-a", "group")
SOURCE_ID = "fiction-origin-source"
BODY = "我喜欢乌龙茶。"
QUERY = "我保存的乌龙茶偏好是什么？只依据本人记录，缺少资料时保留未知。"
RELATION = RelationshipState(stage="familiar", preferred_address="同学", summary="合成组件测试中的既有对话关系。")
PRIVATE_BAD_JSON = "{fiction-private-corrupt-scope"
ORIGIN_FIELDS = [ROLE, SCOPE.platform, SCOPE.adapter, SCOPE.sender_id, SCOPE.conversation_type, SCOPE.conversation_id]
BAD_ORIGINS = [
    pytest.param(PRIVATE_BAD_JSON, id="invalid-json"),
    pytest.param(json.dumps({"not": "a six-field origin"}), id="object-instead-of-fields"),
    pytest.param(json.dumps(ORIGIN_FIELDS[:-1]), id="missing-field"),
    pytest.param(json.dumps([ROLE, SCOPE.platform, 17, SCOPE.sender_id, "group", SCOPE.conversation_id]), id="nontext-field"),
]


async def arrange_revision_scope(root, patch, *, scope_level="conversation"):
    """Importable seed helper shared by tests and the read-only preflight."""
    from inference import review_client

    patch.setenv("CONTEXTUAL_MEMORY_SELECTION_ENABLED", "false")
    patch.setenv("CONTEXTUAL_DECISION_POLICY_ENABLED", "false")
    patch.setenv("DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED", "false")

    async def forbidden_model():
        raise AssertionError("This scope-storage control must never resolve a live reviewer")

    patch.setattr(review_client, "get_context_review_client", forbidden_model)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    database = SQLiteDB(root / "revision-scope.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    stamp = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert await repository.capture_source(
        ROLE, SCOPE, source_message_id=SOURCE_ID, body=BODY, observed_at=stamp,
    ) == "recorded"
    items = extract_memories(BODY, reference_time=stamp)
    assert len(items) == 1 and items[0].evidence == BODY
    if scope_level == "conversation":
        assert await write_rule_memory(repository, ROLE, SCOPE, items[0], SOURCE_ID, observed_at=stamp)
    else:
        await repository.append_claim(
            ROLE, SCOPE, MemoryItem("", "user_fact", items[0].content, .9),
            memory_key=items[0].memory_key, scope_level=scope_level,
            evidence=(BODY,), source_message_id=SOURCE_ID,
            observed_at=stamp.isoformat(), metadata={"origin": "scope_control"},
        )
    rows = await repository.list_memory_records(ROLE, SCOPE, limit=None)
    assert len(rows) == 1 and rows[0]["source_message_ids"] == [SOURCE_ID]
    claim = rows[0]
    await repository.upsert_relationship(ROLE, SCOPE, RELATION)
    profile = CharacterProfile(ROLE, "虚构组件角色", identity="仅用于独立来源权限验证", boundaries=("不编造事实",))
    memory_service = CharacterMemoryService(
        repository, semantic_enabled=False, gate_enabled=False,
        evidence_enabled=True, version_filter_enabled=True,
    )
    context_service = CharacterContextService(
        SimpleNamespace(get_profile=lambda identity: profile if identity == ROLE else None),
        repository, DatabaseMessageRepository(database), memory_service=memory_service,
        source_recall_enabled=False, reference_max_chars=None,
    )
    assert context_service._memory_selector is None and context_service._contextual_policy is None
    turn = TurnInput(
        message=QUERY, platform=SCOPE.platform, adapter=SCOPE.adapter, sender_id=SCOPE.sender_id,
        conversation_id=SCOPE.conversation_id, conversation_type=SCOPE.conversation_type,
        received_at=stamp + timedelta(seconds=1),
    )
    scope = source_scope(ROLE, SCOPE.platform, SCOPE.adapter, SCOPE.sender_id, SCOPE.conversation_type, SCOPE.conversation_id)
    identity = source_identity(scope, SOURCE_ID)
    return SimpleNamespace(
        database=database, repository=repository, memory=memory_service, context=context_service,
        claim=claim, stamp=stamp, turn=turn, identity=identity, pairs=((int(claim["id"]), SOURCE_ID),),
    )


@pytest.fixture
async def revision_rig(tmp_path, monkeypatch):
    rig = await arrange_revision_scope(tmp_path, monkeypatch)
    try:
        yield rig
    finally:
        rig.database.close_connection()


def corrupt_origin(rig, raw):
    connection = rig.database._get_connection()
    cursor = connection.execute(
        "UPDATE memory_sources SET scope_key=? WHERE source_key=? AND state='recorded'",
        (raw, rig.identity["source_key"]),
    )
    assert cursor.rowcount == 1
    connection.commit()


def generation_request(prepared):
    return GenerationRequest(
        message=QUERY, persona_prompt="独立虚构角色组件规则", character_context=prepared.compiled,
        history=prepared.history, context_window_tokens=65536,
    )


@pytest.mark.asyncio
async def test_healthy_real_relationship_claim_and_origin_reach_final_context(revision_rig):
    rig = revision_rig
    revisions = await rig.repository.linked_source_revisions(ROLE, SCOPE, claim_sources=rig.pairs)
    assert len(revisions) == 1 and set(revisions[0]) == {"memory_id", "source_message_id", "observed_at", "body_sha256"}
    prepared = await rig.context.prepare_turn(rig.turn, ROLE)
    assert prepared.relationship == RELATION and prepared.compiled.memory_status == "available"
    assert prepared.compiled.used_memory_ids == (str(rig.claim["id"]),)
    assert prepared.compiled.memory_packets[0].source_record_versions
    revalidator = make_private_context_revalidator(prepared, rig.database)
    refreshed = await revalidator(generation_request(prepared))
    assert refreshed.character_context.used_memory_ids == prepared.compiled.used_memory_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", BAD_ORIGINS)
async def test_malformed_stored_origin_fails_exact_reader_and_real_context_preparation(revision_rig, raw):
    rig = revision_rig
    corrupt_origin(rig, raw)
    with pytest.raises(ValueError) as exact_error:
        await rig.repository.linked_source_revisions(ROLE, SCOPE, claim_sources=rig.pairs)
    with pytest.raises(ValueError) as context_error:
        await rig.context.prepare_turn(rig.turn, ROLE)
    assert PRIVATE_BAD_JSON not in str(exact_error.value) + str(context_error.value)
    # This is a read failure, never an erasure or a repair of corrupt storage.
    assert await rig.repository.get_memory_record(int(rig.claim["id"]), ROLE, SCOPE) is not None
    assert (await rig.repository.get_relationship_record(ROLE, SCOPE))["relationship_stage"] == RELATION.stage


@pytest.mark.asyncio
async def test_origin_corruption_after_healthy_prepare_is_not_silent_authority_loss(revision_rig):
    rig = revision_rig
    prepared = await rig.context.prepare_turn(rig.turn, ROLE)
    assert prepared.compiled.memory_packets and prepared.compiled.memory_packets[0].source_record_versions
    corrupt_origin(rig, PRIVATE_BAD_JSON)
    revalidator = make_private_context_revalidator(prepared, rig.database)
    with pytest.raises(ValueError):
        await revalidator(generation_request(prepared))


@pytest.mark.asyncio
async def test_actual_api_preparation_reports_corrupt_origin_without_generic_no_match(revision_rig):
    rig = revision_rig
    corrupt_origin(rig, PRIVATE_BAD_JSON)
    request = MessageRequest(
        message=QUERY, platform=SCOPE.platform, adapter=SCOPE.adapter,
        userId=SCOPE.sender_id, senderId=SCOPE.sender_id, sessionId=SCOPE.conversation_id,
        conversationId=SCOPE.conversation_id, conversationType=SCOPE.conversation_type,
    )
    with pytest.raises(HTTPException) as caught:
        await _prepare_character_turn(request, ROLE, character_service=rig.context)
    assert caught.value.status_code == 503 and isinstance(caught.value.__cause__, ValueError)
    assert PRIVATE_BAD_JSON not in str(caught.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("outside", ["conversation", "owner", "character"])
async def test_valid_but_unauthorized_original_is_excluded_without_error(revision_rig, outside):
    rig = revision_rig
    other_scope = replace(SCOPE, conversation_id="fiction-room-b") if outside == "conversation" else (
        replace(SCOPE, sender_id="fiction-other-owner") if outside == "owner" else SCOPE
    )
    other_role = "fiction-other-role" if outside == "character" else ROLE
    assert await rig.repository.capture_source(
        other_role, other_scope, source_message_id=SOURCE_ID, body="另一合法范围中的完整虚构原话。",
        observed_at=rig.stamp,
    ) == "recorded"
    other_identity = source_identity(source_scope(
        other_role, other_scope.platform, other_scope.adapter, other_scope.sender_id,
        other_scope.conversation_type, other_scope.conversation_id,
    ), SOURCE_ID)
    connection = rig.database._get_connection()
    connection.execute("DELETE FROM memory_source_links WHERE memory_id=?", (rig.claim["id"],))
    connection.execute("INSERT INTO memory_source_links (memory_id,source_key) VALUES (?,?)",
                       (rig.claim["id"], other_identity["source_key"]))
    connection.commit()
    assert await rig.repository.linked_source_revisions(ROLE, SCOPE, claim_sources=rig.pairs) == []
    prepared = await rig.context.prepare_turn(rig.turn, ROLE)
    refreshed = await make_private_context_revalidator(prepared, rig.database)(generation_request(prepared))
    assert refreshed.character_context.memory_packets == ()
    assert refreshed.character_context.used_memory_ids == ()
    assert refreshed.character_context.memory_status == "no_match"


@pytest.mark.asyncio
async def test_valid_multiple_origins_with_same_pair_remain_ambiguous(tmp_path, monkeypatch):
    rig = await arrange_revision_scope(tmp_path, monkeypatch, scope_level="user_global")
    try:
        # A single correctly linked expanded claim is authorized first.
        first = await rig.repository.linked_source_revisions(ROLE, SCOPE, claim_sources=rig.pairs)
        assert len(first) == 1
        other = replace(SCOPE, conversation_id="fiction-room-b")
        assert await rig.repository.capture_source(
            ROLE, other, source_message_id=SOURCE_ID, body="同一所有者另一会话中的完整虚构原话。", observed_at=rig.stamp,
        ) == "recorded"
        other_identity = source_identity(source_scope(
            ROLE, other.platform, other.adapter, other.sender_id, other.conversation_type, other.conversation_id,
        ), SOURCE_ID)
        connection = rig.database._get_connection()
        connection.execute("INSERT INTO memory_source_links (memory_id,source_key) VALUES (?,?)",
                           (rig.claim["id"], other_identity["source_key"]))
        connection.commit()
        # Both origins are well formed and legal for user_global. The legacy
        # (claim, source_message_id) pair cannot distinguish them, so stays empty.
        assert await rig.repository.linked_source_revisions(ROLE, SCOPE, claim_sources=rig.pairs) == []
        assert await rig.repository.get_memory_record(int(rig.claim["id"]), ROLE, SCOPE) is not None
    finally:
        rig.database.close_connection()


async def baseline_preflight(workdir):
    """Run explicitly with asyncio.run; returns safe counts/types, never text.

    On the unfixed baseline this observes successful context preparation and
    final no_match/zero packets. After the fix it observes ValueError. Neither
    branch invokes a model, writes a correction or relaxes the pytest asserts.
    Use a fresh temporary directory for each call.
    """
    with pytest.MonkeyPatch.context() as patch:
        rig = await arrange_revision_scope(workdir, patch)
        try:
            healthy = await rig.context.prepare_turn(rig.turn, ROLE)
            assert healthy.relationship == RELATION and len(healthy.compiled.memory_packets) == 1
            assert healthy.compiled.memory_packets[0].source_record_versions
            result = {"healthy_packets": 1, "healthy_source_revisions": 1}
            corrupt_origin(rig, PRIVATE_BAD_JSON)
            try:
                after = await rig.context.prepare_turn(rig.turn, ROLE)
                refreshed = await make_private_context_revalidator(after, rig.database)(generation_request(after))
            except ValueError as error:
                return dict(result, corruption_raised=True, error_type=type(error).__name__)
            return dict(
                result, corruption_raised=False, prepared_packets=len(after.compiled.memory_packets),
                revision_status=after.memory_recall.get("claim_revision_status"),
                final_packets=len(refreshed.character_context.memory_packets),
                final_memory_status=refreshed.character_context.memory_status,
            )
        finally:
            rig.database.close_connection()
