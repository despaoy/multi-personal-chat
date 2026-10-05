"""Fictional storage component controls; real model grant checked separately."""

import hashlib
import json
from datetime import datetime, timezone

import pytest

from character.context_builder import compile_reference_context
from character.memory_read_authority import record_version
from character.memory_service import CharacterMemoryService
from character.models import CompiledCharacterContext, UserScope
from db.database import SQLiteDB
from db.memory_source import source_identity, source_scope
from inference.generation_request import GenerationRequest
from inference.structured_context_authority import revalidate_private_memories
from repositories.character_memory import DatabaseCharacterMemoryRepository

ROLE = "fiction-role"
BODY = "我喜欢荆谷饼。请记住，以后这个角色记住本人偏好，在我其他会话继续使用；没有授权其他角色。"


def fields(owner="fixture-owner", conversation="fixture-origin", role=ROLE):
    return dict(
        character_id=role,
        platform="web",
        adapter="revision-component",
        sender_id=owner,
        conversation_type="group",
        conversation_id=conversation,
    )


def prepare(tmp_path, level="user_character", key="preference_荆谷饼", content="用户说喜欢荆谷饼"):
    db = SQLiteDB(tmp_path / "component.sqlite")
    self_clause = "我叫芷林。" if key == "user_name" else "我喜欢荆谷饼。"
    body = (
        self_clause + "请所有角色记住本人信息，允许我的其他会话继续使用。"
        if level == "user_global"
        else self_clause + BODY[len("我喜欢荆谷饼。") :]
    )
    f = fields()
    db.capture_memory_source(**f, source_message_id="fixture-source", body=body, observed_at=datetime.now(timezone.utc))
    # This is an explicit database fixture, never a purported model proposal.
    claim = db.append_character_memory_claim(
        **f,
        scope_level=level,
        memory_type="user_fact",
        memory_key=key,
        content=content,
        evidence_json=json.dumps([self_clause]),
        source_message_id="fixture-source",
        confidence=0.95,
        observed_at=datetime.now(timezone.utc).isoformat(),
    )
    return db, claim


def read(db, claim, **changes):
    return db.linked_memory_source_revisions(
        **(fields() | changes), claim_sources=((int(claim["id"]), "fixture-source"),)
    )


def test_character_grant_fingerprint_survives_scope_switch_without_speech_grant(tmp_path):
    db, claim = prepare(tmp_path)
    original = read(db, claim)
    assert read(db, claim, conversation_id="fixture-destination") == original
    assert len(original) == 1 and set(original[0]) == {"memory_id", "source_message_id", "observed_at", "body_sha256"}
    assert original[0]["body_sha256"] == hashlib.sha256(BODY.encode()).hexdigest()
    destination = fields(conversation="fixture-destination")
    assert not db.list_memory_sources(**destination)
    assert not db.linked_memory_sources(**destination, memory_ids=(int(claim["id"]),))
    assert not db.linked_memory_source_receipts(**destination, claim_sources=((int(claim["id"]), "fixture-source"),))


@pytest.mark.parametrize(
    "change",
    [
        {"sender_id": "other-owner"},
        {"character_id": "other-role"},
        {"platform": "other-platform"},
        {"adapter": "other-adapter"},
    ],
)
def test_character_fingerprint_cannot_follow_other_owner_role_or_adapter(tmp_path, change):
    db, claim = prepare(tmp_path)
    assert not read(db, claim, **change)


def test_conversation_claim_keeps_original_scope_boundary(tmp_path):
    db, claim = prepare(tmp_path, level="conversation")
    assert len(read(db, claim)) == 1
    assert not read(db, claim, conversation_id="fixture-destination")


def test_global_fingerprint_visibility_does_not_expand_original_speech(tmp_path):
    db, claim = prepare(tmp_path, level="user_global")
    other = fields(conversation="fixture-destination", role="other-role")
    assert read(db, claim, character_id="other-role", conversation_id="fixture-destination") == read(db, claim)
    assert not db.list_memory_sources(**other)
    assert not db.linked_memory_sources(**other, memory_ids=(int(claim["id"]),))
    assert not read(db, claim, character_id="other-role", sender_id="other-owner")


@pytest.mark.asyncio
async def test_scope_switch_packet_survives_but_changed_origin_invalidates_it(tmp_path):
    db, claim = prepare(tmp_path)
    repo = DatabaseCharacterMemoryRepository(db)
    s = UserScope("web", "revision-component", "fixture-owner", "fixture-destination", "group")
    packets, _ = await CharacterMemoryService(repo, semantic_enabled=False).load_relevant_memories(
        ROLE, s, "我的荆谷饼偏好是什么？", for_contextual_selection=True
    )
    assert len(packets) == 1 and packets[0].source_record_versions
    ref, ids = compile_reference_context(packets, complete_evidence=True, max_chars=None)
    context = CompiledCharacterContext(
        "虚构画像", "", ref, memory_status="available", used_memory_ids=ids, memory_packets=packets
    )
    request = GenerationRequest(message="只读核对本人偏好", character_context=context)
    assert (await revalidate_private_memories(request, repo, ROLE, s)).character_context.memory_packets == packets
    before = await repo.get_memory_record(int(claim["id"]), ROLE, s)
    # Explicit source-change fault injection; not new user speech or approval.
    con = db._get_connection()
    con.execute(
        "UPDATE memory_sources SET body=? WHERE source_message_id=?", ("完整未经本轮复核的新原话。", "fixture-source")
    )
    con.commit()
    assert record_version(await repo.get_memory_record(int(claim["id"]), ROLE, s)) == record_version(before)
    updated = await revalidate_private_memories(request, repo, ROLE, s)
    assert not updated.character_context.memory_packets


def test_ambiguous_original_scopes_cannot_collapse_to_one_fingerprint(tmp_path):
    db, claim = prepare(tmp_path)
    f = fields(conversation="fixture-destination")
    db.capture_memory_source(
        **f, source_message_id="fixture-source", body="完整不同会话原话。", observed_at=datetime.now(timezone.utc)
    )
    identity = source_identity(
        source_scope(
            *[
                f[n]
                for n in ("character_id", "platform", "adapter", "sender_id", "conversation_type", "conversation_id")
            ]
        ),
        "fixture-source",
    )
    # Deliberate linkage-corruption control, not an accepted model relationship.
    con = db._get_connection()
    con.execute(
        "INSERT INTO memory_source_links(memory_id,source_key) VALUES (?,?)", (claim["id"], identity["source_key"])
    )
    con.commit()
    assert not read(db, claim) and not read(db, claim, conversation_id="fixture-destination")


@pytest.mark.asyncio
async def test_open_compound_owner_heuristic_does_not_hide_name_from_reviewer(tmp_path):
    db, claim = prepare(tmp_path, key="user_name", content="用户说自己叫芷林")
    repo = DatabaseCharacterMemoryRepository(db)
    s = UserScope("web", "revision-component", "fixture-owner", "fixture-origin", "group")
    query = "只读核对我在其他会话允许保存的本人姓名，不能把朋友的姓名当成本人姓名。"
    packets, _, trace = await CharacterMemoryService(repo, semantic_enabled=False).recall_with_diagnostics(
        ROLE, s, query, for_contextual_selection=True
    )
    assert trace["subject_filter_mode"] == "contextual_review"
    assert any(item.memory_key == "user_name" for item in packets)
    packets, _, _ = await CharacterMemoryService(repo, semantic_enabled=False).recall_with_diagnostics(
        ROLE, s, "他的名字是什么？", for_contextual_selection=True
    )
    assert not packets
