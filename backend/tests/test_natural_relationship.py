"""Natural relationship behavior using isolated SQLite, without model inference."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.characters import router
from app.dependencies import get_current_admin
from app.providers import get_character_memory_repository
from character.context_builder import build_user_scope
from character.memory_extractor import next_relationship_stage
from character.models import CharacterProfile, RelationshipState, UserScope
from character.natural_relationship import (
    NoteCommand,
    active_note,
    compile_notes,
    parse_command,
    save_note,
)
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import CharacterContextService, TurnInput

SCOPE = UserScope("qq", "nonebot", "alice", "group1", "group")


@pytest.fixture
def repo(tmp_path):
    return DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "natural.db"))


@pytest.mark.parametrize(
    "text",
    [
        "小说台词：交流偏好：我喜欢茶",
        "“交流偏好：简短些”",
        "不要记住，交流偏好：简短些",
        "交流偏好：简短些？",
        "假如以后别问我工作",
        "你觉得我们更亲密了吗",
        "我很难过",
        "今天聊了一百次",
    ],
)
def test_no_inferred_relationship_write(text):
    assert parse_command(text) is None


@pytest.mark.parametrize(
    "text,category",
    [
        ("交流偏好：回答简短些", "preference"),
        ("以后别问我工资", "boundary"),
        ("今天不想被追问", "transient"),
        ("约定：下次讨论读书计划", "promise"),
        ("修复记录：不要拿我的成绩开玩笑", "repair"),
    ],
)
def test_explicit_commands(text, category):
    assert parse_command(text).category == category


@pytest.mark.parametrize("stage", ["stranger", "acquaintance", "familiar", "close"])
def test_counts_never_change_relationship(stage):
    for count in (0, 10, 50, 10000):
        assert next_relationship_stage(stage, count) == stage


@pytest.mark.asyncio
async def test_notes_scope_expiry_replacement_and_clear(repo):
    first = await save_note(repo, "kisaki", SCOPE, NoteCommand("transient", "今天不想被追问"))
    assert active_note(first)
    assert not active_note(first, datetime.now(timezone.utc) + timedelta(hours=13))
    second = await save_note(repo, "kisaki", SCOPE, NoteCommand("transient", "今天只想安静聊聊"))
    notes = await repo.list_relationship_notes("kisaki", SCOPE)
    assert [r["id"] for r in notes] == [second["id"]]
    assert not await repo.list_relationship_notes("other", SCOPE)
    for scope in (
        UserScope("qq", "nonebot", "bob", "group1", "group"),
        UserScope("qq", "nonebot", "alice", "group2", "group"),
        UserScope("wechat", "nonebot", "alice", "group1", "group"),
    ):
        assert not await repo.list_relationship_notes("kisaki", scope)
    assert "今天只想安静聊聊" not in compile_notes(notes, "解除短期状态")
    await save_note(repo, "kisaki", SCOPE, NoteCommand("transient", "", True))
    assert not await repo.list_memory_records("kisaki", SCOPE, include_inactive=True)


@pytest.mark.asyncio
async def test_notes_deduplicate_and_do_not_enter_generic_recall(repo):
    from character.memory_service import CharacterMemoryService

    first = await save_note(repo, "kisaki", SCOPE, NoteCommand("boundary", "不要追问工资"))
    again = await save_note(repo, "kisaki", SCOPE, NoteCommand("boundary", "不要追问工资"))
    assert first["id"] == again["id"]
    service = CharacterMemoryService(repo)
    items, count = await service.load_relevant_memories("kisaki", SCOPE, "工资")
    assert not items and count == 0
    notes = await repo.list_relationship_notes("kisaki", SCOPE)
    assert "不要追问工资" in compile_notes(notes, "你好")


def test_bad_expiry_and_resolved_fail_closed():
    row = {"memory_key": "relationship:transient", "metadata": {"category": "transient"}}
    assert not active_note(row)
    row["valid_to"] = "invalid"
    assert not active_note(row)
    row.update(valid_to=None, metadata={"category": "boundary", "resolved": True})
    assert not active_note(row)


@pytest.mark.asyncio
async def test_explicit_correction_and_resolution_preserve_expiry(repo):
    first = await save_note(repo, "kisaki", SCOPE, NoteCommand("transient", "今天不想被追问"))
    command = parse_command("更正备忘录：今天不想被追问 => 今天可以提一个问题")
    notes = await repo.list_relationship_notes("kisaki", SCOPE)
    overlay = compile_notes(notes, "更正备忘录：今天不想被追问 => 今天可以提一个问题")
    assert "今天可以提一个问题" in overlay and "今天不想被追问" not in overlay
    corrected = await save_note(repo, "kisaki", SCOPE, command)
    assert corrected["valid_to"] == first["valid_to"]
    assert corrected["memory_key"] == first["memory_key"]
    ending = parse_command("结束备忘录：今天可以提一个问题")
    assert not compile_notes(await repo.list_relationship_notes("kisaki", SCOPE), "结束备忘录：今天可以提一个问题")
    await save_note(repo, "kisaki", SCOPE, ending)
    assert not compile_notes(await repo.list_relationship_notes("kisaki", SCOPE), "你好")
    assert await save_note(repo, "kisaki", SCOPE, command) is None
    assert "未匹配" in compile_notes([], "更正备忘录：不存在 => 新内容")


@pytest.mark.asyncio
async def test_private_scope_shared_only_with_same_user(repo):
    def private_scope(session):
        return build_user_scope(
            platform="qq", adapter="nonebot", sender_id="alice", conversation_id=session, conversation_type="private"
        )

    private = private_scope("session1")
    await save_note(repo, "kisaki", private, NoteCommand("preference", "回答简短"))
    assert await repo.list_relationship_notes("kisaki", private_scope("session2"))
    assert not await repo.list_relationship_notes("kisaki", SCOPE)


@pytest.mark.asyncio
async def test_reference_is_bounded_and_events_are_relevant_only(repo):
    await save_note(repo, "kisaki", SCOPE, NoteCommand("shared_event", "我们讨论过咖啡烘焙"))
    notes = await repo.list_relationship_notes("kisaki", SCOPE)
    assert not compile_notes(notes, "天气怎样")
    assert "咖啡烘焙" in compile_notes(notes, "谈谈咖啡")
    for i in range(12):
        await save_note(repo, "kisaki", SCOPE, NoteCommand("preference", f"偏好{i}" + "字" * 290))
    assert len(compile_notes(await repo.list_relationship_notes("kisaki", SCOPE), "你好")) < 3000


class Profiles:
    def get_profile(self, character_id):
        return CharacterProfile(character_id, "人物", relationship_style=("不以亲近为由追问",))


class EmptyMemory:
    async def load_relevant_memories(self, *args, **kwargs):
        return (), 0


@pytest.mark.asyncio
async def test_current_turn_overlay_then_successful_persistence(repo, monkeypatch):
    from character import memory_llm

    monkeypatch.setattr(memory_llm, "get_memory_enrichment_scheduler", lambda: SimpleNamespace(enabled=False))
    service = CharacterContextService(Profiles(), repo, None, memory_service=EmptyMemory())
    turn = TurnInput(
        "交流偏好：不要追问我的收入",
        "qq",
        "nonebot",
        "alice",
        "group1",
        "group",
        history=({"role": "user", "content": "你好"},),
    )
    prepared = await service.prepare_turn(turn, "kisaki")
    assert "不要追问我的收入" in prepared.compiled.reference_context
    assert "不要追问我的收入" not in prepared.compiled.dynamic_context
    assert "不以亲近为由追问" in prepared.compiled.dynamic_context
    assert not await repo.list_relationship_notes("kisaki", SCOPE)  # prepare never writes
    await repo.upsert_relationship("kisaki", SCOPE, RelationshipState(stage="close"))
    await service.complete_turn(prepared, turn, "知道了", source_message_id="m1")
    assert len(await repo.list_relationship_notes("kisaki", SCOPE)) == 1
    assert (await repo.get_relationship("kisaki", SCOPE)).stage == "close"  # no stale override
    plain = TurnInput("你好", "qq", "nonebot", "alice", "group1", "group", history=turn.history)
    prepared = await service.prepare_turn(plain, "kisaki")
    assert "不要追问我的收入" in prepared.compiled.reference_context


@pytest.mark.asyncio
async def test_fiction_does_not_fall_through_to_memory(repo, monkeypatch):
    from character import memory_llm

    monkeypatch.setattr(memory_llm, "get_memory_enrichment_scheduler", lambda: SimpleNamespace(enabled=False))
    service = CharacterContextService(Profiles(), repo, None, memory_service=EmptyMemory())
    turn = TurnInput(
        "小说台词：我喜欢咖啡，叫我小明",
        "qq",
        "nonebot",
        "alice",
        "group1",
        "group",
        history=({"role": "user", "content": "你好"},),
    )
    prepared = await service.prepare_turn(turn, "kisaki")
    await service.complete_turn(prepared, turn, "好的")
    assert not await repo.list_memory_records("kisaki", SCOPE)
    assert not (await repo.get_relationship("kisaki", SCOPE)).preferred_address


def test_admin_note_crud_preserves_expiry_and_isolation(repo):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_admin] = lambda: {"role": "admin"}
    app.dependency_overrides[get_character_memory_repository] = lambda: repo
    params = dict(
        platform="qq", adapter="nonebot", sender_id="alice", conversation_id="group1", conversation_type="group"
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/characters/kisaki/relationship-notes",
            params=params,
            json={"category": "transient", "content": "今天不想要建议"},
        )
        assert created.status_code == 200
        original = created.json()["memory"]
        url = f"/api/characters/kisaki/memories/{original['id']}"
        assert client.put(url, params={**params, "sender_id": "bob"}, json={"content": "bad"}).status_code == 404
        updated = client.put(url, params=params, json={"content": "今天只想闲聊", "resolved": True})
        assert updated.status_code == 200
        row = updated.json()["memory"]
        assert row["valid_to"] == original["valid_to"]
        assert row["metadata"]["resolved"] and not active_note(row)
        assert client.delete(f"/api/characters/kisaki/memories/{row['id']}", params=params).status_code == 200
        assert client.get("/api/characters/kisaki/memories", params=params).json()["memories"] == []
        assert (
            client.post(
                "/api/characters/kisaki/relationship-notes",
                params=params,
                json={"category": "preference", "content": "   "},
            ).status_code
            == 400
        )
    app.dependency_overrides.pop(get_current_admin)
    with TestClient(app) as client:
        assert client.post(
            "/api/characters/kisaki/relationship-notes",
            params=params,
            json={"category": "preference", "content": "简短"},
        ).status_code in (401, 403)
