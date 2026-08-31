"""Versioned character-memory claim persistence tests."""

from __future__ import annotations

import importlib.util
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import sqlalchemy as sa

from character.models import MemoryItem, UserScope
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

SCOPE = {
    "character_id": "kisaki",
    "platform": "qq",
    "adapter": "nonebot",
    "sender_id": "user_1",
    "conversation_type": "group",
    "conversation_id": "group_1",
}


def _db(tmp_path) -> SQLiteDB:
    return SQLiteDB(tmp_path / "versioned-memory.db")


def _scope(conversation_id: str = "group_1") -> UserScope:
    return UserScope("qq", "nonebot", "user_1", conversation_id, "group")


def _memory(content: str, memory_type: str = "user_fact") -> MemoryItem:
    return MemoryItem(
        memory_id="",
        memory_type=memory_type,  # type: ignore[arg-type]
        content=content,
        importance=0.8,
    )


def test_schema_and_legacy_upsert_keep_revision_zero(tmp_path):
    db = _db(tmp_path)
    first = db.add_or_update_character_memory(
        **SCOPE,
        memory_type="user_fact",
        memory_key="user_name",
        content="用户叫小明",
    )
    second = db.add_or_update_character_memory(
        **SCOPE,
        memory_type="user_fact",
        memory_key="user_name",
        content="用户叫大明",
    )

    assert first["id"] == second["id"]
    assert second["revision"] == 0
    assert second["scope_level"] == "conversation"
    assert second["status"] == "active"

    conn = sqlite3.connect(db.db_path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(character_memories)")}
        unique_columns = []
        for index in conn.execute("PRAGMA index_list(character_memories)"):
            if index[2]:
                names = [row[2] for row in conn.execute(f'PRAGMA index_info("{index[1]}")')]
                if "revision" in names:
                    unique_columns = names
                    break
    finally:
        conn.close()

    assert {
        "scope_level",
        "revision",
        "relation_type",
        "status",
        "parent_memory_id",
        "supersedes_memory_id",
        "confidence",
        "source_message_ids_json",
        "evidence_json",
        "metadata_json",
        "valid_from",
        "valid_to",
        "observed_at",
    } <= columns
    assert unique_columns[-3:] == ["scope_level", "memory_key", "revision"]


async def test_append_supersede_pending_retract_and_noop(tmp_path):
    repo = DatabaseCharacterMemoryRepository(_db(tmp_path))
    scope = _scope()
    first = await repo.append_claim("kisaki", scope, _memory("用户喜欢清咖啡"), memory_key="coffee")
    second = await repo.append_claim(
        "kisaki",
        scope,
        _memory("用户只是不喜欢太苦的咖啡"),
        memory_key="coffee",
        relation_type="SUPERSEDE",
        parent_memory_id=first["id"],
        supersedes_memory_id=first["id"],
    )
    pending = await repo.append_claim(
        "kisaki",
        scope,
        _memory("用户可能明年换工作"),
        memory_key="future_job",
        relation_type="PENDING",
    )
    noop = await repo.append_claim(
        "kisaki",
        scope,
        _memory("没有新事实"),
        memory_key="nothing",
        relation_type="NOOP",
    )

    active = await repo.list_memory_records("kisaki", scope, limit=50)
    audit = await repo.list_memory_records("kisaki", scope, limit=50, include_inactive=True)
    assert [row["id"] for row in active] == [second["id"]]
    assert next(row for row in audit if row["id"] == first["id"])["status"] == "superseded"
    assert pending["status"] == "pending"
    assert noop["persisted"] is False
    assert all(row["memory_key"] != "nothing" for row in audit)

    retraction = await repo.append_claim(
        "kisaki",
        scope,
        _memory("撤回此前咖啡偏好"),
        memory_key="coffee",
        relation_type="RETRACT",
        parent_memory_id=second["id"],
    )
    assert retraction["status"] == "retracted"
    assert await repo.list_memory_records("kisaki", scope, limit=50) == []


async def test_three_scope_layers_and_evidence_decode(tmp_path):
    repo = DatabaseCharacterMemoryRepository(_db(tmp_path))
    group_1 = _scope("group_1")
    group_2 = _scope("group_2")

    await repo.append_claim("kisaki", group_1, _memory("仅当前群可见"), memory_key="conversation")
    await repo.append_claim(
        "kisaki",
        group_1,
        _memory("所有与妃的对话可见"),
        memory_key="character",
        scope_level="user_character",
    )
    global_claim = await repo.append_claim(
        "kisaki",
        group_1,
        _memory("用户长期偏好简洁回答"),
        memory_key="answer_style",
        scope_level="user_global",
        evidence=({"message_id": "msg-1", "quote": "回答简短点"},),
        source_message_id="msg-1",
        source_message_ids=("msg-1", "msg-2"),
        confidence=0.88,
        attributed_to="user",
        metadata={"qualifiers": ["长期偏好"]},
    )

    current = await repo.list_memory_records("kisaki", group_1, limit=20)
    other_group = await repo.list_memory_records("kisaki", group_2, limit=20)
    other_character = await repo.list_memory_records("other", group_2, limit=20)

    assert {row["memory_key"] for row in current} == {"conversation", "character", "answer_style"}
    assert {row["memory_key"] for row in other_group} == {"character", "answer_style"}
    assert {row["memory_key"] for row in other_character} == {"answer_style"}
    assert global_claim["source_message_ids"] == ["msg-1", "msg-2"]
    assert global_claim["evidence"][0]["quote"] == "回答简短点"
    assert global_claim["metadata"]["qualifiers"] == ["长期偏好"]
    assert global_claim["attributed_to"] == "user"


def test_concurrent_coexist_claims_allocate_unique_revisions(tmp_path):
    db = _db(tmp_path)

    def write(index: int):
        return db.append_character_memory_claim(
            **SCOPE,
            memory_type="shared_event",
            memory_key="pets",
            content=f"用户养了宠物 {index}",
            relation_type="COEXIST",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(write, range(12)))

    assert sorted(record["revision"] for record in records) == list(range(1, 13))
    assert len({record["id"] for record in records}) == 12


async def test_erase_physically_removes_complete_derived_lineage(tmp_path):
    repo = DatabaseCharacterMemoryRepository(_db(tmp_path))
    scope = _scope()
    first = await repo.append_claim("kisaki", scope, _memory("敏感原始事实"), memory_key="private")
    second = await repo.append_claim(
        "kisaki",
        scope,
        _memory("敏感事实修正版"),
        memory_key="private",
        relation_type="SUPERSEDE",
        supersedes_memory_id=first["id"],
    )
    await repo.append_claim(
        "kisaki",
        scope,
        _memory("从敏感事实归纳出的结论"),
        memory_key="private_reflection",
        relation_type="COEXIST",
        parent_memory_id=second["id"],
    )

    deleted = await repo.erase_memory("kisaki", scope, memory_id=first["id"])
    assert deleted == 3
    assert await repo.list_memory_records("kisaki", scope, limit=50, include_inactive=True) == []


def test_erase_does_not_follow_forged_cross_user_child_reference(tmp_path):
    db = _db(tmp_path)
    user_a = dict(SCOPE)
    user_b = {**SCOPE, "sender_id": "user_2"}
    root = db.append_character_memory_claim(
        **user_a,
        memory_type="user_fact",
        memory_key="private_a",
        content="用户 A 的敏感事实",
    )
    forged_child = db.append_character_memory_claim(
        **user_b,
        memory_type="user_fact",
        memory_key="private_b",
        content="用户 B 的独立事实",
    )
    conn = db.get_connection()
    conn.execute(
        "UPDATE character_memories SET parent_memory_id = ? WHERE id = ?",
        (root["id"], forged_child["id"]),
    )
    conn.commit()

    deleted = db.erase_character_memories(**user_a, memory_id=root["id"])

    assert deleted == 1
    remaining = db.list_character_memory_claims(**user_b, limit=20, include_inactive=True)
    assert [row["id"] for row in remaining] == [forged_child["id"]]


def test_alembic_007_upgrades_old_sqlite_contract(tmp_path):
    migration_context_module = pytest.importorskip("alembic.migration")
    operations_module = pytest.importorskip("alembic.operations")
    MigrationContext = migration_context_module.MigrationContext
    Operations = operations_module.Operations
    path = tmp_path / "migration.db"
    engine = sa.create_engine(f"sqlite:///{path}")
    migration_path = (
        Path(__file__).resolve().parents[1] / "alembic" / "versions" / "007_versioned_character_memory_claims.py"
    )
    spec = importlib.util.spec_from_file_location("memory_migration_007", migration_path)
    assert spec and spec.loader
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)

    with engine.begin() as conn:
        conn.exec_driver_sql("""
            CREATE TABLE character_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                adapter TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                conversation_type TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.0,
                source_message_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CONSTRAINT uq_character_memory_key UNIQUE (
                    character_id, platform, adapter, sender_id,
                    conversation_type, conversation_id, memory_key
                )
            )
        """)
        conn.exec_driver_sql("""
            INSERT INTO character_memories (
                character_id, platform, adapter, sender_id, conversation_type,
                conversation_id, memory_type, memory_key, content, importance,
                created_at, updated_at
            ) VALUES ('c', 'p', 'a', 'u', 'private', 'u', 'user_fact', 'name',
                      'old', 0.5, '2026-01-01', '2026-01-01')
        """)
        migration.op = Operations(MigrationContext.configure(conn))
        migration.upgrade()

        inspector = sa.inspect(conn)
        columns = {column["name"]: column for column in inspector.get_columns("character_memories")}
        uniques = inspector.get_unique_constraints("character_memories")
        row = conn.execute(sa.text("SELECT * FROM character_memories")).mappings().one()

    assert row["revision"] == 0
    assert row["scope_level"] == "conversation"
    assert str(columns["revision"]["default"]).replace("'", "") in {"1", "(1)"}
    assert any(item["column_names"][-3:] == ["scope_level", "memory_key", "revision"] for item in uniques)
