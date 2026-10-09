"""Container paths, folder detachment and cascades share durable authority."""

import sqlite3

import pytest

from api import knowledge
from db import knowledge_index_state as state
from db.database import SQLiteDB
from db.schemas import KnowledgeBaseUpdate

BODY = "完整合成规则：2026年12月20日周日16:10上课，须收到书面确认。"


def setup(db):
    kb = db.create_knowledge_base("旧资料组", "保留描述")
    folder = db.create_knowledge_folder(kb["id"], "旧目录")
    doc = db.save_knowledge_document(
        {
            "title": "完整现行规则",
            "content": BODY,
            "knowledge_base_id": kb["id"],
            "folder_id": folder["id"],
            "category": folder["name"],
        },
        chunks=[BODY],
    )
    other = db.create_knowledge_base("其他资料组", "别的来源")
    another = db.save_knowledge_document(
        {"title": "其他完整规则", "content": "完整其他规则，不能代替现行课。", "knowledge_base_id": other["id"]},
        chunks=["完整其他规则，不能代替现行课。"],
    )
    revision = db.get_config_value(state.REVISION_KEY)
    assert db.commit_knowledge_index_revision(revision, 2, "abc123")
    return kb, folder, doc, another, revision


def snapshot(db):
    return {
        table: db.execute_sql("SELECT * FROM " + table + " ORDER BY 1")
        for table in ["knowledge_bases", "knowledge_folders", "knowledge_documents", "knowledge_chunks", "config"]
    }


@pytest.mark.parametrize(
    "data,dirty",
    [
        ({"name": "新资料组"}, True),
        ({"description": "新描述"}, False),
        ({"name": "旧资料组"}, False),
        ({"name": None}, False),
        ({}, False),
    ],
)
def test_partial_base_updates_preserve_other_fields_and_dirty_only_new_path(tmp_path, data, dirty):
    db = SQLiteDB(tmp_path / "paths.db")
    kb, folder, doc, another, revision = setup(db)
    chunks = db.get_knowledge_chunks(doc["id"])
    new = db.update_knowledge_base(kb["id"], data)
    assert new["name"] == (data.get("name") or kb["name"]) and new["description"] == data.get(
        "description", kb["description"]
    )
    assert (
        db.get_knowledge_document(doc["id"]) == doc
        and db.get_knowledge_chunks(doc["id"]) == chunks
        and db.get_knowledge_document(another["id"]) == another
    )
    assert db.get_config_value(state.REVISION_KEY) == revision + int(dirty)
    assert db.get_config_value(state.STATUS_KEY) == ("dirty" if dirty else f"complete:2:abc123:{revision}")


@pytest.mark.parametrize("operation", ["rename", "base_delete", "folder_delete"])
def test_actual_dirty_write_failure_rolls_back_all_container_and_source_changes(tmp_path, operation):
    db = SQLiteDB(tmp_path / "paths.db")
    kb, folder, _, _, _ = setup(db)
    before = snapshot(db)
    db.execute_sql(
        "CREATE TRIGGER reject_container_dirty BEFORE INSERT ON config WHEN NEW.key='vector_index_rebuild_status' AND NEW.value='dirty' BEGIN SELECT RAISE(ABORT,'container dirty failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="container dirty failure"):
        if operation == "rename":
            db.update_knowledge_base(kb["id"], {"name": "新资料组"})
        elif operation == "base_delete":
            db.delete_knowledge_base(kb["id"])
        else:
            db.delete_knowledge_folder(folder["id"])
    assert snapshot(db) == before


@pytest.mark.parametrize("operation", ["rename", "base_delete", "folder_delete"])
def test_missing_container_does_not_mutate_authority(tmp_path, operation):
    db = SQLiteDB(tmp_path / "paths.db")
    setup(db)
    before = snapshot(db)
    if operation == "rename":
        assert db.update_knowledge_base(999, {"name": "不存在"}) is None
    elif operation == "base_delete":
        assert db.delete_knowledge_base(999) is False
    else:
        assert db.delete_knowledge_folder(999) is False
    assert snapshot(db) == before


def test_folder_delete_keeps_rules_and_chunks_under_uncategorized(tmp_path):
    db = SQLiteDB(tmp_path / "paths.db")
    kb, folder, doc, another, revision = setup(db)
    chunks = db.get_knowledge_chunks(doc["id"])
    assert db.delete_knowledge_folder(folder["id"])
    current = db.get_knowledge_document(doc["id"])
    assert (
        db.get_knowledge_folder(folder["id"]) is None
        and current["folder_id"] is None
        and current["category"] == "未分类"
        and current["knowledge_base_id"] == kb["id"]
    )
    assert (
        current["content"] == BODY
        and current["title"] == doc["title"]
        and db.get_knowledge_chunks(doc["id"]) == chunks
        and db.get_knowledge_document(another["id"]) == another
    )
    assert db.get_config_value(state.REVISION_KEY) == revision + 1 and db.get_config_value(state.STATUS_KEY) == "dirty"


def test_base_cascade_removes_its_full_sources_and_dirties_once(tmp_path):
    db = SQLiteDB(tmp_path / "paths.db")
    kb, folder, doc, another, revision = setup(db)
    chunks = db.get_knowledge_chunks(another["id"])
    assert db.delete_knowledge_base(kb["id"])
    assert (
        db.get_knowledge_base(kb["id"]) is None
        and db.get_knowledge_folder(folder["id"]) is None
        and db.get_knowledge_document(doc["id"]) is None
        and db.get_knowledge_chunks(doc["id"]) == []
    )
    assert (
        db.get_knowledge_document(another["id"]) == another
        and db.get_knowledge_chunks(another["id"]) == chunks
        and db.get_config_value(state.REVISION_KEY) == revision + 1
        and db.get_config_value(state.STATUS_KEY) == "dirty"
    )


def test_duplicate_rename_rolls_back_and_connection_remains_usable(tmp_path):
    db = SQLiteDB(tmp_path / "paths.db")
    kb, _, _, _, revision = setup(db)
    before = snapshot(db)
    with pytest.raises(sqlite3.IntegrityError):
        db.update_knowledge_base(kb["id"], {"name": "其他资料组", "description": "不应部分写入"})
    assert snapshot(db) == before
    assert (
        db.update_knowledge_base(kb["id"], {"name": "新资料组"})["name"] == "新资料组"
        and db.get_config_value(state.REVISION_KEY) == revision + 1
    )


def test_invalid_revision_rolls_back_rename_without_resetting_authority(tmp_path):
    db = SQLiteDB(tmp_path / "paths.db")
    kb, _, _, _, _ = setup(db)
    db.set_config_value(state.REVISION_KEY, "bad")
    before = snapshot(db)
    with pytest.raises(ValueError):
        db.update_knowledge_base(kb["id"], {"name": "新资料组"})
    assert snapshot(db) == before


@pytest.mark.parametrize("operation", ["rename", "base_delete", "folder_delete"])
@pytest.mark.asyncio
async def test_public_container_route_requires_no_postcommit_database_signal(tmp_path, monkeypatch, operation):
    db = SQLiteDB(tmp_path / "paths.db")
    kb, folder, doc, _, revision = setup(db)
    monkeypatch.setattr(knowledge, "db", db)

    def forbidden():
        raise AssertionError("Public container mutation must already have durable dirty authority")

    monkeypatch.setattr(db, "mark_knowledge_index_dirty", forbidden)
    admin = {"role": "admin"}
    if operation == "rename":
        result = await knowledge.update_knowledge_base(kb["id"], KnowledgeBaseUpdate(name="新资料组"), admin)
    elif operation == "base_delete":
        result = await knowledge.delete_knowledge_base(kb["id"], admin)
    else:
        result = await knowledge.delete_knowledge_folder(folder["id"], admin)
        assert db.get_knowledge_document(doc["id"])["category"] == "未分类"
    assert (
        result["success"]
        and db.get_config_value(state.REVISION_KEY) == revision + 1
        and db.get_config_value(state.STATUS_KEY) == "dirty"
    )
