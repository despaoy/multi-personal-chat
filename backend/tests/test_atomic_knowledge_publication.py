"""Atomic public document publication, rollback and supported import paths."""

import io
import sqlite3

import pytest
from fastapi import UploadFile

from api import knowledge
from db import knowledge_index_state as state
from db.database import SQLiteDB
from db.schemas import KnowledgeDocumentCreate, KnowledgeDocumentUpdate

OLD = "完整旧规则：12月13日15:20在二楼上课，须收到书面确认。"
NEW = "完整新规则：12月20日16:10在三楼上课，须收到书面确认。"


def source(db):
    d = db.save_knowledge_document({"title": "旧规则", "content": OLD}, chunks=[OLD])
    revision = db.get_config_value(state.REVISION_KEY)
    assert db.commit_knowledge_index_revision(revision, 1, "abc123")
    return d, revision


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
def test_actual_dirty_write_failure_rolls_back_whole_publication(tmp_path, operation):
    db = SQLiteDB(tmp_path / "atomic.db")
    d, revision = source(db)
    chunks = db.get_knowledge_chunks(d["id"])
    authority = db.config
    db.execute_sql(
        "CREATE TRIGGER reject_dirty BEFORE INSERT ON config WHEN NEW.key='vector_index_rebuild_status' AND NEW.value='dirty' BEGIN SELECT RAISE(ABORT,'actual dirty failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="actual dirty failure"):
        if operation == "create":
            db.save_knowledge_document({"title": "新规则", "content": NEW}, chunks=[NEW])
        elif operation == "update":
            db.save_knowledge_document({"title": "新规则", "content": NEW}, doc_id=d["id"], chunks=[NEW])
        else:
            db.delete_knowledge_document(d["id"])
    assert db.get_knowledge_documents() == [d] and db.get_knowledge_chunks(d["id"]) == chunks and db.config == authority


def test_second_chunk_failure_rolls_back_document_and_prior_chunks(tmp_path):
    db = SQLiteDB(tmp_path / "atomic.db")
    d, _ = source(db)
    chunks = db.get_knowledge_chunks(d["id"])
    authority = db.config
    db.execute_sql(
        "CREATE TRIGGER reject_second BEFORE INSERT ON knowledge_chunks WHEN NEW.chunkIndex=1 BEGIN SELECT RAISE(ABORT,'second chunk failed'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="second chunk failed"):
        db.save_knowledge_document({"title": "新规则", "content": NEW}, doc_id=d["id"], chunks=[NEW[:10], NEW[10:]])
    assert (
        db.get_knowledge_document(d["id"]) == d
        and db.get_knowledge_chunks(d["id"]) == chunks
        and db.config == authority
    )


def test_successful_body_publication_changes_all_state_and_revision_once(tmp_path):
    db = SQLiteDB(tmp_path / "atomic.db")
    d, revision = source(db)
    new = db.save_knowledge_document(
        {"title": "新规则", "content": NEW, "id": 999, "createdAt": "forged", "unknown": "ignored"},
        doc_id=d["id"],
        chunks=[NEW[:10], NEW[10:]],
    )
    assert (
        new["id"] == d["id"] and new["createdAt"] == d["createdAt"] and new["content"] == NEW and new["chunkCount"] == 2
    )
    assert (
        "".join(c["content"] for c in db.get_knowledge_chunks(d["id"])) == NEW
        and db.get_config_value(state.REVISION_KEY) == revision + 1
        and db.get_config_value(state.STATUS_KEY) == "dirty"
    )
    assert not db.commit_knowledge_index_revision(revision, 1, "old")


@pytest.mark.parametrize("metadata", [{"title": "新标题"}, {"category": "新分类"}, {"sourceType": "file"}])
def test_metadata_only_change_preserves_full_chunks_and_invalidates_once(tmp_path, metadata):
    db = SQLiteDB(tmp_path / "atomic.db")
    d, revision = source(db)
    chunks = db.get_knowledge_chunks(d["id"])
    new = db.save_knowledge_document(metadata, doc_id=d["id"])
    assert (
        all(new[k] == v for k, v in metadata.items())
        and db.get_knowledge_chunks(d["id"]) == chunks
        and db.get_config_value(state.REVISION_KEY) == revision + 1
    )


def test_unchanged_or_nonindexed_file_metadata_keeps_authority(tmp_path):
    db = SQLiteDB(tmp_path / "atomic.db")
    d, revision = source(db)
    status = db.get_config_value(state.STATUS_KEY)
    db.save_knowledge_document({"title": d["title"], "fileSize": 12}, doc_id=d["id"])
    assert db.get_config_value(state.REVISION_KEY) == revision and db.get_config_value(state.STATUS_KEY) == status


def test_empty_replacement_removes_all_old_chunks_and_updates_count(tmp_path):
    db = SQLiteDB(tmp_path / "atomic.db")
    d, revision = source(db)
    new = db.save_knowledge_document({"content": ""}, doc_id=d["id"], chunks=[])
    assert (
        new["content"] == ""
        and new["chunkCount"] == 0
        and db.get_knowledge_chunks(d["id"]) == []
        and db.get_config_value(state.REVISION_KEY) == revision + 1
    )


def test_missing_document_does_not_dirty_existing_index(tmp_path):
    db = SQLiteDB(tmp_path / "atomic.db")
    _, revision = source(db)
    status = db.get_config_value(state.STATUS_KEY)
    assert db.save_knowledge_document({"title": "不存在"}, doc_id=999) is None and not db.delete_knowledge_document(999)
    assert db.get_config_value(state.REVISION_KEY) == revision and db.get_config_value(state.STATUS_KEY) == status


def test_content_without_complete_replacement_is_rejected_without_partial_commit(tmp_path):
    db = SQLiteDB(tmp_path / "atomic.db")
    d, _ = source(db)
    with pytest.raises(ValueError, match="complete replacement"):
        db.save_knowledge_document({"content": NEW}, doc_id=d["id"])
    assert db.get_knowledge_document(d["id"]) == d


@pytest.mark.parametrize("route", ["create", "zip", "scan_root", "scan_folder", "update", "delete"])
@pytest.mark.asyncio
async def test_public_document_paths_use_atomic_publication_before_local_notification(tmp_path, monkeypatch, route):
    db = SQLiteDB(tmp_path / "api.db")
    monkeypatch.setattr(knowledge, "db", db)
    monkeypatch.setattr(knowledge, "VECTOR_DB_AVAILABLE", False)
    monkeypatch.setattr(knowledge, "INPUT_VALIDATOR_AVAILABLE", False)

    def forbidden():
        raise AssertionError("Legacy post-commit database notification must not run")

    monkeypatch.setattr(knowledge, "_mark_rebuild_dirty", forbidden)
    kb = db.create_knowledge_base("完整合成库")
    admin = {"role": "admin"}
    before = 0
    if route in {"update", "delete"}:
        d, before = source(db)
        if route == "update":
            result = await knowledge.update_knowledge_document(
                d["id"], KnowledgeDocumentUpdate(title="新规则", content=NEW), admin
            )
        else:
            result = await knowledge.delete_knowledge_document(d["id"], admin)
    elif route == "create":
        result = await knowledge.create_knowledge_document(
            KnowledgeDocumentCreate(title="新规则", content=NEW, knowledge_base_id=kb["id"]), admin
        )
    elif route == "zip":
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("当前规则/新规则.txt", NEW)
        buffer.seek(0)
        result = await knowledge.upload_zip(kb["id"], UploadFile(buffer, filename="完整规则.zip"), admin)
    else:
        root = tmp_path / "knowledge_bases"
        target = root / "合成来源"
        target.mkdir(parents=True)
        folder = target
        if route == "scan_folder":
            folder = target / "当前规则"
            folder.mkdir()
        (folder / "新规则.txt").write_text(NEW, encoding="utf-8")
        monkeypatch.setattr(knowledge, "KNOWLEDGE_BASES_DIR", root)
        result = await knowledge.import_scanned_directory("合成来源", kb["id"], admin)
    assert (
        result["success"]
        and db.get_config_value(state.REVISION_KEY) == before + 1
        and db.get_config_value(state.STATUS_KEY) == "dirty"
    )
    docs = db.get_knowledge_documents()
    if route == "delete":
        assert docs == []
    else:
        assert (
            len(docs) == 1
            and docs[0]["content"] == NEW
            and docs[0]["chunkCount"] == len(db.get_knowledge_chunks(docs[0]["id"]))
        )
        assert "".join(c["content"] for c in db.get_knowledge_chunks(docs[0]["id"])) == NEW
