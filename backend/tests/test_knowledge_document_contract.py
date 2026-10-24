"""Document patches preserve meaning and failed writes expose only safe errors."""

from knowledge import index_lifecycle

import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from api import knowledge
from db.database import SQLiteDB
from db.schemas import KnowledgeDocumentCreate, KnowledgeDocumentUpdate

OLD = "完整旧规则：10月23日15点在二楼办理，本人携身份证和书面确认，未确认不得付款。"
NEW = "完整新规则：10月26日16点在三楼办理，本人携身份证和书面确认，未确认不得付款；截止为25日18点。"
ADMIN = {"role": "admin"}


@pytest.fixture
def documents(tmp_path, monkeypatch):
    db = SQLiteDB(tmp_path / "source.sqlite")
    base = db.create_knowledge_base("原合成库")
    folder = db.create_knowledge_folder(base["id"], "原目录")
    destination = db.create_knowledge_base("新合成库")
    target = db.create_knowledge_folder(destination["id"], "新目录")
    row = db.save_knowledge_document(
        dict(
            title="原规则",
            content=OLD,
            category="原目录",
            knowledge_base_id=base["id"],
            folder_id=folder["id"],
            sourceType="text",
            sourceUrl="https://example.invalid/old",
            fileType="txt",
            fileSize=100,
        ),
        chunks=[OLD],
    )
    revision = db.get_config_value(index_lifecycle._VECTOR_REBUILD_REVISION_KEY)
    assert db.commit_knowledge_index_revision(revision, 1, "synthetic")
    monkeypatch.setattr(knowledge, "db", db)
    monkeypatch.setattr(index_lifecycle, "db", db)
    monkeypatch.setattr(index_lifecycle, "_vector_index_revision", revision)
    return SimpleNamespace(db=db, row=row, revision=revision, destination=destination, target=target)


def snapshot(rig):
    return rig.db.get_knowledge_documents(), rig.db.get_knowledge_chunks(rig.row["id"]), rig.db.config


async def invoke(rig, operation):
    if operation == "create":
        return await knowledge.create_knowledge_document(
            KnowledgeDocumentCreate(
                title="新增规则", content=NEW, folder_id=rig.target["id"], knowledge_base_id=rig.destination["id"]
            ),
            ADMIN,
        )
    if operation == "update":
        return await knowledge.update_knowledge_document(rig.row["id"], KnowledgeDocumentUpdate(content=NEW), ADMIN)
    return await knowledge.delete_knowledge_document(rig.row["id"], ADMIN)


@pytest.mark.parametrize("mode", ["all", "metadata", "empty", "null", "omitted"])
async def test_document_patch_preserves_full_partial_empty_and_null_semantics(documents, mode):
    rig = documents
    if mode in {"all", "metadata"}:
        patch = dict(
            title="新规则",
            category="新目录",
            knowledge_base_id=rig.destination["id"],
            folder_id=rig.target["id"],
            sourceType="upload",
            sourceUrl="",
            fileType="md",
            fileSize=0,
        )
        if mode == "all":
            patch["content"] = NEW
    elif mode == "empty":
        patch = dict(title="", content="", category="", sourceType="", sourceUrl="", fileType="", fileSize=0)
    elif mode == "null":
        patch = dict.fromkeys(KnowledgeDocumentUpdate.model_fields)
    else:
        patch = {}
    old_chunks = rig.db.get_knowledge_chunks(rig.row["id"])
    request = KnowledgeDocumentUpdate(**patch)
    result = await knowledge.update_knowledge_document(rig.row["id"], request, ADMIN)
    assert result["success"]
    stored = rig.db.get_knowledge_document(rig.row["id"])
    for key in KnowledgeDocumentUpdate.model_fields:
        expected = patch[key] if key in patch and patch[key] is not None else rig.row[key]
        assert stored[key] == expected and result["document"][key] == expected
    chunks = rig.db.get_knowledge_chunks(rig.row["id"])
    if mode == "all":
        assert [row["content"] for row in chunks] == [NEW] and stored["chunkCount"] == 1
    elif mode == "empty":
        assert chunks == [] and stored["chunkCount"] == 0
    else:
        assert chunks == old_chunks and stored["chunkCount"] == 1
    changed = mode in {"all", "metadata", "empty"}
    assert index_lifecycle._get_rebuild_revision() == rig.revision + int(changed)
    assert index_lifecycle._read_rebuild_status()[0] == ("dirty" if changed else "complete")
    assert index_lifecycle._vector_index_revision is None


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
async def test_atomic_document_write_error_preserves_data_and_safe_cause(documents, caplog, operation):
    rig = documents
    before = snapshot(rig)
    rig.db.execute_sql(
        "CREATE TRIGGER reject_dirty BEFORE INSERT ON config WHEN NEW.key='vector_index_rebuild_status' AND NEW.value='dirty' BEGIN SELECT RAISE(ABORT,'synthetic-private-write-detail'); END"
    )
    with pytest.raises(HTTPException) as caught:
        await invoke(rig, operation)
    assert caught.value.status_code == 500 and isinstance(caught.value.__cause__, sqlite3.IntegrityError)
    assert "synthetic-private-write-detail" in str(caught.value.__cause__)
    assert "synthetic-private" not in str(caught.value.detail) + caplog.text
    assert "IntegrityError" in caplog.text
    assert snapshot(rig) == before
    assert (index_lifecycle._vector_index_revision is not None) and index_lifecycle._vector_index_revision == rig.revision


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
async def test_document_lookup_failure_preserves_original_cause_without_writing(
    documents, monkeypatch, caplog, operation
):
    rig = documents
    before = snapshot(rig)
    failure = OSError("synthetic-private-source-lookup")
    lookup = Mock(side_effect=failure)
    method = "get_knowledge_folder" if operation == "create" else "get_knowledge_document"
    with monkeypatch.context() as fault:
        fault.setattr(rig.db, method, lookup)
        with pytest.raises(HTTPException) as caught:
            await invoke(rig, operation)
    assert caught.value.status_code == 500 and caught.value.__cause__ is failure
    assert "synthetic-private" not in str(caught.value.detail) + caplog.text
    assert snapshot(rig) == before and (index_lifecycle._vector_index_revision is not None)
    lookup.assert_called_once()


@pytest.mark.parametrize("operation", ["update", "delete"])
async def test_missing_document_remains_a_business_404(documents, operation):
    before = snapshot(documents)
    with pytest.raises(HTTPException) as caught:
        if operation == "update":
            await knowledge.update_knowledge_document(999, KnowledgeDocumentUpdate(content=NEW), ADMIN)
        else:
            await knowledge.delete_knowledge_document(999, ADMIN)
    assert caught.value.status_code == 404 and caught.value.detail == "文档不存在"
    assert snapshot(documents) == before


async def test_invalid_creation_remains_a_business_422(documents):
    before = snapshot(documents)
    with pytest.raises(HTTPException) as caught:
        await knowledge.create_knowledge_document(KnowledgeDocumentCreate(title="", content=NEW), ADMIN)
    assert caught.value.status_code == 422
    assert snapshot(documents) == before
