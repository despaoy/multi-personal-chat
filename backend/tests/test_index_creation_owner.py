"""Create/import publish source once; the shared search gate owns indexing."""

import io
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile

from api import knowledge
from app import config
from db.database import SQLiteDB
from db.schemas import KnowledgeDocumentCreate, KnowledgeSearchRequest
from knowledge import vector_db as vectors

TEXT = "2026年10月23日15:30在青岚楼二层办理登记。申请人须本人携带身份证和书面确认；没有收到确认不得付款。仅工作日受理，出差者可在下一工作日顺延；联系人为陈老师，截止时间为10月22日18:00。"


@pytest.fixture
def publication(tmp_path, monkeypatch):
    db = SQLiteDB(tmp_path / "source.sqlite")
    monkeypatch.setattr(knowledge, "db", db)
    monkeypatch.setattr(knowledge, "_vector_index_built", False)
    monkeypatch.setattr(knowledge, "_vector_index_revision", None)
    monkeypatch.setattr(knowledge, "VECTOR_DB_AVAILABLE", True)
    monkeypatch.setattr(vectors.VectorDatabase, "_check_gpu_availability", lambda self: False)
    vector = vectors.VectorDatabase(str(tmp_path / "vectors"))
    monkeypatch.setattr(vector, "_model", object())
    encode = Mock(side_effect=lambda texts: np.tile(np.eye(1, 384, dtype=np.float32), (len(texts), 1)))
    monkeypatch.setattr(vector, "_get_embeddings_batch", encode)
    monkeypatch.setattr(vector, "_get_embedding", lambda text: np.eye(1, 384, dtype=np.float32)[0])
    add = Mock(wraps=vector.add_documents)
    monkeypatch.setattr(vector, "add_documents", add)
    legacy = Mock(return_value=vector)
    monkeypatch.setattr(config, "get_vector_db", legacy)
    monkeypatch.setattr(vectors, "get_vector_db", lambda: vector)
    kb = db.create_knowledge_base("完整合成登记库")
    folder = db.create_knowledge_folder(kb["id"], "适用条件")
    assert knowledge._ensure_vector_index()
    revision = knowledge._get_rebuild_revision()
    lookup = Mock(wraps=db.get_knowledge_base)
    monkeypatch.setattr(db, "get_knowledge_base", lookup)
    root = tmp_path / "knowledge_bases"
    monkeypatch.setattr(knowledge, "KNOWLEDGE_BASES_DIR", root)
    return SimpleNamespace(
        db=db,
        vector=vector,
        encode=encode,
        add=add,
        legacy=legacy,
        kb=kb,
        folder=folder,
        revision=revision,
        lookup=lookup,
        root=root,
    )


async def publish(rig, route):
    admin = {"role": "admin"}
    if route == "create":
        return await knowledge.create_knowledge_document(
            KnowledgeDocumentCreate(
                title="登记规则",
                content=TEXT,
                category="旧分类",
                knowledge_base_id=rig.kb["id"],
                folder_id=rig.folder["id"],
            ),
            admin,
        )
    if route == "zip":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("适用条件/登记规则.txt", TEXT)
        buffer.seek(0)
        return await knowledge.upload_zip(rig.kb["id"], UploadFile(buffer, filename="完整登记规则.zip"), admin)
    target = rig.root / "合成来源"
    folder = target / "适用条件" if route == "scan_folder" else target
    folder.mkdir(parents=True)
    (folder / "登记规则.txt").write_text(TEXT, encoding="utf-8")
    return await knowledge.import_scanned_directory("合成来源", rig.kb["id"], admin)


@pytest.mark.parametrize("route", ["create", "zip", "scan_root", "scan_folder"])
async def test_create_and_import_use_one_authoritative_index_publication(publication, route):
    rig = publication
    result = await publish(rig, route)
    assert result["success"]
    docs = rig.db.get_knowledge_documents()
    assert len(docs) == 1 and docs[0]["content"] == TEXT
    doc = docs[0]
    expected_category = "未分类" if route == "scan_root" else "适用条件"
    assert doc["category"] == expected_category and doc["knowledge_base_id"] == rig.kb["id"]
    assert "".join(row["content"] for row in rig.db.get_knowledge_chunks(doc["id"])) == TEXT
    assert knowledge._get_rebuild_revision() == rig.revision + 1
    assert knowledge._read_rebuild_status()[0] == "dirty" and not knowledge._vector_index_built
    rig.add.assert_not_called()
    rig.encode.assert_not_called()
    rig.legacy.assert_not_called()
    if route == "create":
        rig.lookup.assert_not_called()
    assert knowledge._ensure_vector_index()
    rig.add.assert_called_once()
    rig.encode.assert_called_once()
    assert knowledge._read_rebuild_status()[0] == "complete"
    assert knowledge._ensure_vector_index()
    rig.add.assert_called_once()
    matches = rig.vector.search(
        "登记办理条件和未确认时付款限制", threshold=0.0, filters={"knowledge_base_id": rig.kb["id"]}
    )
    assert len(matches) == 1 and TEXT in matches[0]["content"]
    assert matches[0]["document_id"] == doc["id"] and matches[0]["category"] == expected_category
    assert rig.kb["name"] in matches[0]["content"] and "登记规则" in matches[0]["content"]
    reloaded = vectors.VectorDatabase(rig.vector.db_path)
    assert reloaded.metadata == rig.vector.metadata and reloaded.snapshot_validated
    assert reloaded.index.ntotal == len(reloaded.bm25.corpus) == len(reloaded.metadata) == 1


@pytest.mark.parametrize("route", ["create", "zip", "scan_root", "scan_folder"])
async def test_index_failure_keeps_complete_source_and_returns_search_failure(publication, monkeypatch, caplog, route):
    rig = publication
    result = await publish(rig, route)
    assert result["success"]
    rig.add.assert_not_called()
    failure = OSError("synthetic-private-index-flush-detail")
    monkeypatch.setattr(rig.vector, "flush", Mock(side_effect=failure))
    request = KnowledgeSearchRequest(query="登记办理的完整条件及未确认付款限制", knowledgeBaseName=rig.kb["name"])
    with pytest.raises(HTTPException) as raised:
        await knowledge.search_knowledge(request, {"role": "admin"})
    assert raised.value.status_code == 503
    assert "synthetic-private" not in str(raised.value.detail) + caplog.text
    assert knowledge._read_rebuild_status()[0] != "complete" and not knowledge._vector_index_built
    assert knowledge._get_rebuild_revision() == rig.revision + 1
    docs = rig.db.get_knowledge_documents()
    assert len(docs) == 1 and docs[0]["content"] == TEXT
    assert "".join(row["content"] for row in rig.db.get_knowledge_chunks(docs[0]["id"])) == TEXT
    rig.add.assert_called_once()
    rig.encode.assert_called_once()
    rig.legacy.assert_not_called()


@pytest.mark.parametrize("kind", ["base", "folder"])
def test_duplicate_parent_creation_ends_transaction_before_document_commit(publication, kind):
    rig = publication
    if kind == "base":
        assert rig.db.create_knowledge_base(rig.kb["name"]) is None
    else:
        assert rig.db.create_knowledge_folder(rig.kb["id"], rig.folder["name"]) is None
    document = rig.db.save_knowledge_document(
        {"title": "登记规则", "content": TEXT, "knowledge_base_id": rig.kb["id"], "folder_id": rig.folder["id"]},
        chunks=[TEXT],
    )
    assert document["content"] == TEXT
    assert rig.db.get_config_value(knowledge._VECTOR_REBUILD_REVISION_KEY) == rig.revision + 1
    assert len(rig.db.get_knowledge_bases()) == len(rig.db.get_knowledge_folders(rig.kb["id"])) == 1


async def test_scan_reuses_existing_named_base_before_atomic_document_publication(publication):
    rig = publication
    target = rig.root / rig.kb["name"]
    target.mkdir(parents=True)
    (target / "登记规则.txt").write_text(TEXT, encoding="utf-8")
    result = await knowledge.import_scanned_directory(rig.kb["name"], None, {"role": "admin"})
    assert result["success"] and result["createdDocs"] == 1 and result["errors"] == []
    documents = rig.db.get_knowledge_documents()
    assert len(documents) == 1 and documents[0]["content"] == TEXT
    assert documents[0]["knowledge_base_id"] == rig.kb["id"]
    assert knowledge._get_rebuild_revision() == rig.revision + 1
    assert len(rig.db.get_knowledge_bases()) == 1
    rig.add.assert_not_called()
    rig.legacy.assert_not_called()
    assert knowledge._ensure_vector_index()
    rig.add.assert_called_once()
