"""Only configured knowledge sources may be scanned or imported."""

from knowledge import index_lifecycle

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from api import knowledge
from app.dependencies import get_current_admin
from app.runtime import RuntimeContainer
from infra.bounded_executor import BoundedThreadExecutor
from db.database import SQLiteDB

TEXT = "完整登记规则：12月5日14点在二楼办理，本人携身份证和书面确认，未确认不得付款；截止为4日18点。"
OUTSIDE = "完整外部合成资料：12月8日15点到三楼，须携证件并先获得书面确认；这不属于配置的知识来源。"


@pytest.fixture
def sources(tmp_path, monkeypatch):
    root = tmp_path / "knowledge"
    target = root / "source"
    target.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "external.txt").write_text(OUTSIDE, encoding="utf-8")
    db = SQLiteDB(tmp_path / "source.sqlite")
    kb = db.create_knowledge_base("合成目标库")
    monkeypatch.setattr(knowledge, "db", db)
    monkeypatch.setattr(index_lifecycle, "db", db)
    monkeypatch.setattr(knowledge, "KNOWLEDGE_BASES_DIR", root)
    monkeypatch.setattr(index_lifecycle, "_vector_index_revision", None)
    app = FastAPI()
    executor = BoundedThreadExecutor(name="source-boundary-scan", max_workers=1, max_pending=2, default_timeout=3)
    app.state.runtime_container = RuntimeContainer(db=db, is_pg_mode=lambda: False, knowledge_scan_executor=executor)
    app.include_router(knowledge.router)
    app.dependency_overrides[get_current_admin] = lambda: {"role": "admin"}
    try:
        yield SimpleNamespace(root=root, target=target, outside=outside, db=db, kb=kb, app=app)
    finally:
        executor.shutdown()


async def request(rig, *, name=None):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rig.app), base_url="http://test") as client:
        if name is None:
            return await client.get("/api/knowledge/scan")
        return await client.post("/api/knowledge/scan/import", params={"directory_name": name, "kb_id": rig.kb["id"]})


@pytest.mark.parametrize("kind", ["parent", "absolute", "directory_link"])
async def test_import_rejects_external_target_before_database_writes(sources, kind):
    rig = sources
    if kind == "parent":
        name = "../outside"
    elif kind == "absolute":
        name = str(rig.outside)
    else:
        (rig.root / "external-link").symlink_to(rig.outside, target_is_directory=True)
        name = "external-link"
    before = rig.db.config
    response = await request(rig, name=name)
    assert response.status_code == 400 and "超出" in response.json()["detail"]
    assert str(rig.outside) not in response.text and OUTSIDE not in response.text
    assert rig.db.get_knowledge_documents() == [] and rig.db.get_knowledge_folders(rig.kb["id"]) == []
    assert rig.db.config == before


@pytest.mark.parametrize("kind", ["root_file", "folder_file", "folder_directory"])
@pytest.mark.parametrize("has_good_source", [False, True])
async def test_external_entries_are_failed_items_without_hiding_saved_source(sources, kind, has_good_source):
    rig = sources
    if kind == "root_file":
        (rig.target / "external.txt").symlink_to(rig.outside / "external.txt")
    elif kind == "folder_file":
        folder = rig.target / "rules"
        folder.mkdir()
        (folder / "external.txt").symlink_to(rig.outside / "external.txt")
    else:
        (rig.target / "external-folder").symlink_to(rig.outside, target_is_directory=True)
    if has_good_source:
        (rig.target / "good.txt").write_text(TEXT, encoding="utf-8")
    response = await request(rig, name="source")
    result = response.json()
    assert response.status_code == 200 and result["success"] is False
    assert result["status"] == ("partial" if has_good_source else "failed")
    assert result["createdDocs"] == int(has_good_source) and len(result["errors"]) == 1
    assert OUTSIDE not in response.text and str(rig.outside) not in response.text
    docs = rig.db.get_knowledge_documents()
    assert len(docs) == int(has_good_source)
    if has_good_source:
        assert docs[0]["content"] == TEXT and docs[0]["knowledge_base_id"] == rig.kb["id"]
        assert "".join(row["content"] for row in rig.db.get_knowledge_chunks(docs[0]["id"])) == TEXT
    if kind == "folder_directory":
        assert rig.db.get_knowledge_folders(rig.kb["id"]) == []


@pytest.mark.parametrize("kind", ["file", "directory"])
async def test_scan_refuses_external_links_without_returning_external_tree(sources, kind):
    rig = sources
    if kind == "file":
        (rig.target / "external.txt").symlink_to(rig.outside / "external.txt")
    else:
        (rig.target / "external-folder").symlink_to(rig.outside, target_is_directory=True)
    response = await request(rig)
    assert response.status_code == 400 and "超出" in response.json()["detail"]
    assert str(rig.outside) not in response.text and OUTSIDE not in response.text
    assert rig.db.get_knowledge_documents() == []


@pytest.mark.parametrize("kind", ["self", "ancestor"])
async def test_scan_reports_link_cycle_as_failure(sources, kind):
    target = sources.target if kind == "self" else sources.root
    (sources.target / "cycle").symlink_to(target, target_is_directory=True)
    response = await request(sources)
    assert response.status_code == 400 and "循环" in response.json()["detail"]
    assert sources.db.get_knowledge_documents() == []


@pytest.mark.parametrize("kind", ["file", "directory"])
async def test_internal_links_keep_alias_names_complete_text_and_ownership(sources, kind):
    rig = sources
    if kind == "file":
        (rig.root / "original.txt").write_text(TEXT, encoding="utf-8")
        (rig.target / "alias.txt").symlink_to(rig.root / "original.txt")
        name, title = "source", "alias"
    else:
        (rig.target / "rules.txt").write_text(TEXT, encoding="utf-8")
        (rig.root / "alias").symlink_to(rig.target, target_is_directory=True)
        name, title = "alias", "rules"
    scan = await request(rig)
    assert scan.status_code == 200 and scan.json()["success"]
    selected = next(row for row in scan.json()["directories"] if row["name"] == name)
    assert selected["fileCount"] == 1
    result = (await request(rig, name=name)).json()
    assert result["success"] and result["status"] == "complete" and result["createdDocs"] == 1
    doc = rig.db.get_knowledge_documents()[0]
    assert doc["title"] == title and doc["content"] == TEXT and doc["knowledge_base_id"] == rig.kb["id"]
    assert "".join(row["content"] for row in rig.db.get_knowledge_chunks(doc["id"])) == TEXT


async def test_missing_optional_source_root_stays_an_explicit_empty_scan(sources, monkeypatch):
    monkeypatch.setattr(knowledge, "KNOWLEDGE_BASES_DIR", sources.root / "absent")
    response = await request(sources)
    assert response.status_code == 200
    assert response.json() == {"success": True, "directories": [], "message": "知识库目录不存在"}
