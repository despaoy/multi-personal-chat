"""Declared duplicates are distinct from real knowledge storage failures."""

from knowledge import index_lifecycle

import asyncio
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from api import knowledge
from app.dependencies import get_current_admin
from db.database import SQLiteDB


@pytest.fixture
def creation(tmp_path, monkeypatch):
    database = SQLiteDB(tmp_path / "creation.sqlite")
    owner = database.create_knowledge_base("完整合成知识库")
    monkeypatch.setattr(knowledge, "db", database)
    monkeypatch.setattr(index_lifecycle, "db", database)
    app = FastAPI()
    app.include_router(knowledge.router)
    app.dependency_overrides[get_current_admin] = lambda: {"role": "admin"}
    return SimpleNamespace(db=database, owner=owner, app=app)


@pytest.mark.parametrize("kind", ["base", "folder"])
async def test_valid_request_storage_failure_is_not_reported_as_duplicate_and_recovers(creation, kind):
    rig = creation
    table = "knowledge_bases" if kind == "base" else "knowledge_folders"
    url = "/api/knowledge/bases" if kind == "base" else f"/api/knowledge/bases/{rig.owner['id']}/folders"
    rig.db.execute_sql(f"CREATE TRIGGER reject_complete_request BEFORE INSERT ON {table} WHEN NEW.name='完整故障请求' BEGIN SELECT RAISE(ABORT,'synthetic-private duplicate storage failure'); END")
    invoke = rig.db.create_knowledge_base if kind == "base" else lambda name: rig.db.create_knowledge_folder(rig.owner["id"], name)
    with pytest.raises(sqlite3.IntegrityError, match="synthetic-private"):
        invoke("完整故障请求")
    assert not rig.db._get_connection().in_transaction
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rig.app, raise_app_exceptions=False), base_url="http://test") as client:
        failed = await client.post(url, json={"name": "完整故障请求", "description": "请求完整，数据库拒绝写入"})
        assert failed.status_code == 500 and "synthetic-private" not in failed.text
        assert rig.db.execute_sql(f"SELECT COUNT(*) AS count FROM {table} WHERE name='完整故障请求'")[0]["count"] == 0
        recovered = await client.post(url, json={"name": "完整恢复请求", "description": "保留完整描述"})
        assert recovered.status_code == 200 and recovered.json()["success"]
        duplicate = await client.post(url, json={"name": "完整恢复请求", "description": "不得覆盖"})
        assert duplicate.status_code == 409
    get = rig.db.get_knowledge_base if kind == "base" else rig.db.get_knowledge_folder
    key = "base" if kind == "base" else "folder"
    assert get(recovered.json()[key]["id"])["description"] == "保留完整描述"
    assert not rig.db._get_connection().in_transaction


@pytest.mark.parametrize("kind", ["base", "folder"])
async def test_concurrent_duplicate_creates_exactly_once(creation, kind):
    rig = creation
    invoke = rig.db.create_knowledge_base if kind == "base" else lambda name: rig.db.create_knowledge_folder(rig.owner["id"], name)
    results = await asyncio.gather(asyncio.to_thread(invoke, "并发完整请求"), asyncio.to_thread(invoke, "并发完整请求"))
    assert sum(row is not None for row in results) == 1
    table = "knowledge_bases" if kind == "base" else "knowledge_folders"
    assert rig.db.execute_sql(f"SELECT COUNT(*) AS count FROM {table} WHERE name='并发完整请求'")[0]["count"] == 1


def test_folder_name_conflict_is_scoped_to_its_base(creation):
    rig = creation
    other = rig.db.create_knowledge_base("第二完整知识库")
    one = rig.db.create_knowledge_folder(rig.owner["id"], "相同目录名", "第一组描述")
    two = rig.db.create_knowledge_folder(other["id"], "相同目录名", "第二组描述")
    assert one is not None and two is not None and one["id"] != two["id"]
    assert rig.db.create_knowledge_folder(rig.owner["id"], "相同目录名", "不得覆盖") is None
    assert rig.db.get_knowledge_folder(one["id"])["description"] == "第一组描述"
