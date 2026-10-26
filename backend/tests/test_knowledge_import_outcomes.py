"""Real batch imports report confirmed work, errors and unknown writes honestly."""

from app.runtime import RuntimeContainer

from knowledge.index_lifecycle import default_knowledge_index as index_lifecycle

import asyncio
import io
import threading
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI

from api import knowledge
from app.dependencies import get_current_admin
from db.database import SQLiteDB
from infra.bounded_executor import BoundedThreadExecutor

TEXT = "2026年10月23日15:30在青岚楼二层办理登记。申请人须本人携带身份证和书面确认；未收到确认不得付款。仅工作日受理，出差者可顺延一个工作日；联系人为陈老师，截止时间为10月22日18:00。"
ROUTES = ["zip", "scan_root", "scan_folder"]


@pytest.fixture
def imports(tmp_path, monkeypatch):
    db = SQLiteDB(tmp_path / 'source.sqlite')
    kb = db.create_knowledge_base('完整合成登记库')
    db.create_knowledge_folder(kb['id'], '适用条件')
    monkeypatch.setattr(index_lifecycle, 'db', db)
    monkeypatch.setattr(index_lifecycle.db, '_knowledge_index', index_lifecycle, raising=False)
    root = tmp_path / 'knowledge_bases'
    monkeypatch.setattr(knowledge, 'KNOWLEDGE_BASES_DIR', root)
    opened = []
    original_zip = zipfile.ZipFile

    class TrackedZip(original_zip):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)
    monkeypatch.setattr(zipfile, 'ZipFile', TrackedZip)
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(db=index_lifecycle.db, is_pg_mode=lambda: False, startup_env={})
    app.include_router(knowledge.router)
    app.dependency_overrides[get_current_admin] = lambda: {'role': 'admin'}
    return SimpleNamespace(db=db, kb=kb, root=root, app=app, opened=opened)


async def publish(rig, route, entries, *, use_existing_base=True, directory_name="合成来源", text_encoding="unicode"):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rig.app), base_url="http://test") as client:
        if route == "zip":
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                for name, content in entries:
                    archive.writestr("适用条件/" + name, content)
            response = await client.post(
                f"/api/knowledge/bases/{rig.kb['id']}/upload-zip",
                files={"file": ("完整登记规则.zip", buffer.getvalue(), "application/zip")},
                params={"text_encoding": text_encoding},
            )
        else:
            target = rig.root / directory_name
            folder = target / "适用条件" if route == "scan_folder" else target
            folder.mkdir(parents=True, exist_ok=True)
            for name, content in entries:
                (folder / name).write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
            params = {"directory_name": directory_name, "text_encoding": text_encoding}
            if use_existing_base:
                params["kb_id"] = rig.kb["id"]
            response = await client.post("/api/knowledge/scan/import", params=params)
    assert all(archive.fp is None for archive in rig.opened)
    return response


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("outcome", ["complete", "partial", "failed", "empty"])
async def test_actual_api_reports_batch_outcomes_and_keeps_committed_source(imports, route, outcome, caplog):
    rig = imports
    rig.db.execute_sql("""CREATE TRIGGER reject_import BEFORE INSERT ON knowledge_documents
        WHEN NEW.title LIKE '02-broken%' BEGIN SELECT RAISE(ABORT,'synthetic-private-import-failure'); END""")
    entries = {
        "complete": [("01-good.txt", TEXT), ("03-good.txt", TEXT)],
        "partial": [("01-good.txt", TEXT), ("02-broken.txt", TEXT)],
        "failed": [("02-broken-a.txt", TEXT), ("02-broken-b.txt", TEXT)],
        "empty": [("empty.txt", " \n ")],
    }[outcome]
    response = await publish(rig, route, entries)
    assert response.status_code == 200
    result = response.json()
    expected_count = {"complete": 2, "partial": 1, "failed": 0, "empty": 0}[outcome]
    assert result["status"] == ("complete" if outcome == "empty" else outcome)
    assert result["success"] is (outcome in {"complete", "empty"})
    assert result["createdDocs"] == expected_count
    assert len(result["errors"]) == {"complete": 0, "partial": 1, "failed": 2, "empty": 0}[outcome]
    assert "synthetic-private" not in response.text + caplog.text
    documents = rig.db.get_knowledge_documents()
    assert len(documents) == expected_count
    for document in documents:
        assert document["content"] == TEXT and document["knowledge_base_id"] == rig.kb["id"]
        assert "".join(row["content"] for row in rig.db.get_knowledge_chunks(document["id"])) == TEXT
    if outcome == "partial":
        assert "部分" in result["message"] and "1" in result["message"]
    elif outcome == "failed":
        assert "失败" in result["message"]
    elif outcome == "empty":
        assert "未找到" in result["message"]


@pytest.mark.parametrize("route", ROUTES)
async def test_unsupported_encoding_is_a_failed_item_including_root_files(imports, route):
    response = await publish(imports, route, [("encoding.txt", b"\x81")])
    result = response.json()
    assert result["status"] == "failed" and result["success"] is False
    assert result["createdDocs"] == 0 and len(result["errors"]) == 1
    assert "编码" in result["errors"][0]
    assert imports.db.get_knowledge_documents() == []


@pytest.mark.parametrize("route", ROUTES)
async def test_post_commit_notification_failure_still_counts_saved_documents(imports, monkeypatch, caplog, route):
    monkeypatch.setattr(
        index_lifecycle,
        "_invalidate_local_knowledge_index",
        Mock(side_effect=OSError("synthetic-private-notification-detail")),
    )
    response = await publish(imports, route, [("01-good.txt", TEXT), ("03-good.txt", TEXT)])
    result = response.json()
    assert result["status"] == "partial" and result["success"] is False
    assert result["createdDocs"] == len(imports.db.get_knowledge_documents()) == 2
    assert len(result["errors"]) == 2
    assert "synthetic-private" not in response.text + caplog.text


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("operation", ["write", "notification"])
async def test_real_executor_timeout_stops_batch_and_preserves_late_commit(imports, monkeypatch, route, operation):
    rig = imports
    release = threading.Event()
    entered = threading.Event()
    executor = BoundedThreadExecutor(name="controlled-import", max_workers=1, max_pending=2, default_timeout=0.1)
    target = rig.db.save_knowledge_document if operation == "write" else index_lifecycle._invalidate_local_knowledge_index

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return target(*args, **kwargs)

    if operation == "write":
        monkeypatch.setattr(rig.db, "save_knowledge_document", delayed)
    else:
        monkeypatch.setattr(index_lifecycle, "_invalidate_local_knowledge_index", delayed)

    async def run_db(function, *args, **kwargs):
        return await executor.run(function, *args, **kwargs)

    monkeypatch.setattr(knowledge, "run_db", run_db)
    try:
        response = await publish(rig, route, [("01-good.txt", TEXT), ("03-not-attempted.txt", TEXT)])
        assert entered.is_set()
        result = response.json()
        assert result["status"] == "unconfirmed" and result["success"] is False
        assert result["createdDocs"] == (0 if operation == "write" else 1)
        assert "未确认" in result["message"] and "核对" in result["message"]
    finally:
        release.set()
        await asyncio.to_thread(executor.shutdown, wait=True)
    documents = rig.db.get_knowledge_documents()
    assert len(documents) == 1 and documents[0]["title"] == "01-good"
    assert documents[0]["content"] == TEXT


async def test_timeout_while_creating_base_reports_unknown_without_inventing_identity(imports, monkeypatch):
    rig = imports
    release = threading.Event()
    executor = BoundedThreadExecutor(name="controlled-import-base", max_workers=1, max_pending=2, default_timeout=0.1)
    create_base = rig.db.create_knowledge_base

    def delayed(*args, **kwargs):
        assert release.wait(3)
        return create_base(*args, **kwargs)

    monkeypatch.setattr(rig.db, "create_knowledge_base", delayed)

    async def run_db(function, *args, **kwargs):
        return await executor.run(function, *args, **kwargs)

    monkeypatch.setattr(knowledge, "run_db", run_db)
    try:
        response = await publish(
            rig, "scan_root", [("01-good.txt", TEXT)], use_existing_base=False, directory_name="新合成来源"
        )
        result = response.json()
        assert result["status"] == "unconfirmed" and result["createdDocs"] == 0
        assert result["success"] is False and "knowledgeBase" not in result
    finally:
        release.set()
        await asyncio.to_thread(executor.shutdown, wait=True)
    assert rig.db.get_knowledge_base_by_name("新合成来源") is not None
    assert rig.db.get_knowledge_documents() == []


async def test_valid_double_dot_filename_uses_canonical_zip_validation(imports):
    response = await publish(imports, "zip", [("登记..v2.txt", TEXT)])
    result = response.json()
    assert result["status"] == "complete" and result["createdDocs"] == 1
    assert imports.db.get_knowledge_documents()[0]["title"] == "登记..v2"


async def test_zip_traversal_is_rejected_before_writes_and_archive_is_closed(imports):
    response = await publish(imports, "zip", [("../escape.txt", TEXT)])
    assert response.status_code == 400
    assert imports.db.get_knowledge_documents() == []



@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-32", "gbk"])
async def test_import_preserves_declared_source_text_and_raw_size(imports, route, encoding):
    source = "完整规则：借期14天，不得自动续借。\r\n周末不受理；只有本人携带书面确认才可申请。\r\n"
    raw = source.encode(encoding)
    selected = "gbk" if encoding == "gbk" else "unicode"
    result = (await publish(imports, route, [("完整规则.txt", raw)], text_encoding=selected)).json()
    assert result["success"] and result["status"] == "complete"
    row, = imports.db.get_knowledge_documents()
    assert row["content"] == source and row["fileSize"] == len(raw)


@pytest.mark.parametrize("route", ROUTES)
async def test_unlabelled_legacy_text_requires_explicit_choice_and_can_recover(imports, route):
    source = "完整规则：仅本人可申请，不得自动续借。"
    raw = source.encode("gbk")
    failed = (await publish(imports, route, [("完整规则.txt", raw)])).json()
    assert failed["status"] == "failed" and not failed["success"] and failed["createdDocs"] == 0
    assert imports.db.get_knowledge_documents() == [] and "文本格式" in failed["errors"][0]
    complete = (await publish(imports, route, [("完整规则.txt", raw)], text_encoding="gbk")).json()
    assert complete["success"] and complete["createdDocs"] == 1
    assert imports.db.get_knowledge_documents()[0]["content"] == source


@pytest.mark.parametrize("route", ROUTES)
async def test_invalid_encoding_choice_is_rejected_before_document_writes(imports, route):
    response = await publish(imports, route, [("完整规则.txt", TEXT)], text_encoding="unsupported")
    assert response.status_code == 422 and imports.db.get_knowledge_documents() == []


@pytest.mark.parametrize("route", ROUTES)
async def test_declared_bom_takes_priority_and_corrupt_bytes_report_partial(imports, route):
    source = "完整规则：仅本人可申请，不得自动续借。\r\n"
    response = await publish(imports, route, [("01-marked.txt", source.encode("utf-8-sig")),
                                              ("02-legacy.txt", source.encode("gbk")), ("03-corrupt.txt", b"\x81")],
                             text_encoding="gbk")
    result = response.json()
    assert result["status"] == "partial" and not result["success"] and result["createdDocs"] == 2
    assert [row["content"] for row in imports.db.get_knowledge_documents()] == [source, source]
    assert len(result["errors"]) == 1 and "文本格式" in result["errors"][0]
