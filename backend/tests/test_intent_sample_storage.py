"""Publish complete sample files and keep invalid or failed edits explicit."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from api import knowledge
from app.dependencies import get_current_admin
from db.schemas import IntentSamplesSaveRequest
from knowledge import intent_trainer as trainer

URL = "/api/knowledge/train-intent/samples"
ORIGINAL = {"合成角色": ["已经审查的完整合成问题"]}


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setattr(trainer, "SAMPLES_DIR", tmp_path)
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(ORIGINAL, ensure_ascii=False), encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)
    return path


def unchanged(store):
    assert json.loads(store.read_text(encoding="utf-8")) == ORIGINAL
    assert set(store.parent.iterdir()) == {store}


def app_for_admin():
    app = FastAPI()
    app.include_router(knowledge.router)
    app.dependency_overrides[get_current_admin] = lambda: {"role": "admin"}
    return app


@pytest.mark.parametrize("exists", [True, False])
def test_save_closes_complete_utf8_candidate_before_one_atomic_publication(store, monkeypatch, exists):
    if not exists:
        store.unlink()
    payload = {"合成角色": ["完整的新问题", "保留否定：她没有红色物品"], "none": []}
    replace = trainer.os.replace
    observations = []

    def publish(candidate, target):
        candidate, target = Path(candidate), Path(target)
        assert target == store
        assert candidate.parent.parent == store.parent
        if exists:
            assert json.loads(target.read_text(encoding="utf-8")) == ORIGINAL
        else:
            assert not target.exists()
        assert candidate.read_text(encoding="utf-8") == json.dumps(payload, ensure_ascii=False, indent=2)
        observations.append(candidate)
        replace(candidate, target)

    monkeypatch.setattr(trainer.os, "replace", publish)
    assert trainer.save_samples(payload) == {"success": True, "stats": {"合成角色": 2, "none": 0}}
    assert len(observations) == 1
    assert trainer.get_samples()["samples"] == payload
    assert set(store.parent.iterdir()) == {store}
    if os.name == "posix":
        assert store.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("payload", [{"合成角色": [object()]}, {"合成角色": None}])
def test_serialization_and_statistics_fail_before_touching_old_samples(store, payload):
    with pytest.raises(TypeError):
        trainer.save_samples(payload)
    unchanged(store)


@pytest.mark.parametrize("error_type", [OSError, asyncio.CancelledError])
def test_interrupted_candidate_write_keeps_old_file_and_cleans_staging(store, monkeypatch, error_type):
    original_write = Path.write_text

    def fail_write(path, text, *args, **kwargs):
        if path.parent.name.startswith(".samples-"):
            original_write(path, "incomplete synthetic JSON", encoding="utf-8")
            raise error_type("synthetic-private-write-detail")
        return original_write(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(error_type):
        trainer.save_samples({"合成角色": ["新的完整合成问题"]})
    unchanged(store)


@pytest.mark.parametrize("exists", [True, False])
def test_publication_failure_never_deletes_or_creates_target(store, monkeypatch, exists):
    if not exists:
        store.unlink()
    error = PermissionError("synthetic-private-rename-detail")
    replace = Mock(side_effect=error)
    monkeypatch.setattr(trainer.os, "replace", replace)
    with pytest.raises(PermissionError) as caught:
        trainer.save_samples({"合成角色": ["新的完整合成问题"]})
    assert caught.value is error
    assert replace.call_count == 1
    if exists:
        unchanged(store)
    else:
        assert not store.exists() and list(store.parent.iterdir()) == []


def test_staging_creation_failure_preserves_reviewed_samples(store, monkeypatch):
    error = PermissionError("synthetic-private-directory-detail")
    monkeypatch.setattr(trainer, "TemporaryDirectory", Mock(side_effect=error))
    with pytest.raises(PermissionError) as caught:
        trainer.save_samples({"合成角色": ["新的完整合成问题"]})
    assert caught.value is error
    unchanged(store)


def test_frontend_shaped_crud_requests_and_explicit_empty_save(store):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_for_admin()), base_url="http://test"
        ) as client:
            assert (await client.get(URL)).json()["samples"] == ORIGINAL
            assert (await client.post(URL, json={"label": "合成角色", "text": "新增完整问题"})).json() == {
                "success": True
            }
            assert (
                await client.put(URL, json={"label": "合成角色", "index": 0, "text": "修改后的完整问题"})
            ).json() == {"success": True}
            assert (await client.delete(URL, params={"label": "合成角色", "index": 1})).json() == {"success": True}
            assert (await client.get(URL)).json()["samples"] == {"合成角色": ["修改后的完整问题"]}
            payload = {"合成装备": ["装备的完整合成问题"], "none": []}
            response = await client.patch(URL, json={"samples": payload})
            assert response.status_code == 200 and response.json() == {
                "success": True,
                "stats": {"合成装备": 1, "none": 0},
            }
            assert (await client.get(URL)).json()["samples"] == payload
            response = await client.patch(URL, json={"samples": {}})
            assert response.status_code == 200 and response.json() == {"success": True, "stats": {}}

    asyncio.run(scenario())
    assert json.loads(store.read_text(encoding="utf-8")) == {}
    assert set(store.parent.iterdir()) == {store}


@pytest.mark.parametrize(
    "method,payload",
    [
        ("PATCH", {}),
        ("PATCH", {"samples": None}),
        ("PATCH", {"samples": {"合成角色": "不是数组"}}),
        ("PATCH", {"samples": {"合成角色": [None]}}),
        ("PATCH", {"samples": {"合成角色": [123]}}),
        ("POST", {"label": "合成角色"}),
        ("POST", {"text": "完整问题"}),
        ("PUT", {"label": "合成角色", "text": "完整问题"}),
        ("PUT", {"label": "合成角色", "index": -1, "text": "完整问题"}),
    ],
)
def test_invalid_request_cannot_clear_or_corrupt_samples(store, method, payload, monkeypatch):
    save = Mock(side_effect=AssertionError("invalid input reached persistence"))
    monkeypatch.setattr(trainer, "save_samples", save)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_for_admin()), base_url="http://test"
        ) as client:
            response = await client.request(method, URL, json=payload)
            assert response.status_code == 422

    asyncio.run(scenario())
    save.assert_not_called()
    unchanged(store)


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE", "PATCH"])
def test_storage_error_is_safe_503_for_each_sample_endpoint(store, method, monkeypatch, caplog):
    error = OSError("synthetic-private-storage-detail")
    if method == "GET":
        monkeypatch.setattr(trainer, "get_samples", Mock(side_effect=error))
    else:
        monkeypatch.setattr(trainer.os, "replace", Mock(side_effect=error))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_for_admin()), base_url="http://test"
        ) as client:
            kwargs = {"params": {"label": "合成角色", "index": 0}} if method == "DELETE" else {}
            if method in {"POST", "PUT"}:
                kwargs["json"] = {"label": "合成角色", "text": "新的完整合成问题", "index": 0}
            elif method == "PATCH":
                kwargs["json"] = {"samples": {"合成角色": ["新的完整合成问题"]}}
            response = await client.request(method, URL, **kwargs)
            assert response.status_code == 503
            assert "success" not in response.json()
            assert "synthetic-private" not in response.text
            assert "存储空间" in response.json()["detail"]

    asyncio.run(scenario())
    assert "synthetic-private" not in caplog.text
    unchanged(store)


def test_missing_sample_file_is_legitimate_empty_result(store):
    store.unlink()
    assert trainer.get_samples() == {"samples": {}, "stats": {}}


def test_corrupt_existing_json_is_not_reported_as_empty_samples(store):
    store.write_text("incomplete synthetic JSON", encoding="utf-8")

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_for_admin()), base_url="http://test"
        ) as client:
            response = await client.get(URL)
            assert response.status_code == 503
            assert "samples" not in response.json()

    asyncio.run(scenario())


def test_api_failure_retains_original_cause(store, monkeypatch):
    error = PermissionError("synthetic-private-detail")
    monkeypatch.setattr(trainer.os, "replace", Mock(side_effect=error))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            knowledge.batch_save_intent_samples(
                IntentSamplesSaveRequest(samples={"合成角色": ["新的问题"]}), current_user={"role": "admin"}
            )
        )
    assert caught.value.status_code == 503
    assert caught.value.__cause__ is error
    unchanged(store)


@pytest.mark.parametrize("status", [401, 403])
def test_all_sample_routes_still_require_admin_before_storage_access(store, monkeypatch, status):
    def deny():
        raise HTTPException(status_code=status, detail="denied")

    app = app_for_admin()
    app.dependency_overrides[get_current_admin] = deny
    get = Mock(side_effect=AssertionError("unauthorized storage read"))
    save = Mock(side_effect=AssertionError("unauthorized storage write"))
    monkeypatch.setattr(trainer, "get_samples", get)
    monkeypatch.setattr(trainer, "save_samples", save)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                response = await client.request(
                    method,
                    URL,
                    params={"label": "合成角色", "index": 0},
                    json={"samples": {}, "label": "合成角色", "text": "完整问题", "index": 0},
                )
                assert response.status_code == status

    asyncio.run(scenario())
    get.assert_not_called()
    save.assert_not_called()
    unchanged(store)


def test_missing_sample_business_error_stays_400(store):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_for_admin()), base_url="http://test"
        ) as client:
            response = await client.put(URL, json={"label": "不存在", "index": 0, "text": "完整问题"})
            assert response.status_code == 400

    asyncio.run(scenario())
    unchanged(store)
