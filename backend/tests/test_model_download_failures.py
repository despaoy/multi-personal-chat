"""A known model's failed download cannot become an unknown-model result."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from api import models
from app.dependencies import get_current_admin
from app.providers import get_model_management_service
from db.schemas import ModelDownloadRequest
from services.model_management import ModelDownloadError, ModelManagerService

MODEL = "qwen3-8b"
ENDPOINTS = ["/api/models/download", "/api/models/check-7b"]


@pytest.fixture
def rig(monkeypatch, tmp_path):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("MODEL_PROVIDER", "mock")
    from inference.model_manager import MODEL_CONFIGS, ModelManager

    manager = ModelManager(base_dir=tmp_path)
    config = MODEL_CONFIGS[MODEL]
    target = manager.models_dir / config.name
    assert target.resolve().is_relative_to(tmp_path.resolve())

    def complete_snapshot(*, repo_id, local_dir, resume_download):
        assert repo_id == config.repo_id and resume_download is True
        assert Path(local_dir) == target
        target.mkdir(parents=True, exist_ok=True)
        for name in config.required_files:
            (target / name).write_text("{}", encoding="utf-8")
        (target / "model.safetensors").write_bytes(b"synthetic fixture, not actual weights")
        return local_dir

    snapshot = Mock(side_effect=complete_snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    service = ModelManagerService(manager)
    app = FastAPI()
    app.include_router(models.router)
    app.dependency_overrides[get_current_admin] = lambda: {"role": "admin"}
    app.dependency_overrides[get_model_management_service] = lambda: service
    try:
        yield SimpleNamespace(
            manager=manager,
            config=config,
            target=target,
            complete=complete_snapshot,
            snapshot=snapshot,
            service=service,
            app=app,
        )
    finally:
        manager.shutdown()


def request(rig, endpoint, name=MODEL):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rig.app), base_url="http://test") as client:
            return await client.post(endpoint, json={"model_name": name, "force": False})

    return asyncio.run(run())


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_complete_download_and_existing_files_keep_success_shape(rig, endpoint):
    response = request(rig, endpoint)
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message": f"模型下载完成: {rig.config.name}",
        "model_name": MODEL,
        "path": str(rig.target),
    }
    assert rig.manager.check_model_exists(MODEL) and rig.snapshot.call_count == 1
    assert request(rig, endpoint).json()["success"] is True
    assert rig.snapshot.call_count == 1


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_known_model_upstream_text_cannot_claim_unknown_model(rig, endpoint, caplog):
    error = OSError("未知模型: synthetic-private-upstream-detail")
    rig.snapshot.side_effect = error
    response = request(rig, endpoint)
    assert response.status_code == 502
    assert "success" not in response.json()
    assert "可能已保存部分文件" in response.json()["detail"]
    assert "OSError" in response.json()["detail"]
    assert "synthetic-private" not in response.text + caplog.text
    assert rig.snapshot.call_count == 1
    assert not rig.manager.check_model_exists(MODEL)


def test_unknown_catalog_name_keeps_400_and_never_starts_download(rig):
    response = request(rig, ENDPOINTS[0], name="synthetic-not-in-catalog")
    assert response.status_code == 400 and response.json() == {"detail": "模型下载失败"}
    rig.snapshot.assert_not_called()
    assert not rig.target.exists()


def test_partial_files_are_not_success_and_later_explicit_attempt_can_finish(rig):
    def partial_snapshot(**kwargs):
        rig.target.mkdir()
        (rig.target / rig.config.required_files[0]).write_text("{}", encoding="utf-8")
        raise OSError("synthetic-private-partial-detail")

    rig.snapshot.side_effect = partial_snapshot
    first = request(rig, ENDPOINTS[0])
    assert first.status_code == 502 and "可能已保存部分文件" in first.json()["detail"]
    assert rig.target.is_dir() and not rig.manager.check_model_exists(MODEL)
    assert rig.snapshot.call_count == 1
    assert not next(row for row in rig.manager.list_available_models() if row["name"] == MODEL)["downloaded"]
    rig.snapshot.side_effect = rig.complete
    assert request(rig, ENDPOINTS[0]).status_code == 200
    assert rig.snapshot.call_count == 2 and rig.manager.check_model_exists(MODEL)


def test_original_download_cause_survives_domain_thread_and_http_boundaries(rig):
    error = OSError("synthetic-private-original-detail")
    rig.snapshot.side_effect = error
    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            models.download_model(
                ModelDownloadRequest(model_name=MODEL), current_user={"role": "admin"}, service=rig.service
            )
        )
    assert caught.value.status_code == 502
    assert isinstance(caught.value.__cause__, ModelDownloadError)
    assert caught.value.__cause__.__cause__ is error
    assert "synthetic-private" not in str(caught.value.detail)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("result", [None, {"success": False, "error": "未知模型: synthetic-private-legacy-detail"}])
def test_invalid_service_result_is_failure_without_error_text_classification(
    rig, endpoint, result, monkeypatch, caplog
):
    monkeypatch.setattr(rig.service, "download_model", AsyncMock(return_value=result))
    response = request(rig, endpoint)
    assert response.status_code == 502 and response.json() == {"detail": "模型下载失败"}
    assert "synthetic-private" not in response.text + caplog.text
    rig.snapshot.assert_not_called()
    assert not rig.target.exists()


def test_pre_download_directory_check_failure_is_safe_503(rig, monkeypatch, caplog):
    error = PermissionError("synthetic-private-inspection-detail")
    monkeypatch.setattr(rig.service, "check_model_exists", AsyncMock(side_effect=error))
    response = request(rig, ENDPOINTS[1])
    assert response.status_code == 503
    assert "模型检查未完成" in response.json()["detail"]
    assert "synthetic-private" not in response.text + caplog.text
    rig.snapshot.assert_not_called()


def test_cancelled_download_is_not_a_fake_result():
    service = SimpleNamespace(download_model=AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            models.download_model(
                ModelDownloadRequest(model_name=MODEL), current_user={"role": "admin"}, service=service
            )
        )


@pytest.mark.parametrize("status", [401, 403])
def test_both_download_routes_require_admin_before_any_model_work(rig, monkeypatch, status):
    def deny():
        raise HTTPException(status_code=status, detail="denied")

    rig.app.dependency_overrides[get_current_admin] = deny
    inspect = Mock(side_effect=AssertionError("unauthorized model inspection"))
    download = Mock(side_effect=AssertionError("unauthorized download"))
    monkeypatch.setattr(rig.manager, "check_model_exists", inspect)
    monkeypatch.setattr(rig.manager, "download_model_from_hf", download)
    for endpoint in ENDPOINTS:
        assert request(rig, endpoint).status_code == status
    inspect.assert_not_called()
    download.assert_not_called()
    rig.snapshot.assert_not_called()
