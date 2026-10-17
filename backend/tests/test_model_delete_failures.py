"""Filesystem deletion errors are not unknown-model business results."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from api import models
from app.dependencies import get_current_admin
from app.providers import get_model_management_service
from services.model_management import ModelManagerService

MODEL = "qwen3-8b"
URL = "/api/models/" + MODEL


@pytest.fixture
def rig(monkeypatch, tmp_path):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("MODEL_PROVIDER", "mock")
    from inference.model_manager import MODEL_CONFIGS, ModelManager

    manager = ModelManager(base_dir=tmp_path)
    config = MODEL_CONFIGS[MODEL]
    target = manager.models_dir / config.name
    assert target.resolve().is_relative_to(tmp_path.resolve())
    target.mkdir()
    for required in config.required_files:
        (target / required).write_text("{}", encoding="utf-8")
    weights = target / "model.safetensors"
    weights.write_bytes(b"synthetic fixture, not actual weights")
    assert manager.check_model_exists(MODEL)
    service = ModelManagerService(manager)
    app = FastAPI()
    app.include_router(models.router)
    app.dependency_overrides[get_current_admin] = lambda: {"role": "admin"}
    app.dependency_overrides[get_model_management_service] = lambda: service
    try:
        yield SimpleNamespace(manager=manager, target=target, weights=weights, service=service, app=app)
    finally:
        manager.shutdown()


def request(rig, url=URL):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rig.app), base_url="http://test") as client:
            return await client.delete(url)

    return asyncio.run(run())


def test_successful_delete_and_absent_target_are_idempotent(rig):
    for _ in range(2):
        response = request(rig)
        assert response.status_code == 200
        assert response.json() == {"success": True, "message": "模型已删除"}
        assert not rig.target.exists()
        assert not rig.manager.check_model_exists(MODEL)


def test_unknown_model_is_400_without_touching_catalog_directory(rig, monkeypatch):
    delete = Mock(side_effect=AssertionError("unknown model reached filesystem deletion"))
    monkeypatch.setattr(shutil, "rmtree", delete)
    response = request(rig, "/api/models/synthetic-unknown-model")
    assert response.status_code == 400 and response.json() == {"detail": "删除模型失败"}
    delete.assert_not_called()
    assert rig.weights.exists()


@pytest.mark.parametrize(
    "stage,error_type", [("delete", PermissionError), ("delete", FileNotFoundError), ("inspect", PermissionError)]
)
def test_filesystem_failure_is_safe_503_without_retry(rig, monkeypatch, caplog, stage, error_type):
    error = error_type("synthetic-private-filesystem-detail")
    delete = Mock(side_effect=error)
    if stage == "delete":
        monkeypatch.setattr(shutil, "rmtree", delete)
    else:
        original = Path.lstat

        def inspect(path, *args, **kwargs):
            if path == rig.target:
                raise error
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "lstat", inspect)
        monkeypatch.setattr(shutil, "rmtree", Mock(side_effect=AssertionError("inspection failure reached deletion")))
    response = request(rig)
    assert response.status_code == 503
    assert "success" not in response.json()
    assert "可能已删除部分文件" in response.json()["detail"]
    assert "synthetic-private" not in response.text + caplog.text
    if stage == "delete":
        assert delete.call_count == 1
    assert rig.weights.exists()


def test_original_cause_survives_manager_thread_and_api(rig, monkeypatch):
    error = PermissionError("synthetic-private-original-detail")
    monkeypatch.setattr(shutil, "rmtree", Mock(side_effect=error))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(models.delete_model(MODEL, current_user={"role": "admin"}, service=rig.service))
    assert caught.value.status_code == 503 and caught.value.__cause__ is error


def test_partial_deletion_is_reported_and_later_explicit_attempt_can_complete(rig, monkeypatch, caplog):
    error = PermissionError("synthetic-private-partial-detail")

    def partial(path):
        assert Path(path) == rig.target
        rig.weights.unlink()
        raise error

    with monkeypatch.context() as context:
        context.setattr(shutil, "rmtree", partial)
        response = request(rig)
    assert response.status_code == 503 and "可能已删除部分文件" in response.json()["detail"]
    assert rig.target.is_dir() and not rig.weights.exists()
    assert list(rig.target.iterdir())
    assert "synthetic-private" not in response.text + caplog.text
    assert request(rig).status_code == 200 and not rig.target.exists()


def test_dangling_catalog_link_cannot_be_reported_as_absent_success(rig):
    shutil.rmtree(rig.target)
    missing = rig.target.parent / "synthetic-missing-link-target"
    rig.target.symlink_to(missing, target_is_directory=True)
    assert rig.target.is_symlink() and not rig.target.exists()
    response = request(rig)
    assert response.status_code == 503
    assert rig.target.is_symlink() and not missing.exists()


def test_cancelled_delete_is_not_converted_to_http_success_or_failure():
    service = SimpleNamespace(delete_model=AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(models.delete_model(MODEL, current_user={"role": "admin"}, service=service))


@pytest.mark.parametrize("status", [401, 403])
def test_admin_rejection_happens_before_delete_service(rig, monkeypatch, status):
    def deny():
        raise HTTPException(status_code=status, detail="denied")

    rig.app.dependency_overrides[get_current_admin] = deny
    delete = Mock(side_effect=AssertionError("unauthorized deletion"))
    monkeypatch.setattr(rig.manager, "delete_model", delete)
    assert request(rig).status_code == status
    delete.assert_not_called()
    assert rig.weights.exists()
