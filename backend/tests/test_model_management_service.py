from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.models import router
from app.dependencies import get_current_admin, get_current_user
from app.providers import get_model_management_service
from services.model_management import ModelManagerService


class FakeModelManager:
    def __init__(self) -> None:
        self.models = [{"name": "qwen3-8b", "downloaded": False}]
        self.calls: list[tuple[str, object]] = []

    def list_available_models(self):
        self.calls.append(("list", None))
        return self.models

    def check_model_exists(self, model_name: str) -> bool:
        self.calls.append(("check", model_name))
        return model_name == "installed"

    def download_model_from_hf(self, model_name: str, force: bool = False):
        self.calls.append(("download", (model_name, force)))
        return {"success": True, "model_name": model_name, "force": force}

    def delete_model(self, model_name: str) -> bool:
        self.calls.append(("delete", model_name))
        return model_name == "installed"


async def test_model_manager_service_maps_all_operations() -> None:
    manager = FakeModelManager()
    service = ModelManagerService(manager)

    assert await service.list_available_models() == manager.models
    assert await service.check_model_exists("installed") is True
    assert await service.download_model("candidate", force=True) == {
        "success": True,
        "model_name": "candidate",
        "force": True,
    }
    assert await service.delete_model("installed") is True
    assert manager.calls == [
        ("list", None),
        ("check", "installed"),
        ("download", ("candidate", True)),
        ("delete", "installed"),
    ]


def _build_client(service) -> TestClient:
    application = FastAPI()
    application.include_router(router)
    application.dependency_overrides[get_current_user] = lambda: {
        "user_id": 1,
        "username": "tester",
    }
    application.dependency_overrides[get_current_admin] = lambda: {
        "user_id": 1,
        "username": "admin",
        "role": "admin",
    }
    application.dependency_overrides[get_model_management_service] = lambda: service
    return TestClient(application)


def test_model_routes_use_injected_service_and_preserve_responses() -> None:
    manager = FakeModelManager()
    service = ModelManagerService(manager)

    with _build_client(service) as client:
        listed = client.get("/api/models")
        checked = client.get("/api/models/check/installed")
        downloaded = client.post(
            "/api/models/download",
            json={"model_name": "candidate", "force": True},
        )
        missing_delete = client.delete("/api/models/missing")
        deleted = client.delete("/api/models/installed")
        downloaded_7b = client.post("/api/models/check-7b")

    assert listed.json() == {"success": True, "models": manager.models}
    assert checked.json() == {
        "success": True,
        "model_name": "installed",
        "downloaded": True,
    }
    assert downloaded.json() == {
        "success": True,
        "model_name": "candidate",
        "force": True,
    }
    assert missing_delete.status_code == 400
    assert missing_delete.json() == {"detail": "删除模型失败"}
    assert deleted.json() == {"success": True, "message": "模型已删除"}
    assert downloaded_7b.json() == {
        "success": True,
        "model_name": "qwen3-8b",
        "force": False,
    }


class FailingModelService:
    async def list_available_models(self):
        raise RuntimeError("C:/private/model/token")


def test_model_routes_do_not_expose_internal_errors() -> None:
    with _build_client(FailingModelService()) as client:
        response = client.get("/api/models")

    assert response.status_code == 500
    assert response.json() == {"detail": "列出模型失败"}
    assert "private" not in response.text

class RejectedDownloadService:
    def __init__(self, error: str) -> None:
        self.error = error

    async def download_model(self, model_name: str, *, force: bool = False):
        return {"success": False, "error": self.error}


def test_model_download_route_maps_manager_failures_to_stable_http_errors() -> None:
    with _build_client(RejectedDownloadService("network token=/private")) as client:
        upstream_failure = client.post(
            "/api/models/download",
            json={"model_name": "qwen3-8b", "force": False},
        )
    with _build_client(RejectedDownloadService("未知模型: missing")) as client:
        unknown_model = client.post(
            "/api/models/download",
            json={"model_name": "missing", "force": False},
        )

    assert upstream_failure.status_code == 502
    assert upstream_failure.json() == {"detail": "模型下载失败"}
    assert "private" not in upstream_failure.text
    assert unknown_model.status_code == 400


def _build_real_manager(monkeypatch, base_dir: Path):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("MODEL_PROVIDER", "mock")
    from inference.model_manager import ModelManager

    return ModelManager(base_dir=base_dir)


def test_model_manager_resumes_partial_download_and_uses_catalog_root(tmp_path, monkeypatch) -> None:
    manager = _build_real_manager(monkeypatch, tmp_path)
    try:
        from inference.model_manager import MODEL_CONFIGS

        config = MODEL_CONFIGS["qwen3-8b"]
        partial_dir = manager.models_dir / config.name
        partial_dir.mkdir(parents=True)
        (partial_dir / "config.json").write_text("{}", encoding="utf-8")
        calls = []

        def snapshot_download(*, repo_id: str, local_dir: str, resume_download: bool):
            calls.append((repo_id, local_dir, resume_download))
            target = Path(local_dir)
            for required in config.required_files:
                (target / required).write_text("{}", encoding="utf-8")
            (target / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            return local_dir

        monkeypatch.setitem(
            sys.modules,
            "huggingface_hub",
            SimpleNamespace(snapshot_download=snapshot_download),
        )
        result = manager.download_model_from_hf("qwen3-8b")

        assert manager.models_dir == tmp_path / "models"
        assert result["success"] is True
        assert len(calls) == 1
        assert manager.check_model_exists("qwen3-8b") is True
    finally:
        manager.shutdown()


def test_model_manager_serializes_model_directory_mutations(tmp_path, monkeypatch) -> None:
    manager = _build_real_manager(monkeypatch, tmp_path)
    active = 0
    peak_active = 0
    counter_lock = threading.Lock()

    def fake_download(model_name: str, force: bool = False):
        nonlocal active, peak_active
        with counter_lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.03)
        with counter_lock:
            active -= 1
        return {"success": True, "model_name": model_name}

    manager._download_model_from_hf_locked = fake_download
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(manager.download_model_from_hf, ["qwen3-8b", "qwen3-8b"]))
        assert all(item["success"] for item in results)
        assert peak_active == 1
    finally:
        manager.shutdown()
