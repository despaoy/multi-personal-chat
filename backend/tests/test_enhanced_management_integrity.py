from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import gzip
import shutil
import sqlite3

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_managed_api_keys_are_authenticatable_listable_and_revocable(tmp_path) -> None:
    from db.database import SQLiteDB
    from infra.access_control import AccessControlManager, AuthenticationError, Role

    database = SQLiteDB(tmp_path / "access.db")
    manager = AccessControlManager(database)
    created = manager.create_api_key(Role.API_USER, "integration", 20)

    assert created["api_key"].startswith(created["key_prefix"])
    assert created["key_prefix"].count("_") >= 3
    rows = manager.list_api_keys()
    assert len(rows) == 1
    assert rows[0]["id"] > 0
    assert "api_key" not in rows[0]

    principal = await manager.authenticate(created["api_key"])
    assert principal["role"] is Role.API_USER
    assert manager.revoke_api_key_by_id(rows[0]["id"]) is True
    assert manager.revoke_api_key_by_id(rows[0]["id"]) is False
    with pytest.raises(AuthenticationError):
        await manager.authenticate(created["api_key"])


def test_managed_api_keys_flow_through_security_dependencies(tmp_path, monkeypatch) -> None:
    from app import config as app_config
    from app.dependencies import get_current_admin, get_current_user
    from infra.access_control import AccessControlManager, Role
    from middleware.security import SecurityMiddleware

    from db.database import SQLiteDB

    database = SQLiteDB(tmp_path / "middleware-access.db")
    manager = AccessControlManager(database)
    user_key = manager.create_api_key(Role.API_USER)["api_key"]
    admin_key = manager.create_api_key(Role.ADMIN)["api_key"]
    monkeypatch.setattr(app_config, "access_control_mgr", manager)

    application = FastAPI()
    application.add_middleware(SecurityMiddleware, api_keys=[])

    @application.get("/user")
    async def user_route(user: dict = Depends(get_current_user)):
        return user

    @application.get("/admin")
    async def admin_route(user: dict = Depends(get_current_admin)):
        return user

    with TestClient(application) as client:
        assert client.get("/user", headers={"X-API-Key": user_key}).status_code == 200
        assert client.get("/admin", headers={"X-API-Key": user_key}).status_code == 403
        response = client.get("/admin", headers={"X-API-Key": admin_key})
        assert response.status_code == 200
        assert response.json()["role"] == "admin"


@dataclass
class _BackupInfo:
    path: str = "/data/backups/example.bak.gz"

    def to_dict(self) -> dict:
        return {
            "filename": "example.bak.gz",
            "path": self.path,
            "backup_type": "full",
            "timestamp": datetime(2026, 1, 1).isoformat(),
            "size_bytes": 123,
            "sha256": "abc",
            "db_name": "example",
        }


@pytest.mark.asyncio
async def test_enhanced_backup_endpoints_use_the_real_manager_contract(monkeypatch) -> None:
    from api import enhanced
    from infra.backup_manager import BackupType

    class BackupManager:
        def __init__(self) -> None:
            self.received_type = None

        def list_backups(self):
            return [_BackupInfo()]

        async def backup(self, backup_type):
            self.received_type = backup_type
            return _BackupInfo()

    manager = BackupManager()
    monkeypatch.setattr(enhanced, "backup_mgr", lambda: manager)

    listed = await enhanced.list_backups({"role": "admin"})
    created = await enhanced.create_backup("full", {"role": "admin"})

    assert listed["backups"][0]["filename"] == "example.bak.gz"
    assert created["backup"]["backup_type"] == "full"
    assert manager.received_type is BackupType.FULL
    with pytest.raises(HTTPException) as exc_info:
        await enhanced.restore_backup("example.bak.gz", {"role": "admin"})
    assert exc_info.value.status_code == 409


def test_relative_sqlite_database_path_is_anchored_to_backend(monkeypatch) -> None:
    from db.database import BACKEND_DIR, _database_path_from_env

    monkeypatch.setenv("DATABASE_PATH", "runtime/local.db")

    assert _database_path_from_env() == BACKEND_DIR / "runtime/local.db"

def test_offline_sqlite_restore_validates_and_preserves_a_safety_copy(tmp_path) -> None:
    from scripts.restore_sqlite_backup import restore_sqlite_backup

    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    backup = tmp_path / "source_full.bak.gz"

    for path, value in ((source, "new"), (target, "old")):
        connection = sqlite3.connect(path)
        try:
            connection.execute("CREATE TABLE state (value TEXT NOT NULL)")
            connection.execute("INSERT INTO state VALUES (?)", (value,))
            connection.commit()
        finally:
            connection.close()

    with source.open("rb") as source_file, gzip.open(backup, "wb") as backup_file:
        shutil.copyfileobj(source_file, backup_file)

    safety_copy = restore_sqlite_backup(backup, target)

    assert safety_copy is not None and safety_copy.exists()
    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT value FROM state").fetchone()[0] == "new"
    with sqlite3.connect(safety_copy) as connection:
        assert connection.execute("SELECT value FROM state").fetchone()[0] == "old"

@pytest.mark.asyncio
async def test_enhanced_status_distinguishes_available_code_from_active_resources(monkeypatch) -> None:
    from api import enhanced

    async def no_vllm_stats():
        return None

    monkeypatch.setattr(enhanced, "BACKUP_MANAGER_AVAILABLE", True)
    monkeypatch.setattr(enhanced, "ACCESS_CONTROL_AVAILABLE", True)
    monkeypatch.setattr(enhanced, "_get_vllm_load_balancer_stats", no_vllm_stats)
    monkeypatch.setattr(enhanced, "backup_mgr", lambda: None)
    monkeypatch.setattr(enhanced, "access_control_mgr", lambda: None)
    monkeypatch.setattr(enhanced, "connection_pool", lambda: None)
    monkeypatch.setattr(enhanced, "http_client_pool", lambda: None)

    result = await enhanced.get_enhanced_status({"role": "admin"})

    assert result["availableFeatures"]["backupManager"] is True
    assert result["availableFeatures"]["accessControl"] is True
    assert result["enhancedFeatures"]["backupManager"] is False
    assert result["enhancedFeatures"]["accessControl"] is False
    assert result["enhancedFeatures"]["loadBalancer"] is False


@pytest.mark.asyncio
async def test_api_key_storage_failures_return_stable_503_responses(monkeypatch) -> None:
    from api import enhanced
    from db.schemas import ApiKeyCreateRequest

    class BrokenManager:
        def list_api_keys(self):
            raise RuntimeError("sqlite path=/private/access.db")

        def create_api_key(self, *args):
            raise RuntimeError("sqlite path=/private/access.db")

        def revoke_api_key_by_id(self, key_id):
            raise RuntimeError("sqlite path=/private/access.db")

    monkeypatch.setattr(enhanced, "access_control_mgr", lambda: BrokenManager())

    calls = (
        enhanced.list_api_keys({"role": "admin"}),
        enhanced.create_api_key(
            ApiKeyCreateRequest(role="api_user"),
            {"role": "admin"},
        ),
        enhanced.revoke_api_key(1, {"role": "admin"}),
    )
    for call in calls:
        with pytest.raises(HTTPException) as exc_info:
            await call
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == "API Key 存储暂时不可用"
        assert "private" not in exc_info.value.detail
