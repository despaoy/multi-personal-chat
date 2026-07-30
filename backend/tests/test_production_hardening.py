"""Regression tests for production hardening fixes."""

from __future__ import annotations

import ast
import io
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def test_database_url_selects_postgresql_when_flag_is_unset():
    from db.adapter import _should_use_postgresql

    assert _should_use_postgresql(
        {"DATABASE_URL": "postgresql+asyncpg://user:pass@db/app"}
    )
    assert _should_use_postgresql(
        {
            "USE_POSTGRESQL": "false",
            "DATABASE_URL": "postgresql+asyncpg://user:pass@db/app",
        }
    ) is False
    assert _should_use_postgresql({"DATABASE_URL": "sqlite:///local.db"}) is False


def test_production_validation_rejects_unsupported_database_and_workers():
    from infra.deployment import validate_deployment_environment

    base = {
        "ENVIRONMENT": "production",
        "ASTRBOT_INTEGRATION_TOKEN": "a" * 32,
        "QQCHAT_BACKEND_URL": "http://backend:8000",
        "VLLM_BASE_URL": "http://vllm:8001",
        "JWT_SECRET": "j" * 32,
        "ALLOWED_ORIGINS": "https://admin.example.com",
        "LOG_LEVEL": "INFO",
    }

    missing_url = validate_deployment_environment(
        {
            **base,
            "USE_POSTGRESQL": "true",
            "PG_HOST": "db",
            "PG_USER": "app",
            "PG_PASSWORD": "secret",
            "PG_DATABASE": "app",
        }
    )
    assert any("DATABASE_URL" in error for error in missing_url.errors)

    wrong_scheme = validate_deployment_environment(
        {**base, "DATABASE_URL": "sqlite:///app.db"}
    )
    assert any("PostgreSQL URL" in error for error in wrong_scheme.errors)

    multiple_workers = validate_deployment_environment(
        {
            **base,
            "DATABASE_URL": "postgresql://user:pass@db/app",
            "BACKEND_WORKERS": "2",
        }
    )
    assert any("BACKEND_WORKERS" in error for error in multiple_workers.errors)


def test_postgresql_sync_adapter_keeps_api_method_contracts():
    source = (BACKEND_ROOT / "db" / "pg_database.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    adapter = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SyncPgAdapter"
    )
    methods = {
        node.name: node
        for node in adapter.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    save_args = [arg.arg for arg in methods["save_claw_tool"].args.args]
    assert save_args == ["self", "name", "description", "code", "enabled"]
    assert methods["get_training_tasks"].args.args[1].arg == "status"
    update_args = [arg.arg for arg in methods["update_training_task"].args.args]
    assert update_args[:3] == ["self", "task_id", "data"]


def test_path_validation_rejects_prefix_collision():
    from api.training import _validate_path

    base = BACKEND_ROOT / "data"
    sibling = BACKEND_ROOT / "database-escape"
    with pytest.raises(ValueError):
        _validate_path(str(sibling), str(base))


def test_zip_validation_rejects_traversal_and_accepts_text():
    from api.knowledge import _validated_zip_entries

    unsafe_buffer = io.BytesIO()
    with zipfile.ZipFile(unsafe_buffer, "w") as archive:
        archive.writestr("../secret.txt", "secret")
    unsafe_buffer.seek(0)
    with zipfile.ZipFile(unsafe_buffer) as archive:
        with pytest.raises(HTTPException) as exc:
            _validated_zip_entries(archive)
    assert exc.value.status_code == 400

    safe_buffer = io.BytesIO()
    with zipfile.ZipFile(safe_buffer, "w") as archive:
        archive.writestr("character/profile.txt", "hello")
    safe_buffer.seek(0)
    with zipfile.ZipFile(safe_buffer) as archive:
        entries = _validated_zip_entries(archive)
    assert [entry.filename for entry in entries] == ["character/profile.txt"]


def test_claw_validator_blocks_dunder_escape_and_allows_basic_code():
    from api.claw import _validate_tool_code

    _validate_tool_code("return sum([1, 2, 3])")
    with pytest.raises(ValueError):
        _validate_tool_code("return ().__class__.__base__.__subclasses__()")
    with pytest.raises(ValueError):
        _validate_tool_code("import os\nreturn os.getcwd()")


@pytest.mark.asyncio
async def test_claw_execution_is_opt_in_in_production(monkeypatch):
    from api.claw import ToolExecuteRequest, execute_tool_code

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("CLAW_CODE_EXECUTION_ENABLED", raising=False)

    with pytest.raises(HTTPException) as exc:
        await execute_tool_code(ToolExecuteRequest(code="return 1"), {"user_id": 1})
    assert exc.value.status_code == 403


@pytest.mark.skipif(sys.platform == "win32", reason="Codex Windows sandbox blocks multiprocessing pipes")
def test_claw_executor_runs_in_child_and_enforces_timeout(monkeypatch):
    from api.claw import _run_in_sandbox_process

    monkeypatch.setenv("CLAW_EXECUTION_TIMEOUT", "5")
    success = _run_in_sandbox_process("return sum([1, 2, 3])", {})
    assert success["success"] is True
    assert success["result"] == "6"

    timed_out = _run_in_sandbox_process("while True:\n    pass", {})
    assert timed_out["success"] is False
    assert "timed out" in timed_out["error"] or "exited without a result" in timed_out["error"]


def test_production_jwt_secret_must_be_explicit_and_strong():
    from app.config import _validate_jwt_secret

    with pytest.raises(RuntimeError):
        _validate_jwt_secret("", "production")
    with pytest.raises(RuntimeError):
        _validate_jwt_secret("short", "production")
    assert _validate_jwt_secret("s" * 32, "production") == "s" * 32

def test_database_startup_probe_uses_adapter_contract():
    from app.main import _initialize_database

    class FakeDatabase:
        def __init__(self):
            self.initialized = False
            self.queries = []

        def init(self):
            self.initialized = True

        def execute_sql(self, query, params=None):
            self.queries.append(query)
            return [{"value": 1}]

        def get_connection(self):
            raise AssertionError("SQLite-only connection API must not be used")

    database = FakeDatabase()
    _initialize_database(database)
    assert database.initialized is True
    assert database.queries == ["SELECT 1"]

def test_lora_served_name_mapping_can_be_configured(monkeypatch):
    from inference.lora_utils import resolve_lora_served_name

    assert resolve_lora_served_name("hutao_lora_7b") == "hutao"
    monkeypatch.setenv("LORA_SERVED_NAME_MAP", '{"custom_lora": "custom-served"}')
    assert resolve_lora_served_name("custom_lora") == "custom-served"
    with pytest.raises(ValueError):
        resolve_lora_served_name("../escape")


@pytest.mark.asyncio
async def test_vllm_runtime_lora_load_uses_official_endpoint(monkeypatch):
    from inference.vllm_client import VLLMClient

    requests = []

    class FakeResponse:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self._payload = payload or {}
            self.text = ""

        def json(self):
            return self._payload

    class FakeClient:
        async def get(self, url, **kwargs):
            return FakeResponse(200, {"data": [{"id": "qwen3-8b-instruct-awq"}]})

        async def post(self, url, **kwargs):
            requests.append((url, kwargs.get("json")))
            return FakeResponse(200)

    client = VLLMClient(base_urls="http://vllm:8001", model="qwen3-8b-instruct-awq")

    async def fake_ensure_client():
        return FakeClient()

    monkeypatch.setattr(client, "_ensure_client", fake_ensure_client)
    await client.load_lora_adapter("minamo", "/loras/minamo/final")

    assert requests == [
        (
            "http://vllm:8001/v1/load_lora_adapter",
            {"lora_name": "minamo", "lora_path": "/loras/minamo/final"},
        )
    ]


@pytest.mark.asyncio
async def test_lora_status_is_not_changed_when_runtime_load_fails(monkeypatch):
    from api import loras

    class FakeDb:
        def __init__(self):
            self.updated = False

        def get_loras(self):
            return [{"id": "1", "name": "minamo_lora", "status": "inactive"}]

        def update_lora_status(self, lora_id, status):
            self.updated = True
            return {"id": lora_id, "name": "minamo_lora", "status": status}

    class FakeRequest:
        async def json(self):
            return {"status": "active"}

    class FailingClient:
        async def load_lora_adapter(self, name, path):
            raise RuntimeError("load failed")

    async def fake_get_client():
        return FailingClient()

    class FakeChecker:
        def __init__(self, **kwargs):
            pass

        def check_adapter(self, name):
            return type(
                "Report",
                (),
                {"compatible": True, "errors": [], "warnings": [], "base_model_mismatch": False},
            )()

    fake_db = FakeDb()
    monkeypatch.setattr(loras, "db", fake_db)
    monkeypatch.setattr(loras, "AdapterChecker", FakeChecker)
    monkeypatch.setattr(loras, "_resolve_vllm_adapter_path", lambda name: "/loras/minamo")
    import api.generate
    monkeypatch.setattr(api.generate, "get_vllm_client", fake_get_client)

    with pytest.raises(HTTPException) as exc:
        await loras.update_lora_status("1", FakeRequest(), {"user_id": 1})

    assert exc.value.status_code == 502
    assert fake_db.updated is False


@pytest.mark.asyncio
async def test_lora_switch_unloads_previous_adapter_before_loading_new(monkeypatch):
    from api import loras

    events = []

    class FakeDb:
        def get_loras(self):
            return [
                {"id": "1", "name": "old_lora", "status": "active"},
                {"id": "2", "name": "new_lora", "status": "inactive"},
            ]

        def update_lora_status(self, lora_id, status):
            events.append(("db", lora_id, status))
            return {"id": lora_id, "name": "new_lora", "status": status}

    class FakeRequest:
        async def json(self):
            return {"status": "active"}

    class FakeClient:
        async def unload_lora_adapter(self, name):
            events.append(("unload", name))

        async def load_lora_adapter(self, name, path):
            events.append(("load", name, path))

    class FakeChecker:
        def __init__(self, **kwargs):
            pass

        def check_adapter(self, name):
            return type(
                "Report",
                (),
                {"compatible": True, "errors": [], "warnings": [], "base_model_mismatch": False},
            )()

    async def fake_get_client():
        return FakeClient()

    monkeypatch.setattr(loras, "db", FakeDb())
    monkeypatch.setattr(loras, "AdapterChecker", FakeChecker)
    monkeypatch.setattr(loras, "_resolve_vllm_adapter_path", lambda name: f"/loras/{name}")
    import api.generate
    monkeypatch.setattr(api.generate, "get_vllm_client", fake_get_client)

    await loras.update_lora_status("2", FakeRequest(), {"user_id": 1})

    assert events == [
        ("unload", "old_lora"),
        ("load", "new_lora", "/loras/new_lora"),
        ("db", "2", "active"),
    ]


def test_postgresql_adapter_exposes_training_persistence_contract():
    source = (BACKEND_ROOT / "db" / "pg_database.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    for class_name in ("PgDatabase", "SyncPgAdapter"):
        methods = {
            node.name
            for node in classes[class_name].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert {
            "save_training_task",
            "get_training_task",
            "get_all_training_tasks",
            "get_active_training_by_lora_name",
        }.issubset(methods)


def test_all_db_backends_expose_iter_chunks_with_document_contract():
    """All DB backends (SQLiteDB, PgDatabase, SyncPgAdapter) must implement
    iter_chunks_with_document to support vector index rebuild without N+1
    queries. PgDatabase provides both async generator and a non-generator
    helper (get_chunks_with_document) used by the sync adapter.
    """
    source = (BACKEND_ROOT / "db" / "pg_database.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    pg_methods = {
        node.name
        for node in classes["PgDatabase"].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    sync_methods = {
        node.name
        for node in classes["SyncPgAdapter"].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    # PgDatabase: async generator + non-generator helper for SyncPgAdapter
    assert "iter_chunks_with_document" in pg_methods, "PgDatabase must implement iter_chunks_with_document"
    assert "get_chunks_with_document" in pg_methods, "PgDatabase must implement get_chunks_with_document helper"
    # SyncPgAdapter: sync generator
    assert "iter_chunks_with_document" in sync_methods, "SyncPgAdapter must implement iter_chunks_with_document"

    # SQLiteDB: sync generator (already exists, verify contract)
    sqlite_source = (BACKEND_ROOT / "db" / "database.py").read_text(encoding="utf-8")
    sqlite_tree = ast.parse(sqlite_source)
    sqlite_classes = {
        node.name: node for node in sqlite_tree.body if isinstance(node, ast.ClassDef)
    }
    sqlite_methods = {
        node.name
        for node in sqlite_classes["SQLiteDB"].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "iter_chunks_with_document" in sqlite_methods, "SQLiteDB must implement iter_chunks_with_document"


def test_production_registration_allows_only_bootstrap_user(monkeypatch):
    from api import auth

    class FakeDb:
        def __init__(self, count):
            self.count = count

        def execute_sql(self, query):
            return [{"count": self.count}]

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ALLOW_PUBLIC_REGISTRATION", "false")

    monkeypatch.setattr(auth, "db", FakeDb(0))
    assert auth._registration_allowed() is True

    monkeypatch.setattr(auth, "db", FakeDb(1))
    assert auth._registration_allowed() is False

    monkeypatch.setenv("ALLOW_PUBLIC_REGISTRATION", "true")
    assert auth._registration_allowed() is True


def test_forwarded_ip_requires_explicit_trusted_proxy(monkeypatch):
    from starlette.requests import Request
    from middleware import security

    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"x-forwarded-for", b"203.0.113.8")],
        "client": ("127.0.0.1", 12345),
        "server": ("localhost", 8000),
        "scheme": "http",
        "query_string": b"",
    })

    monkeypatch.setattr(security, "TRUST_PROXY_HEADERS", False)
    assert security._get_client_ip(request) == "127.0.0.1"
    monkeypatch.setattr(security, "TRUST_PROXY_HEADERS", True)
    assert security._get_client_ip(request) == "203.0.113.8"


@pytest.mark.asyncio
async def test_admin_status_requires_auth_and_cors_wraps_error():
    import httpx
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from middleware.security import SecurityMiddleware

    async def protected_status(request):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/api/stats", protected_status)])
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(
            "/api/stats",
            headers={"Origin": "http://localhost:5000"},
        )

    assert response.status_code == 401
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5000"



def test_training_resource_names_cannot_escape_output_directory():
    from api.training import _validate_resource_name

    assert _validate_resource_name("胡桃_lora-1") == "胡桃_lora-1"
    for unsafe in ("..", "../escape", "/tmp/escape", "folder\\escape", "bad:name"):
        with pytest.raises(HTTPException):
            _validate_resource_name(unsafe)


@pytest.mark.asyncio
async def test_lora_activation_returns_409_base_model_mismatch(monkeypatch):
    """基座不匹配应返回 409 LORA_BASE_MODEL_MISMATCH，且不调用 vLLM、不改动数据库。

    覆盖场景：胡桃/Minamo 的 Qwen2.5-7B LoRA 在 Qwen3-8B vLLM 上激活。
    预检阶段拦截，避免下游 vLLM 400 被包装成模糊的 502。
    """
    from api import loras

    class FakeDb:
        def __init__(self):
            self.updated = False

        def get_loras(self):
            return [{"id": "1", "name": "hutao_lora_7b", "status": "inactive"}]

        def update_lora_status(self, lora_id, status):
            self.updated = True
            return {"id": lora_id, "name": "hutao_lora_7b", "status": status}

    class FakeRequest:
        async def json(self):
            return {"status": "active"}

    class FakeClient:
        async def load_lora_adapter(self, name, path):
            raise AssertionError(
                "load_lora_adapter must not be called when base_model mismatches"
            )

        async def unload_lora_adapter(self, name):
            raise AssertionError(
                "unload_lora_adapter must not be called during pre-check failure"
            )

    class MismatchChecker:
        def __init__(self, **kwargs):
            pass

        def check_adapter(self, name):
            return type(
                "Report",
                (),
                {
                    "compatible": False,
                    "errors": ["base_model 不匹配: adapter=Qwen2.5-7B-Instruct, expected=Qwen3-8B-Instruct"],
                    "warnings": [],
                    "base_model_mismatch": True,
                    "expected_base_model": "/root/autodl-tmp/runtime/models/Qwen3-8B-Instruct",
                    "actual_base_model": "/root/hutao-training/models/Qwen2.5-7B-Instruct",
                },
            )()

    async def fake_get_client():
        return FakeClient()

    fake_db = FakeDb()
    monkeypatch.setattr(loras, "db", fake_db)
    monkeypatch.setattr(loras, "AdapterChecker", MismatchChecker)
    monkeypatch.setattr(loras, "_resolve_vllm_adapter_path", lambda name: f"/loras/{name}")
    import api.generate
    monkeypatch.setattr(api.generate, "get_vllm_client", fake_get_client)

    with pytest.raises(HTTPException) as exc:
        await loras.update_lora_status("1", FakeRequest(), {"user_id": 1})

    assert exc.value.status_code == 409
    detail = exc.value.detail
    assert detail["code"] == "LORA_BASE_MODEL_MISMATCH"
    assert "Qwen3-8B-Instruct" in detail["expected"]
    assert "Qwen2.5-7B-Instruct" in detail["actual"]
    # 预检失败不应改动数据库
    assert fake_db.updated is False


@pytest.mark.asyncio
async def test_lora_activation_returns_409_for_other_incompatibility(monkeypatch):
    """非基座兼容性错误（如缺权重文件）应返回普通 409，不带 LORA_BASE_MODEL_MISMATCH code。"""
    from api import loras

    class FakeDb:
        def __init__(self):
            self.updated = False

        def get_loras(self):
            return [{"id": "1", "name": "broken_lora", "status": "inactive"}]

        def update_lora_status(self, lora_id, status):
            self.updated = True
            return {"id": lora_id, "name": "broken_lora", "status": status}

    class FakeRequest:
        async def json(self):
            return {"status": "active"}

    class FakeClient:
        async def load_lora_adapter(self, name, path):
            raise AssertionError("load_lora_adapter must not be called on pre-check failure")

    class BrokenChecker:
        def __init__(self, **kwargs):
            pass

        def check_adapter(self, name):
            return type(
                "Report",
                (),
                {
                    "compatible": False,
                    "errors": ["adapter_model.safetensors / adapter_model.bin 不存在"],
                    "warnings": [],
                    "base_model_mismatch": False,
                    "expected_base_model": "",
                    "actual_base_model": "",
                },
            )()

    async def fake_get_client():
        return FakeClient()

    fake_db = FakeDb()
    monkeypatch.setattr(loras, "db", fake_db)
    monkeypatch.setattr(loras, "AdapterChecker", BrokenChecker)
    monkeypatch.setattr(loras, "_resolve_vllm_adapter_path", lambda name: f"/loras/{name}")
    import api.generate
    monkeypatch.setattr(api.generate, "get_vllm_client", fake_get_client)

    with pytest.raises(HTTPException) as exc:
        await loras.update_lora_status("1", FakeRequest(), {"user_id": 1})

    assert exc.value.status_code == 409
    detail = exc.value.detail
    # 非 base_model 错误不应携带 LORA_BASE_MODEL_MISMATCH code
    assert "code" not in detail or detail.get("code") != "LORA_BASE_MODEL_MISMATCH"
    assert fake_db.updated is False


@pytest.mark.asyncio
async def test_lora_activation_409_base_model_mismatch_via_testclient(monkeypatch):
    """真实 HTTP 链路验证：409 LORA_BASE_MODEL_MISMATCH 经过 FastAPI 序列化后
    响应体结构正确，且 admin 依赖被正确触发。

    覆盖此前直接调用路由函数时未经过的链路：认证依赖、JSON 序列化、
    HTTPException 到 HTTP 响应的转换。
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.dependencies import get_current_admin
    from api import loras

    class FakeDb:
        def __init__(self):
            self.updated = False

        def get_loras(self):
            return [{"id": "1", "name": "hutao_lora_7b", "status": "inactive"}]

        def update_lora_status(self, lora_id, status):
            self.updated = True
            return {"id": lora_id, "name": "hutao_lora_7b", "status": status}

    class MismatchChecker:
        def __init__(self, **kwargs):
            pass

        def check_adapter(self, name):
            return type(
                "Report",
                (),
                {
                    "compatible": False,
                    "errors": ["base_model 不匹配"],
                    "warnings": [],
                    "base_model_mismatch": True,
                    "expected_base_model": "/models/Qwen3-8B-Instruct",
                    "actual_base_model": "/models/Qwen2.5-7B-Instruct",
                },
            )()

    fake_db = FakeDb()
    monkeypatch.setattr(loras, "db", fake_db)
    monkeypatch.setattr(loras, "AdapterChecker", MismatchChecker)
    monkeypatch.setattr(loras, "_resolve_vllm_adapter_path", lambda name: f"/loras/{name}")

    app = FastAPI()
    app.include_router(loras.router)
    # 绕过 admin 认证依赖
    app.dependency_overrides[get_current_admin] = lambda: {"user_id": 1, "username": "tester"}

    client = TestClient(app)
    try:
        response = client.put(
            "/api/loras/1/status",
            json={"status": "active"},
        )
    finally:
        try:
            client.close()
        except Exception:
            pass

    assert response.status_code == 409
    body = response.json()
    detail = body["detail"]
    assert detail["code"] == "LORA_BASE_MODEL_MISMATCH"
    assert "Qwen3-8B-Instruct" in detail["expected"]
    assert "Qwen2.5-7B-Instruct" in detail["actual"]
    # 预检失败不应改动数据库
    assert fake_db.updated is False
