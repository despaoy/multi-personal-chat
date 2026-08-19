"""Migration behavior regression tests.

Covers the critical fixes that AST-based tests cannot verify:
- M1: session.begin_nested() isolates single-row failures (异常传播后由业务捕获)
- M1: _migrate_table 返回 (inserted, failed)，migrate() 据此非零退出
- M4: task_id → id 是复制语义（两列都有值，通过 execute 参数断言）
- C1: _sync_pg_sequences 返回 (synced, failed) 且使用 pg_get_serial_sequence
- M-1: _validate_table_name 拒绝非法标识符
- m1: migrate() 非零退出是行为测试（mock _sync_pg_sequences 返回失败）

These tests use mock pg_db / session objects — no real PostgreSQL required.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# m1 fix: 确保 db.pg_database 模块加载时 DATABASE_URL 可用，
# 避免 PgDatabase() 单例初始化抛出 ValueError。
# 测试通过 monkeypatch 替换 pg_db，不会真实连接数据库。
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")

# 如果 asyncpg 未安装，插入 mock 的 db.pg_database 模块，避免导入失败
import types as _types
_mock_pg_module = _types.ModuleType("db.pg_database")
_mock_pg_module.pg_db = MagicMock()
_mock_pg_module.pg_db.init = AsyncMock()
_mock_pg_module.pg_db.close = AsyncMock()
# 提供 migrate() 内部导入所需的表对象（用 MagicMock 代替）
for _tbl_name in [
    "config_table", "users_table", "loras_table", "messages_table",
    "knowledge_bases_table", "knowledge_folders_table",
    "knowledge_documents_table", "knowledge_chunks_table",
    "user_data_table", "saved_dialogues_table",
    "api_keys_table", "claw_tools_table",
    "audit_logs_table", "intent_samples_table",
    "intent_active_kbs_table", "training_tasks_table",
    "integration_message_dedup_table", "conversations_table",
    "integration_events_table", "model_invocations_table",
    "gold_eval_runs_table", "experiment_runs_table",
    "retrieval_eval_questions_table", "preference_pairs_table",
    "adapter_compatibility_table", "feedback_table",
]:
    setattr(_mock_pg_module, _tbl_name, MagicMock())

# 尝试导入真实 asyncpg；如果可用，让真实模块覆盖 mock；否则用 mock
try:
    import asyncpg  # noqa: F401
    # asyncpg 可用，尝试导入真实 db.pg_database
    try:
        import db.pg_database  # noqa: F401
    except Exception:
        # 真实导入失败（如 DATABASE_URL 问题），回退到 mock
        sys.modules["db.pg_database"] = _mock_pg_module
except ImportError:
    # asyncpg 不可用，直接用 mock
    sys.modules["db.pg_database"] = _mock_pg_module

from db.migration import _migrate_table, _sync_pg_sequences, _validate_table_name


# ============================================
# Mock helpers
# ============================================

class FakeColumn:
    """模拟 SQLAlchemy Column"""
    def __init__(self, name: str):
        self.name = name


class FakeTable:
    """模拟 SQLAlchemy Table"""
    def __init__(self, column_names: list[str]):
        self.columns = [FakeColumn(n) for n in column_names]


class FakeSavepointCtx:
    """模拟 session.begin_nested() 返回的 async context manager。

    M2 fix: 真实 SQLAlchemy begin_nested() 在块内抛异常时会回滚 savepoint，
    然后继续向外传播异常（由业务代码的 try/except 捕获）。
    因此 __aexit__ 返回 False（不吞异常），让异常传播到外层 except。
    """
    def __init__(self, session: "FakeSession"):
        self._session = session

    async def __aenter__(self):
        self._session.begin_nested_called = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # M2 fix: 返回 False 表示不吞异常，让异常传播到业务 try/except
        # 真实 SQLAlchemy 行为：savepoint 回滚后异常继续传播
        if exc_type is not None:
            self._session.last_row_failed = True
        return False


class FakeSession:
    """模拟 SQLAlchemy AsyncSession

    M2 fix: execute() 记录所有调用的 (stmt, params) 以便测试断言参数。
    """

    def __init__(self, execute_side_effects=None):
        self.begin_nested_called = False
        self.last_row_failed = False
        self._execute_side_effects = execute_side_effects or []
        self._execute_call_index = 0
        self.committed = False
        self.rolled_back = False
        self._active = True
        self.execute_calls: list[tuple] = []  # 记录 (stmt, params)

    def begin_nested(self):
        return FakeSavepointCtx(self)

    async def execute(self, stmt, params=None):
        self.execute_calls.append((stmt, params))
        if self._execute_call_index < len(self._execute_side_effects):
            effect = self._execute_side_effects[self._execute_call_index]
            self._execute_call_index += 1
            if isinstance(effect, Exception):
                raise effect
            result = MagicMock()
            result.rowcount = effect
            return result
        result = MagicMock()
        result.rowcount = 1
        return result

    async def commit(self):
        self.committed = True

    async def rollback(self):
        self.rolled_back = True
        self._active = False

    @property
    def is_active(self):
        return self._active

    def in_transaction(self):
        return self._active


class FakePgDb:
    """模拟 PgDatabase，提供 async_session() 上下文管理器"""

    def __init__(self, session: FakeSession):
        self._session = session

    def async_session(self):
        outer = self

        class _Ctx:
            async def __aenter__(self):
                return outer._session

            async def __aexit__(self, *args):
                pass

        return _Ctx()


# ============================================
# M-1: _validate_table_name
# ============================================

def test_validate_table_name_accepts_valid():
    assert _validate_table_name("training_tasks") == "training_tasks"
    assert _validate_table_name("config") == "config"
    assert _validate_table_name("_underscore_start") == "_underscore_start"


def test_validate_table_name_rejects_injection():
    with pytest.raises(ValueError):
        _validate_table_name("users; DROP TABLE users;--")
    with pytest.raises(ValueError):
        _validate_table_name("table`name")
    with pytest.raises(ValueError):
        _validate_table_name("name WITH space")
    with pytest.raises(ValueError):
        _validate_table_name("")


# ============================================
# M1: begin_nested isolates single-row failures + returns (inserted, failed)
# ============================================

@pytest.mark.asyncio
async def test_migrate_table_uses_begin_nested_for_isolation():
    """单行失败不应污染事务，后续行仍可插入；返回值包含 failed 计数。"""
    rows = [
        {"id": 1, "name": "row1"},
        {"id": 2, "name": "row2"},
        {"id": 3, "name": "row3"},
    ]
    table = FakeTable(["id", "name"])
    # 第 2 行 execute 抛异常，会被 begin_nested 块内的 try/except 捕获
    session = FakeSession(execute_side_effects=[1, RuntimeError("duplicate key"), 1])
    pg_db = FakePgDb(session)

    inserted, failed = await _migrate_table(pg_db, "test_table", table, rows, conflict_columns=["id"])

    # begin_nested 被调用，第 2 行失败被外层 except 捕获，第 1、3 行成功
    assert session.begin_nested_called is True
    assert inserted == 2
    assert failed == 1
    assert session.committed is True


@pytest.mark.asyncio
async def test_migrate_table_empty_rows_returns_zero_zero():
    """空表跳过迁移，返回 (0, 0)。"""
    table = FakeTable(["id", "name"])
    session = FakeSession()
    pg_db = FakePgDb(session)

    inserted, failed = await _migrate_table(pg_db, "test_table", table, [], conflict_columns=["id"])
    assert inserted == 0
    assert failed == 0



@pytest.mark.asyncio
async def test_migrate_table_invalid_conflict_column_returns_failed_count():
    rows = [{"id": 1}, {"id": 2}]
    table = FakeTable(["id"])
    session = FakeSession()
    inserted, failed = await _migrate_table(
        FakePgDb(session), "test_table", table, rows, conflict_columns=["missing"]
    )
    assert (inserted, failed) == (0, 2)
    assert session.execute_calls == []


@pytest.mark.asyncio
async def test_migrate_table_row_without_target_columns_is_failure():
    table = FakeTable(["id"])
    inserted, failed = await _migrate_table(
        FakePgDb(FakeSession()), "test_table", table, [{"legacy_only": "value"}]
    )
    assert (inserted, failed) == (0, 1)

# ============================================
# M4: task_id → id copy semantics（通过 execute 参数断言）
# ============================================

@pytest.mark.asyncio
async def test_migrate_table_task_id_copied_to_both_columns():
    """training_tasks 的 task_id 应同时写入 id 和 task_id 两列。

    M2 fix: 通过检查 execute() 收到的 params 断言两列都有值，
    而非仅断言 begin_nested_called。
    """
    rows = [{"task_id": "task-001", "lora_name": "test_lora", "status": "pending"}]
    table = FakeTable(["id", "task_id", "lora_name", "status"])
    session = FakeSession(execute_side_effects=[1])
    pg_db = FakePgDb(session)

    inserted, failed = await _migrate_table(
        pg_db, "training_tasks", table, rows,
        conflict_columns=["id"],
        column_mapping={"task_id": "id"},
    )

    assert inserted == 1
    assert failed == 0
    # 找到 INSERT 调用的 params（begin_nested 块内的 execute）
    # execute_calls[0] 是第一次 execute 调用，params 应包含 id 和 task_id
    assert len(session.execute_calls) >= 1
    _, params = session.execute_calls[0]
    assert params is not None
    # M4 核心断言：id 和 task_id 都应有值（复制语义）
    assert params.get("id") == "task-001", f"id 列应有值 task-001，实际: {params.get('id')}"
    assert params.get("task_id") == "task-001", f"task_id 列应有值 task-001，实际: {params.get('task_id')}"


@pytest.mark.asyncio
async def test_migrate_table_no_mapping_preserves_original_column_only():
    """无 column_mapping 时，原始列名原样写入，不复制。"""
    rows = [{"task_id": "task-001", "status": "pending"}]
    table = FakeTable(["task_id", "status"])
    session = FakeSession(execute_side_effects=[1])
    pg_db = FakePgDb(session)

    inserted, failed = await _migrate_table(pg_db, "test_table", table, rows, conflict_columns=["task_id"])

    assert inserted == 1
    _, params = session.execute_calls[0]
    assert params.get("task_id") == "task-001"
    # 无映射时不应出现 id 列
    assert "id" not in params


# ============================================
# C1 + M1: _sync_pg_sequences returns (synced, failed)
# ============================================

@pytest.mark.asyncio
async def test_sync_pg_sequences_returns_synced_failed_counts():
    """_sync_pg_sequences 应返回 (synced, failed) 元组。"""
    seq_result = MagicMock()
    seq_result.fetchall.return_value = [
        ("public", "users_id_seq", "users", "id"),
        ("public", "messages_id_seq", "messages", "id"),
    ]
    session = FakeSession(execute_side_effects=[None, None])
    call_idx = [0]
    async def fake_execute(stmt, params=None):
        if call_idx[0] == 0:
            call_idx[0] += 1
            return seq_result
        call_idx[0] += 1
        return MagicMock()
    session.execute = fake_execute
    pg_db = FakePgDb(session)

    synced, failed = await _sync_pg_sequences(pg_db)
    assert synced == 2
    assert failed == 0


@pytest.mark.asyncio
async def test_sync_pg_sequences_counts_failures():
    """序列同步失败时 failed > 0。"""
    seq_result = MagicMock()
    seq_result.fetchall.return_value = [
        ("public", "users_id_seq", "users", "id"),
    ]
    session = FakeSession()
    call_idx = [0]
    async def fake_execute(stmt, params=None):
        if call_idx[0] == 0:
            call_idx[0] += 1
            return seq_result
        call_idx[0] += 1
        raise RuntimeError("permission denied")
    session.execute = fake_execute
    pg_db = FakePgDb(session)

    synced, failed = await _sync_pg_sequences(pg_db)
    assert synced == 0
    assert failed == 1


# ============================================
# C1: uses pg_get_serial_sequence (not pg_get_serial_identifier)
# ============================================

def test_sync_pg_sequences_uses_correct_function():
    """源码中应使用 pg_get_serial_sequence，不使用 pg_get_serial_identifier。"""
    source = (BACKEND_ROOT / "db" / "migration.py").read_text(encoding="utf-8")
    assert "pg_get_serial_sequence" in source
    assert "pg_get_serial_identifier" not in source


def test_migrate_table_uses_begin_nested():
    """源码中 _migrate_table 应使用 session.begin_nested()。"""
    source = (BACKEND_ROOT / "db" / "migration.py").read_text(encoding="utf-8")
    assert "begin_nested" in source


def test_migrate_table_column_mapping_is_copy_semantics():
    """源码中 column_mapping 应保留原始列（复制而非替换）。"""
    source = (BACKEND_ROOT / "db" / "migration.py").read_text(encoding="utf-8")
    assert "if key in valid_columns" in source
    assert "values[key] = value" in source


# ============================================
# m1: migrate() 非零退出行为测试
# ============================================

@pytest.mark.asyncio
async def test_migrate_exits_nonzero_when_sequence_sync_fails(monkeypatch, tmp_path):
    """migrate() 在 _sync_pg_sequences 返回 failed > 0 时应 sys.exit(1)。

    M2 fix: 不再搜索源码字符串，而是 mock _sync_pg_sequences 返回失败，
    断言 pytest.raises(SystemExit)。
    """
    # 创建一个假的 SQLite 数据库文件
    import sqlite3
    fake_db_path = tmp_path / "fake.db"
    conn = sqlite3.connect(fake_db_path)
    conn.execute("CREATE TABLE config (key TEXT, value TEXT)")
    conn.commit()
    conn.close()

    # mock _sync_pg_sequences 返回 (0, 1) 表示 1 个失败
    async def fake_sync(pg_db):
        return 0, 1
    monkeypatch.setattr("db.migration._sync_pg_sequences", fake_sync)

    # mock pg_db 避免真实连接（直接操作模块属性）
    import db.pg_database as _pg_mod
    fake_pg_db = MagicMock()
    fake_pg_db.init = AsyncMock()
    fake_pg_db.close = AsyncMock()
    monkeypatch.setattr(_pg_mod, "pg_db", fake_pg_db)

    # mock _migrate_table 返回 (0, 0)
    async def fake_migrate_table(pg_db, table_name, pg_table, rows, conflict_cols, col_mapping):
        return 0, 0
    monkeypatch.setattr("db.migration._migrate_table", fake_migrate_table)

    # mock _read_sqlite_table 返回空列表
    monkeypatch.setattr("db.migration._read_sqlite_table", lambda conn, name: [])

    # 导入 migrate 并执行，应抛出 SystemExit
    from db.migration import migrate

    with pytest.raises(SystemExit) as exc_info:
        await migrate(fake_db_path)

    assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_migrate_exits_nonzero_when_row_migration_fails(monkeypatch, tmp_path):
    """migrate() 在 _migrate_table 返回 failed > 0 时也应 sys.exit(1)。"""
    import sqlite3
    fake_db_path = tmp_path / "fake.db"
    conn = sqlite3.connect(fake_db_path)
    conn.execute("CREATE TABLE config (key TEXT, value TEXT)")
    conn.commit()
    conn.close()

    # mock _migrate_table 返回 (5, 2) 表示 2 行失败
    async def fake_migrate_table(pg_db, table_name, pg_table, rows, conflict_cols, col_mapping):
        return 5, 2
    monkeypatch.setattr("db.migration._migrate_table", fake_migrate_table)

    # mock _sync_pg_sequences 返回成功
    async def fake_sync(pg_db):
        return 10, 0
    monkeypatch.setattr("db.migration._sync_pg_sequences", fake_sync)

    import db.pg_database as _pg_mod
    fake_pg_db = MagicMock()
    fake_pg_db.init = AsyncMock()
    fake_pg_db.close = AsyncMock()
    monkeypatch.setattr(_pg_mod, "pg_db", fake_pg_db)

    monkeypatch.setattr("db.migration._read_sqlite_table", lambda conn, name: [])

    from db.migration import migrate

    with pytest.raises(SystemExit) as exc_info:
        await migrate(fake_db_path)

    assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_migrate_exits_zero_when_all_success(monkeypatch, tmp_path):
    """migrate() 在全部成功时应正常返回（退出码 0 / 无 SystemExit）。"""
    import sqlite3
    fake_db_path = tmp_path / "fake.db"
    conn = sqlite3.connect(fake_db_path)
    conn.execute("CREATE TABLE config (key TEXT, value TEXT)")
    conn.commit()
    conn.close()

    async def fake_migrate_table(pg_db, table_name, pg_table, rows, conflict_cols, col_mapping):
        return 5, 0
    monkeypatch.setattr("db.migration._migrate_table", fake_migrate_table)

    async def fake_sync(pg_db):
        return 10, 0
    monkeypatch.setattr("db.migration._sync_pg_sequences", fake_sync)

    import db.pg_database as _pg_mod
    fake_pg_db = MagicMock()
    fake_pg_db.init = AsyncMock()
    fake_pg_db.close = AsyncMock()
    monkeypatch.setattr(_pg_mod, "pg_db", fake_pg_db)

    monkeypatch.setattr("db.migration._read_sqlite_table", lambda conn, name: [])

    from db.migration import migrate

    # 不应抛出 SystemExit
    await migrate(fake_db_path)
