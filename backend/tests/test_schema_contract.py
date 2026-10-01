"""Schema contract tests: ORM, SQLite runtime, and Alembic stay aligned."""

from __future__ import annotations

import ast
from pathlib import Path

from db.models import metadata


BACKEND = Path(__file__).resolve().parents[1]

SCHEMA_COLUMN_CONTRACTS = (
    ("claw_tools", "description"),
    ("claw_tools", "code"),
    ("claw_tools", "enabled"),
    ("intent_active_kbs", "isActive"),
    ("model_invocations", "platform"),
    ("model_invocations", "usedRag"),
    ("model_invocations", "usedLora"),
    ("saved_dialogues", "dialogue_count"),
    ("training_tasks", "lora_name"),
)


def _norm_default(value):
    if value is None:
        return None
    return str(value).replace("'", "")


def test_sqlite_runtime_matches_orm_schema(tmp_path):
    """Check every table and the established nullable/default contracts once."""
    from db.database import SQLiteDB

    db = SQLiteDB(tmp_path / "schema.db")
    try:
        connection = db.get_connection()
        for table_name, table in metadata.tables.items():
            info = {
                row["name"]: row
                for row in connection.execute(f'PRAGMA table_info("{table_name}")')
            }
            assert set(info) == set(table.columns.keys()), table_name
            for contract_table, column in SCHEMA_COLUMN_CONTRACTS:
                if contract_table != table_name:
                    continue
                col = table.columns[column]
                assert bool(info[column]["notnull"]) == (not col.nullable), (
                    f"{table_name}.{column} NOT NULL mismatch"
                )
                orm_default = col.server_default.arg if col.server_default is not None else None
                assert _norm_default(info[column]["dflt_value"]) == _norm_default(orm_default), (
                    f"{table_name}.{column} default mismatch"
                )
    finally:
        db.close_connection()


def test_session_settings_not_in_orm():
    assert "session_settings" not in metadata.tables
    assert "api_keys" in metadata.tables


def test_api_keys_orm_columns():
    table = metadata.tables["api_keys"]
    assert [c.name for c in table.columns] == [
        "id", "key_hash", "key_prefix", "role", "description",
        "created_at", "revoked_at", "last_used_at", "is_active", "rate_limit",
    ]
    assert table.c.key_hash.nullable is False
    assert table.c.key_hash.unique is True
    assert table.c.is_active.nullable is False
    assert table.c.created_at.nullable is False


def _migration_create_table_columns(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "create_table"):
            continue
        name = ast.literal_eval(node.args[0])
        columns = {}
        for arg in node.args[1:]:
            if not (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) and arg.func.attr == "Column"):
                continue
            cname = ast.literal_eval(arg.args[0])
            kwargs = {}
            for kw in arg.keywords:
                if kw.arg in {"nullable", "server_default"}:
                    try:
                        kwargs[kw.arg] = ast.literal_eval(kw.value)
                    except Exception:
                        kwargs[kw.arg] = None
            columns[cname] = kwargs
        result[name] = columns
    return result


def _migration_alter_nullable(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = {}
    upgrade = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "upgrade"), None)
    if upgrade is None:
        return result
    for node in ast.walk(upgrade):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "alter_column"):
            continue
        try:
            table = ast.literal_eval(node.args[0])
            column = ast.literal_eval(node.args[1])
        except (ValueError, TypeError):
            continue
        nullable = None
        for kw in node.keywords:
            if kw.arg == "nullable":
                try:
                    nullable = ast.literal_eval(kw.value)
                except Exception:
                    nullable = None
        result[(table, column)] = nullable

    # Migration 005 declares the repeated SQLite/PostgreSQL alterations once
    # and applies them through batch_op/op in a loop.
    for node in upgrade.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "constraint_changes" for target in node.targets):
            continue
        if not isinstance(node.value, (ast.Tuple, ast.List)):
            continue
        for item in node.value.elts:
            if not isinstance(item, (ast.Tuple, ast.List)) or len(item.elts) < 2:
                continue
            try:
                table = ast.literal_eval(item.elts[0])
                column = ast.literal_eval(item.elts[1])
            except (ValueError, TypeError):
                continue
            result[(table, column)] = False
    return result


def test_migration_005_matches_orm_for_key_columns():
    """Check migration 005 against its original ORM column contract."""
    migration_005 = next((BACKEND / "alembic" / "versions").glob("005_*.py"))
    alters = _migration_alter_nullable(migration_005)

    # api_keys is created in 005.
    migrations = {}
    for path in sorted((BACKEND / "alembic" / "versions").glob("*.py")):
        migrations.update(_migration_create_table_columns(path))
    api = migrations["api_keys"]
    assert api["key_hash"]["nullable"] is False
    assert api["is_active"]["nullable"] is False

    expected_not_null = [
        ("claw_tools", "description"),
        ("claw_tools", "code"),
        ("claw_tools", "enabled"),
        ("intent_active_kbs", "isActive"),
        ("model_invocations", "platform"),
        ("model_invocations", "usedRag"),
        ("model_invocations", "usedLora"),
        ("saved_dialogues", "dialogue_count"),
        ("training_tasks", "lora_name"),
    ]
    for table, column in expected_not_null:
        assert alters.get((table, column)) is False, f"{table}.{column} should be altered to NOT NULL"

    # ORM agrees with the final NOT NULL contract.
    for table, column in expected_not_null:
        col = metadata.tables[table].columns[column]
        assert col.nullable is False, f"{table}.{column} should be NOT NULL in ORM"


def test_migration_005_has_cross_database_legacy_merge_guards():
    migration = next((BACKEND / "alembic" / "versions").glob("005_*.py"))
    source = migration.read_text(encoding="utf-8")

    assert 'dialect == "sqlite"' in source
    assert "batch_alter_table" in source
    assert "INSERT OR IGNORE INTO conversations" in source
    assert "ON CONFLICT (platform" in source
    assert '"conversationType" in session_columns' in source
    assert 'op.drop_table("session_settings")' in source
