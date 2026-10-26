"""Database choice is explicit and selected-backend failures stop startup."""

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from db.adapter import _should_use_postgresql
from db.database import SQLiteDB


@pytest.mark.parametrize("value", ["", "maybe", "flase", "ture"])
def test_invalid_database_flag_is_not_an_sqlite_selection(value):
    with pytest.raises(ValueError, match="USE_POSTGRESQL"):
        _should_use_postgresql({"USE_POSTGRESQL": value, "DATABASE_URL": "postgresql://fixture@db/fixture"})


@pytest.mark.parametrize("url", ["mysql://fixture@db/fixture", "https://db/fixture", "fiction-private-invalid-url"])
def test_unsupported_database_url_does_not_select_sqlite(url):
    with pytest.raises(ValueError, match="DATABASE_URL") as error:
        _should_use_postgresql({"DATABASE_URL": url})
    assert url not in str(error.value)


@pytest.mark.parametrize("components", [
    {"PG_HOST": "db"},
    {"PG_HOST": "db", "PG_USER": "fixture", "PG_PASSWORD": "fiction-private-password"},
    {"PG_USER": "fixture", "PG_PASSWORD": "fiction-private-password", "PG_DATABASE": "fixture"},
])
def test_partial_postgresql_configuration_fails_without_changing_backend(components):
    with pytest.raises(ValueError, match="PostgreSQL") as error:
        _should_use_postgresql(components)
    assert "fiction-private-password" not in str(error.value)


@pytest.mark.parametrize("value,expected", [(" ON ", True), ("off", False)])
def test_explicit_database_choice_keeps_priority_over_other_configuration(value, expected):
    assert _should_use_postgresql({"USE_POSTGRESQL": value, "DATABASE_URL": "fiction-unused-url", "PG_HOST": "db"}) is expected


@pytest.fixture
def load_adapter(monkeypatch, tmp_path):
    # Execute the actual composition module with controlled dependency imports;
    # no production configuration, PostgreSQL connection or user data is used.
    store = SQLiteDB(tmp_path / "selected-store.sqlite")
    imports = []
    original_import = builtins.__import__
    failure = None

    def controlled_import(name, *args, **kwargs):
        if name in {"db.pg_database", "db.database"}:
            imports.append(name)
            if name == "db.pg_database" and failure is not None:
                raise failure
            return SimpleNamespace(sync_pg_db=store, SQLiteDB=lambda: store)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", controlled_import)

    def load(*, pg, environment, fail=None):
        nonlocal failure
        failure = fail
        monkeypatch.setenv("USE_POSTGRESQL", pg)
        monkeypatch.setenv("ENVIRONMENT", environment)
        namespace = {"__name__": "fixture_database_composition"}
        path = Path(__file__).resolve().parents[1] / "db" / "adapter.py"
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        return namespace

    yield load, imports, store
    store.close_connection()


@pytest.mark.parametrize("environment", ["development", "production"])
@pytest.mark.parametrize("pg", [True, False])
def test_startup_imports_only_the_selected_backend(load_adapter, environment, pg):
    load, imports, store = load_adapter
    namespace = load(pg=str(pg), environment=environment)
    assert imports == ["db.pg_database" if pg else "db.database"]
    assert namespace["db"] is store and namespace["is_pg_mode"]() is pg


@pytest.mark.parametrize("environment", ["development", "production"])
def test_missing_selected_dependency_never_imports_sqlite(load_adapter, environment, caplog):
    load, imports, _ = load_adapter
    failure = ModuleNotFoundError("fiction-private-dependency-path")
    with pytest.raises(RuntimeError, match="PostgreSQL") as error:
        load(pg="true", environment=environment, fail=failure)
    assert error.value.__cause__ is failure and imports == ["db.pg_database"]
    assert "fiction-private-dependency-path" not in str(error.value) + caplog.text


@pytest.mark.parametrize("environment", ["development", "production"])
def test_selected_backend_configuration_failure_preserves_the_original_error(load_adapter, environment):
    load, imports, _ = load_adapter
    failure = ValueError("fiction-backend-config-error")
    with pytest.raises(ValueError) as error:
        load(pg="true", environment=environment, fail=failure)
    assert error.value is failure and imports == ["db.pg_database"]


def test_invalid_flag_stops_startup_before_importing_any_backend(load_adapter):
    load, imports, _ = load_adapter
    with pytest.raises(ValueError, match="USE_POSTGRESQL"):
        load(pg="flase", environment="development")
    assert imports == []



@pytest.mark.parametrize("mode", ["definitions", "postgres_api", "sqlite_adapter"])
def test_database_creation_belongs_to_selected_adapter_not_definition_import(tmp_path, mode):
    import json
    import os
    import subprocess
    import sys

    source = """import importlib,json,logging,sqlite3,sys
from pathlib import Path
logging.disable(logging.CRITICAL)
from app import env
env.load_backend_env=lambda *args,**kwargs:None
mode=sys.argv[1]
opens=[]
original=sqlite3.connect
def connect(*args,**kwargs):
 opens.append(str(args[0]))
 if mode!='sqlite_adapter':raise OSError('Unused SQLite must not be opened')
 return original(*args,**kwargs)
sqlite3.connect=connect
try:
 definitions=importlib.import_module('db.database')
 assert not opens
 assert definitions.get_lora_path_by_id('absent-adapter',loras=[]) is None
 if mode=='postgres_api':importlib.import_module('api.generate')
 if mode=='sqlite_adapter':
  selected=importlib.import_module('db.adapter')
  assert selected.db is importlib.import_module('db.adapter').db
  assert len(opens)==1 and Path(selected.db.db_path).is_file()
  selected.db.close_connection()
 print(json.dumps(dict(kind='complete',sqlite_connections=len(opens))))
except Exception as error:
 print(json.dumps(dict(kind=type(error).__name__,sqlite_connections=len(opens))))
"""
    env = dict(os.environ, USE_POSTGRESQL="false" if mode=="sqlite_adapter" else "true",
        DATABASE_URL="postgresql+asyncpg://unused@127.0.0.1:1/unused",
        DATABASE_PATH=str(tmp_path/"selected.sqlite"), LORA_PATH=str(tmp_path/"loras"),
        ENVIRONMENT="test", JWT_SECRET="synthetic-database-owner-secret-at-least-32-characters")
    result = subprocess.run([sys.executable,"-c",source,mode],env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"kind":"complete","sqlite_connections":int(mode=="sqlite_adapter")}
    assert (tmp_path/"selected.sqlite").exists() == (mode=="sqlite_adapter")
