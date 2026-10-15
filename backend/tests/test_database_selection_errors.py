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
            return SimpleNamespace(sync_pg_db=store, db=store)
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
    assert namespace["get_db"]() is store and namespace["is_pg_mode"]() is pg


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
