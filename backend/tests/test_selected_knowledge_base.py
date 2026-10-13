"""Selected routes use current database identities and never broaden on failure."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from db.database import SQLiteDB

KB_NAME = "虚构夏屿展览"
QUERY = "核对虚构夏屿展览的费用、开放时段与雨天例外；没有对应资料则说明未知。"
OTHER = "别的展览虚构规则：仅周六开放、收费17元、雨天停止，预约不能免雨天限制。"


@pytest.fixture
def database(tmp_path):
    database = SQLiteDB(tmp_path / "selected-base.sqlite")
    database.create_knowledge_base(KB_NAME)
    other = database.create_knowledge_base("虚构其他展览")
    database.save_knowledge_document(
        dict(title="其他展览完整规程", content=OTHER, knowledge_base_id=other["id"]), chunks=[OTHER]
    )
    yield database
    database.close_connection()


def test_name_lookup_uses_current_unique_name_after_rename(database):
    base = database.get_knowledge_base_by_name(KB_NAME)
    assert base["name"] == KB_NAME
    renamed = '虚构新名称\' OR 1=1 --'
    database.update_knowledge_base(base["id"], dict(name=renamed))
    assert database.get_knowledge_base_by_name(KB_NAME) is None
    assert database.get_knowledge_base_by_name(renamed)["id"] == base["id"]
    assert database.get_knowledge_base_by_name("' OR 1=1 --") is None


@pytest.mark.parametrize("mode", ["found", "missing", "read_error"])
def test_postgres_name_lookup_and_sync_adapter_preserve_result_or_failure(mode):
    from db.pg_database import PgDatabase, SyncPgAdapter

    error = OSError("Declared PostgreSQL session failure")
    row = dict(id=8, name=KB_NAME)
    seen = dict(active=0, closed=False)

    class Session:
        async def execute(self, statement):
            # The name must be a bound value rather than SQL text interpolation.
            assert list(statement.compile().params.values()) == [KB_NAME]
            if mode == "read_error":
                raise error
            return SimpleNamespace(fetchone=lambda: SimpleNamespace(_mapping=row) if mode == "found" else None)

    @asynccontextmanager
    async def session():
        seen["active"] += 1
        try:
            yield Session()
        finally:
            seen["active"] -= 1

    async def init():
        pass

    async def close():
        seen["closed"] = True

    pg = object.__new__(PgDatabase)
    pg.async_session, pg.init, pg.close = session, init, close
    adapter = SyncPgAdapter(pg)
    try:
        if mode == "read_error":
            with pytest.raises(OSError) as caught:
                adapter.get_knowledge_base_by_name(KB_NAME)
            assert caught.value is error
        else:
            assert adapter.get_knowledge_base_by_name(KB_NAME) == (row if mode == "found" else None)
    finally:
        adapter.close()
    assert seen == dict(active=0, closed=True)
