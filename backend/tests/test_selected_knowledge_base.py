"""Selected routes use current database identities and never broaden on failure."""

import sqlite3
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from db.database import SQLiteDB
from db.schemas import MessageRequest

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
    from api.generate import _resolve_kb_id

    base = database.get_knowledge_base_by_name(KB_NAME)
    assert base["id"] == _resolve_kb_id(KB_NAME, database=database)
    renamed = '虚构新名称\' OR 1=1 --'
    database.update_knowledge_base(base["id"], dict(name=renamed))
    assert database.get_knowledge_base_by_name(KB_NAME) is None
    assert database.get_knowledge_base_by_name(renamed)["id"] == base["id"]
    assert database.get_knowledge_base_by_name("' OR 1=1 --") is None
    with pytest.raises(LookupError, match="Selected knowledge base is missing"):
        _resolve_kb_id(KB_NAME, database=database)
    assert _resolve_kb_id(renamed, database=database) == base["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("injected", [True, False])
@pytest.mark.parametrize("mode", ["existing_empty", "missing", "read_error", "unrouted"])
async def test_selected_base_route_stops_on_failure_and_preserves_legitimate_no_match(database, monkeypatch, injected, mode):
    from api import generate
    from knowledge import intent_detector

    selected = database.get_knowledge_base_by_name(KB_NAME)
    if mode == "missing":
        database.delete_knowledge_base(selected["id"])
    error = sqlite3.OperationalError("Declared isolated database read failure")
    if mode == "read_error":
        def fail(_name):
            raise error
        monkeypatch.setattr(database, "get_knowledge_base_by_name", fail)
    seen = dict(retrieval=0, model=0)

    def forbidden_global(_name):
        pytest.fail("An injected database must not route through the global database")

    def no_document(_id):
        pytest.fail("No source was admitted in this no-match control")

    monkeypatch.setattr(generate, "db", SimpleNamespace(
        get_knowledge_base_by_name=forbidden_global, get_knowledge_document=no_document,
    ) if injected else database)
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "false")
    monkeypatch.setenv("RAG_CITATIONS_ENABLED", "false")
    monkeypatch.setenv("CHARACTER_INDEPENDENT_TASKS_ENABLED", "false")
    monkeypatch.setattr(intent_detector, "needs_rag", lambda _: (True, "declared unit route", None if mode == "unrouted" else KB_NAME))
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "完整虚构测试：只按证据回答，缺失资料保持未知。")
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: SimpleNamespace(window_tokens=65536, evidence_max_chars=0))

    async def retrieve(query, top_k, filters, **kwargs):
        seen["retrieval"] += 1
        assert query == QUERY
        assert filters == (None if mode == "unrouted" else dict(knowledge_base_id=selected["id"]))
        return dict(results=[], citations=[], abstained=True)

    async def model(**kwargs):
        seen["model"] += 1
        wire = "\n".join(message["content"] for message in kwargs["messages"])
        assert QUERY in wire and OTHER not in wire
        return "当前没有可核对的资料，费用、开放时段与雨天例外均未知。"

    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    pending = generate._generate_with_retrieval(
        MessageRequest(message=QUERY), None, runtime_config=dict(useKnowledgeBase=True, maxTokens=256),
        model_generate=model, message_db=database if injected else None,
    )
    if mode in {"missing", "read_error"}:
        with pytest.raises(HTTPException) as caught:
            await pending
        assert caught.value.status_code == 503
        if mode == "read_error":
            assert caught.value.__cause__ is error
        else:
            assert isinstance(caught.value.__cause__, LookupError)
        assert seen == dict(retrieval=0, model=0)
    else:
        _, used, metadata = await pending
        assert used and metadata["abstained"] and metadata["answerMode"] == "abstention"
        assert seen == dict(retrieval=1, model=1)
    assert database.get_knowledge_document(1)["content"] == OTHER


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
