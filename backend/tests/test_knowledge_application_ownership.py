"""Authenticated applications keep knowledge, index and cache authority together."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from api import knowledge
from app.config import create_access_token
from app.runtime import RuntimeContainer
from db.database import SQLiteDB
from knowledge import rag_helper, vector_db


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setattr(vector_db.VectorDatabase, "_check_gpu_availability", lambda self: False)
    monkeypatch.setattr(
        vector_db.VectorDatabase,
        "_get_embeddings_batch",
        lambda self, texts: np.tile(np.eye(1, 384, dtype=np.float32), (len(texts), 1)),
    )
    monkeypatch.setattr(
        vector_db.VectorDatabase, "_get_embedding", lambda self, text: np.eye(1, 384, dtype=np.float32)[0]
    )
    forbidden = Mock(side_effect=AssertionError("Application knowledge must not use process-default storage"))
    monkeypatch.setattr(knowledge, "db", SimpleNamespace(get_knowledge_bases=forbidden), raising=False)
    monkeypatch.setattr(vector_db, "get_vector_db", forbidden)
    monkeypatch.setattr(rag_helper, "get_vector_db", forbidden)
    databases = [SQLiteDB(tmp_path / f"application-{index}.sqlite") for index in range(2)]
    bodies = [
        "甲库完整规则：仅周二办理，费用68元，必须携带原件；预约不能代替原件，雨天不受理。",
        "乙库完整规则：仅周四办理，不收费，必须携带登记原件；雨天照常，预约不能代替原件。",
    ]
    for database, body in zip(databases, bodies, strict=True):
        database.add_user("synthetic-admin", "unused-hash-for-signed-token")
        base = database.create_knowledge_base("同名资料库", "当前应用独立来源")
        database.save_knowledge_document(
            dict(title="完整办理规则", content=body, knowledge_base_id=base["id"]), chunks=[body]
        )
    try:
        yield SimpleNamespace(databases=databases, bodies=bodies, forbidden=forbidden)
    finally:
        for database in databases:
            database.close_connection()


def client(database, *, authenticated=True):
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(db=database, is_pg_mode=lambda: False, startup_env={})
    app.include_router(knowledge.router)
    headers = {"Authorization": "Bearer " + create_access_token("synthetic-admin", 1, "admin")} if authenticated else {}
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="https://isolated", headers=headers
    )


async def test_authenticated_metadata_and_document_reads_use_the_application_store(stores):
    for database, body in zip(stores.databases, stores.bodies, strict=True):
        async with client(database) as api:
            bases = await api.get("/api/knowledge/bases")
            assert bases.status_code == 200 and bases.json()["bases"][0]["name"] == "同名资料库"
            document = await api.get("/api/knowledge/documents/1")
            assert document.status_code == 200 and document.json()["document"]["content"] == body
            stats = await api.get("/api/knowledge/stats")
            assert stats.status_code == 200 and stats.json()["success"]
        async with client(database, authenticated=False) as api:
            assert (await api.get("/api/knowledge/bases")).status_code == 401
    stores.forbidden.assert_not_called()


async def test_equal_document_ids_and_queries_keep_index_cache_and_updates_in_their_store(stores):
    async def search(database, expected, excluded):
        async with client(database) as api:
            response = await api.post(
                "/api/knowledge/search",
                json={"query": "办理费用与原件条件", "topK": 3, "knowledgeBaseName": "同名资料库"},
            )
            assert response.status_code == 200
            results = response.json()["results"]
            assert len(results) == 1 and expected in results[0]["content"] and excluded not in results[0]["content"]

    await asyncio.gather(
        *(
            search(database, body, stores.bodies[1 - index])
            for index, (database, body) in enumerate(zip(stores.databases, stores.bodies, strict=True))
        )
    )
    revised = "甲库新规则：仅周五办理，费用39元，必须携带原件；雨天暂停，预约不能代替原件。"
    async with client(stores.databases[0]) as api:
        response = await api.put("/api/knowledge/documents/1", json={"content": revised})
        assert response.status_code == 200 and response.json()["success"]
    await asyncio.gather(
        search(stores.databases[0], revised, stores.bodies[0]), search(stores.databases[1], stores.bodies[1], revised)
    )
    assert stores.databases[1].get_knowledge_document(1)["content"] == stores.bodies[1]
    stores.forbidden.assert_not_called()


async def test_owned_database_failure_is_explicit_and_does_not_borrow_another_store(stores, monkeypatch):
    failed = Mock(side_effect=OSError("synthetic-private-owned-store-fault"))
    monkeypatch.setattr(stores.databases[0], "get_knowledge_bases", failed)
    async with client(stores.databases[0]) as api:
        response = await api.get("/api/knowledge/bases")
        assert response.status_code == 500 and "synthetic-private" not in response.text
    async with client(stores.databases[1]) as api:
        response = await api.get("/api/knowledge/bases")
        assert response.status_code == 200 and response.json()["success"]
    failed.assert_called_once()
    stores.forbidden.assert_not_called()


async def test_chat_retrieval_verifies_complete_originals_from_its_owned_database(stores, monkeypatch):
    from knowledge import chat_retrieval

    monkeypatch.setattr(chat_retrieval, "get_provider_context_budget", lambda: SimpleNamespace(window_tokens=65536))
    bundles = await asyncio.gather(
        *(
            chat_retrieval.retrieve_chat_knowledge(
                "《完整办理规则》的办理费用与原件条件", 3, {"knowledge_base_id": 1}, database=database
            )
            for database in stores.databases
        )
    )
    for bundle, body in zip(bundles, stores.bodies, strict=True):
        assert bundle["original_source_packets"][0]["original_body"] == body
        assert bundle["source_coverage"][0]["original_source_receipt"]["authority"] == "fresh_knowledge_document_read"
    failure = OSError("synthetic-owned-index-read-fault")
    monkeypatch.setattr(stores.databases[0], "get_config_value", Mock(side_effect=failure))
    with pytest.raises(OSError) as raised:
        await chat_retrieval.retrieve_chat_knowledge(
            "完整办理规则", 3, {"knowledge_base_id": 1}, database=stores.databases[0]
        )
    assert raised.value is failure
    stores.forbidden.assert_not_called()


def test_manual_training_source_reads_use_the_selected_database(stores, monkeypatch):
    from db import adapter
    from knowledge import intent_trainer

    monkeypatch.setattr(adapter, "db", SimpleNamespace(get_knowledge_documents=stores.forbidden))
    for database, body in zip(stores.databases, stores.bodies, strict=True):
        assert intent_trainer._load_kb_documents(database.get_knowledge_bases(), database=database) == {
            "同名资料库": "【完整办理规则】\n" + body
        }
    stores.forbidden.assert_not_called()
