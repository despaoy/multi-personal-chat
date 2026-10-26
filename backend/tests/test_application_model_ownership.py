"""Settings, connection snapshots and generation share one database owner."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from api import config, generate
from app.config import create_access_token
from app.runtime import RuntimeContainer
from cache.config_cache import invalidate_config_cache
from db.adapter import db as default_database
from db.database import SQLiteDB
from db.schemas import MessageRequest
from inference import model_manager
from inference.request_context import resolve_request_context

QUESTION = "合成资料：周二09:00开门、17:00闭馆，暴雨暂停开放；预约不能替代原件。请保留时间、否定及例外。"
ANSWER = "测试控制回复：周二09:00开门、17:00闭馆，暴雨暂停开放；预约不能替代原件。"


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai_compat")
    monkeypatch.setenv("MODEL_STORAGE_PATH", str(tmp_path / "models"))
    invalidate_config_cache()
    databases = [SQLiteDB(tmp_path / f"application-{index}.sqlite") for index in range(2)]
    for index, database in enumerate(databases):
        database.add_user("synthetic-owner", "unused-hash-for-signed-token")
        database.update_config(
            {
                "modelProvider": "openai_compat",
                "openaiCompatBaseUrl": f"https://owner-{index}.invalid/v1",
                "openaiCompatApiKey": f"synthetic-owner-{index}-credential",
                "openaiCompatModel": f"fixture-model-{index}",
                "temperature": 0.2 + index * 0.5,
                "maxTokens": 128,
                "topP": 1.0,
                "useKnowledgeBase": False,
            }
        )
    forbidden = Mock(side_effect=AssertionError("Application settings must not use the default database"))
    monkeypatch.setattr(default_database, "_get_connection", forbidden)
    calls = []

    def respond(request):
        calls.append((request.url.host, request.headers["authorization"], json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": ANSWER}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60},
            },
        )

    @asynccontextmanager
    async def client(self, timeout=120.0):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as connection:
            yield connection

    monkeypatch.setattr(model_manager.BaseProvider, "_acquire_http_client", client)
    try:
        yield SimpleNamespace(databases=databases, forbidden=forbidden, calls=calls)
    finally:
        for database in databases:
            owner = getattr(database, "_model_manager", None)
            if owner is not None:
                owner.shutdown()
            database.close_connection()
        invalidate_config_cache()


def client(database):
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(db=database, is_pg_mode=lambda: False, startup_env={})
    app.include_router(config.router)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="https://isolated",
        headers={"Authorization": "Bearer " + create_access_token("synthetic-owner", 1, "admin")},
    )


async def test_authenticated_settings_and_model_status_follow_the_application_database(stores):
    for index, database in enumerate(stores.databases):
        async with client(database) as api:
            response = await api.get("/api/config")
            assert response.status_code == 200
            assert response.json()["config"]["temperature"] == 0.2 + index * 0.5
            assert f"synthetic-owner-{index}-credential" not in response.text
            status = await api.get("/api/model/status")
            assert status.status_code == 200
            assert status.json()["status"]["providers"]["openai_compat"]["modelName"] == f"fixture-model-{index}"
            api.headers.pop("Authorization")
            assert (await api.get("/api/config")).status_code == 401
    stores.forbidden.assert_not_called()


async def test_settings_save_and_provider_selection_stay_in_the_selected_store(stores):
    other_before = stores.databases[1].config
    async with client(stores.databases[0]) as api:
        response = await api.put("/api/config", json={"openaiCompatModel": "revised-fixture-model", "temperature": 0.3})
        assert response.status_code == 200 and response.json()["success"]
        assert response.json()["config"]["openaiCompatModel"] == "revised-fixture-model"
        selected = await api.put("/api/model/provider", json={"provider": "openai_compat"})
        assert selected.status_code == 200 and selected.json()["success"]
        assert selected.json()["status"]["providers"]["openai_compat"]["modelName"] == "revised-fixture-model"
        assert (await api.get("/api/config")).json()["config"]["temperature"] == 0.3
    assert stores.databases[1].config == other_before
    stores.forbidden.assert_not_called()


async def test_request_snapshots_keep_their_connection_during_setting_updates(stores):
    contexts = [
        await resolve_request_context(model_manager.get_model_manager(database=database))
        for database in stores.databases
    ]
    identities = [context.connection_id for context in contexts]
    async with client(stores.databases[0]) as api:
        assert (await api.put("/api/config", json={"openaiCompatModel": "revised-fixture-model"})).status_code == 200
    await asyncio.gather(
        *(
            context.provider.async_complete(
                [{"role": "user", "content": QUESTION}], temperature=0.2, max_tokens=128, top_p=1.0
            )
            for context in contexts
        )
    )
    assert [context.connection_id for context in contexts] == identities
    by_host = {host: (credential, body) for host, credential, body in stores.calls}
    for index in range(2):
        credential, body = by_host[f"owner-{index}.invalid"]
        assert credential == f"Bearer synthetic-owner-{index}-credential"
        assert body["model"] == f"fixture-model-{index}" and body["messages"][-1]["content"] == QUESTION
    refreshed = await resolve_request_context(model_manager.get_model_manager(database=stores.databases[0]))
    assert refreshed.provider.model == "revised-fixture-model" and refreshed.connection_id != identities[0]
    stores.forbidden.assert_not_called()


async def test_chat_generation_uses_its_database_model(stores, monkeypatch):
    from services import chat_answer

    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "circuit_breaker_registry", None)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(chat_answer, "get_lora_system_prompt", lambda _: "只依据完整输入保留时间、否定及例外。")
    for index, database in enumerate(stores.databases):
        response = await generate._generate_reply_impl(
            MessageRequest(message=QUESTION, loraName="default"),
            persist_message=False,
            record_invocation=False,
            enable_rag=False,
            message_db=database,
        )
        assert response.reply == ANSWER and response.modelInvoked
        host, credential, body = stores.calls[-1]
        assert host == f"owner-{index}.invalid" and body["model"] == f"fixture-model-{index}"
        assert credential == f"Bearer synthetic-owner-{index}-credential"
        assert body["temperature"] == 0.2 + index * 0.5 and body["max_tokens"] == 128
        assert QUESTION in str(body["messages"])
    stores.forbidden.assert_not_called()


async def test_owned_configuration_failure_is_reported_without_a_default_store(stores):
    failure = OSError("synthetic-private-owned-configuration-fault")

    class BrokenDatabase:
        @property
        def config(self):
            raise failure

    database = BrokenDatabase()
    with pytest.raises(HTTPException) as raised:
        await config.get_config(current_user={"role": "admin"}, database=database)
    assert raised.value.status_code == 503 and raised.value.__cause__ is failure
    assert "synthetic-private" not in str(raised.value.detail)
    with pytest.raises(OSError) as raised:
        model_manager.get_model_manager(database=database)
    assert raised.value is failure
    assert getattr(database, "_model_manager", None) is None
    stores.forbidden.assert_not_called()


def test_concurrent_resolution_reuses_one_manager_per_database(stores, monkeypatch):
    database = stores.databases[0]
    with ThreadPoolExecutor(max_workers=4) as pool:
        owners = list(pool.map(lambda _: model_manager.get_model_manager(database=database), range(8)))
    assert all(owner is owners[0] for owner in owners)
    assert model_manager.get_model_manager(database=stores.databases[1]) is not owners[0]
    assert model_manager.get_model_manager(database=database).get_current_provider().model == "fixture-model-0"
    from db import adapter

    monkeypatch.setattr(adapter, "db", database)
    assert model_manager.get_model_manager() is owners[0]
    stores.forbidden.assert_not_called()
