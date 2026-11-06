"""Complete isolated settings writes must publish the selected runtime too."""

import sqlite3

import httpx
import pytest

from api import config as api
from app.main import create_app
from app.runtime import RuntimeContainer
from db.database import SQLiteDB
from inference.model_manager import ModelProvider, get_model_manager


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("MODEL_PROVIDER", "vllm")
    monkeypatch.setenv("MODEL_STORAGE_PATH", str(tmp_path / "models"))
    database = SQLiteDB(tmp_path / "settings.sqlite")
    database.update_config({"modelProvider": "vllm", "maxTokens": 333})
    manager = get_model_manager(database=database)
    application = create_app(
        RuntimeContainer(
            db=database,
            is_pg_mode=lambda: False,
            startup_env={"ENVIRONMENT": "test", "SECURITY_MIDDLEWARE_ENABLED": "false"},
        )
    )
    application.dependency_overrides[api.get_current_admin] = lambda: {"role": "admin"}
    try:
        yield database, manager, application
    finally:
        manager.shutdown()
        database.close_connection()


async def put(application, body):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application, raise_app_exceptions=False), base_url="http://settings-test"
    ) as client:
        return await client.put("/api/config", json=body)


@pytest.mark.asyncio
async def test_one_save_publishes_complete_settings_and_actual_selected_provider(rig):
    database, manager, app = rig
    response = await put(
        app,
        {
            "modelProvider": "openai_compat",
            "openaiCompatBaseUrl": "https://fiction.invalid/v1",
            "openaiCompatModel": "fiction-model",
            "openaiCompatApiKey": "fiction-private-settings-key",
            "maxTokens": 887,
        },
    )
    assert response.status_code == 200 and response.json()["success"] is True
    assert database.config["modelProvider"] == manager._current_provider.value == "openai_compat"
    provider = manager.get_current_provider()
    assert provider.model == "fiction-model" and provider.api_key == "fiction-private-settings-key"
    assert response.json()["config"]["maxTokens"] == 887
    assert "fiction-private-settings-key" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["fiction-unknown", "mock"])
async def test_invalid_provider_cannot_be_saved_or_selected(rig, provider):
    database, manager, app = rig
    original = database.config
    response = await put(app, {"modelProvider": provider, "maxTokens": 887})
    assert response.status_code == 422
    assert database.config == original and manager._current_provider == ModelProvider.VLLM


def block_provider(database):
    database.execute_sql(
        "CREATE TRIGGER settings_write_failure BEFORE INSERT ON config WHEN NEW.key='modelProvider' BEGIN SELECT RAISE(ABORT,'fiction-private-settings-storage'); END"
    )


def test_sqlite_batch_failure_rolls_back_before_recovery(rig):
    database, _, _ = rig
    original = database.config
    block_provider(database)
    with pytest.raises(sqlite3.IntegrityError):
        database.update_config({"maxTokens": 887, "modelProvider": "openai_compat"})
    assert not database.get_connection().in_transaction
    assert database.config == original
    database.execute_sql("DROP TRIGGER settings_write_failure")
    database.update_config({"maxTokens": 887, "modelProvider": "openai_compat"})
    assert database.config["maxTokens"] == 887 and database.config["modelProvider"] == "openai_compat"


@pytest.mark.asyncio
async def test_actual_save_failure_keeps_runtime_and_then_same_operation_recovers(rig, caplog):
    database, manager, app = rig
    original = database.config
    block_provider(database)
    body = {"maxTokens": 887, "modelProvider": "openai_compat"}
    response = await put(app, body)
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["stage"] == "config_persistence" and detail["configPersisted"] is None
    assert detail["providerSwitched"] is False
    assert database.config == original and manager._current_provider == ModelProvider.VLLM
    assert "fiction-private-settings-storage" not in response.text + caplog.text
    database.execute_sql("DROP TRIGGER settings_write_failure")
    restored = await put(app, body)
    assert restored.status_code == 200 and restored.json()["success"]
    assert database.config["maxTokens"] == 887
    assert database.config["modelProvider"] == manager._current_provider.value == "openai_compat"


@pytest.mark.asyncio
async def test_ordinary_prompt_text_is_not_mistaken_for_a_masked_credential(rig):
    database, _, app = rig
    prompt = "请保留 **** 这个 Markdown 格式示例。"
    response = await put(app, {"systemPrompt": prompt})
    assert response.status_code == 200 and response.json()["success"]
    assert response.json()["config"]["systemPrompt"] == database.config["systemPrompt"] == prompt
