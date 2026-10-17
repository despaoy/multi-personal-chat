"""Setting writes must report confirmed or unconfirmed effects after a failure."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from api import config
from inference import model_manager


class Database:
    def __init__(self):
        self.values = {"openaiCompatApiKey": "fiction-stored-private-key"}
        self.write_error = self.read_error = None
        self.write_calls = self.read_calls = 0

    def update_config(self, update):
        self.write_calls += 1
        if self.write_error:
            raise self.write_error
        self.values.update(update)

    @property
    def config(self):
        self.read_calls += 1
        if self.read_error:
            raise self.read_error
        return dict(self.values)


@pytest.fixture
def rig(monkeypatch):
    owned = Database()
    invalidate, refresh = Mock(), Mock()
    provider = SimpleNamespace(_refresh_db_config=refresh)
    manager = SimpleNamespace(_providers={model_manager.ModelProvider.OPENAI_COMPAT: provider})
    acquire = Mock(return_value=manager)
    monkeypatch.setattr(config, "db", owned)
    monkeypatch.setattr(config, "invalidate_config_cache", invalidate)
    monkeypatch.setattr(model_manager, "get_model_manager", acquire)
    return SimpleNamespace(owned=owned, invalidate=invalidate, refresh=refresh, acquire=acquire)


def request(body):
    return SimpleNamespace(json=AsyncMock(return_value=body))


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["config_persistence", "config_cache", "runtime_config", "config_readback"])
async def test_failed_save_reports_actual_stage_and_confirmed_state(rig, caplog, stage):
    failure = OSError("synthetic-private-setting-operation")
    if stage == "config_persistence":
        rig.owned.write_error = failure
    elif stage == "config_cache":
        rig.invalidate.side_effect = failure
    elif stage == "runtime_config":
        rig.refresh.side_effect = failure
    else:
        rig.owned.read_error = failure
    body = {"openaiCompatModel": "synthetic-model", "maxTokens": 512}
    with pytest.raises(HTTPException) as caught:
        await config.update_config(request(body), current_user={"role": "admin"})
    error = caught.value
    assert error.status_code == 503 and error.__cause__ is failure
    assert error.detail["stage"] == stage
    assert error.detail["configPersisted"] is (None if stage == "config_persistence" else True)
    assert "synthetic-private" not in str(error.detail) + caplog.text
    assert rig.owned.write_calls == 1
    assert (rig.owned.values.get("maxTokens") == 512) == (stage != "config_persistence")
    assert rig.invalidate.call_count == int(stage != "config_persistence")
    assert rig.refresh.call_count == int(stage in {"runtime_config", "config_readback"})
    assert rig.owned.read_calls == int(stage == "config_readback")


@pytest.mark.asyncio
async def test_write_acknowledgement_failure_remains_unconfirmed(rig, monkeypatch):
    failure = OSError("synthetic-private-confirmation-error")

    def save_then_fail(update):
        rig.owned.values.update(update)
        raise failure

    monkeypatch.setattr(rig.owned, "update_config", save_then_fail)
    with pytest.raises(HTTPException) as caught:
        await config.update_config(request({"maxTokens": 512}), current_user={"role": "admin"})
    assert caught.value.status_code == 503 and caught.value.__cause__ is failure
    assert caught.value.detail["configPersisted"] is None
    assert rig.owned.values["maxTokens"] == 512
    rig.invalidate.assert_not_called()


@pytest.mark.asyncio
async def test_success_preserves_masked_credentials_and_normal_response(rig):
    result = await config.update_config(
        request({"maxTokens": 512, "openaiCompatApiKey": "****"}), current_user={"role": "admin"}
    )
    assert result["success"] and result["message"] == "配置已更新" and result["config"]["maxTokens"] == 512
    assert rig.owned.values["openaiCompatApiKey"] == "fiction-stored-private-key"
    assert "fiction-stored-private-key" not in str(result)
    assert rig.owned.write_calls == rig.owned.read_calls == 1
    rig.invalidate.assert_called_once()
    rig.acquire.assert_not_called()


@pytest.mark.asyncio
async def test_malformed_json_has_no_setting_side_effect(rig):
    incoming = request(None)
    failure = json.JSONDecodeError("synthetic-private-json", "synthetic-private-input", 0)
    incoming.json.side_effect = failure
    with pytest.raises(HTTPException) as caught:
        await config.update_config(incoming, current_user={"role": "admin"})
    assert caught.value.status_code == 400 and caught.value.__cause__ is failure
    assert "synthetic-private" not in caught.value.detail
    assert rig.owned.write_calls == 0
    rig.invalidate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [None, []])
async def test_non_object_body_keeps_existing_422_contract(rig, body):
    with pytest.raises(HTTPException) as caught:
        await config.update_config(request(body), current_user={"role": "admin"})
    assert caught.value.status_code == 422 and rig.owned.write_calls == 0
    rig.invalidate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["astrbotIntegrationToken", "ASTRBOTINTEGRATIONTOKENS"])
async def test_server_only_integration_tokens_still_cannot_be_saved(rig, key):
    with pytest.raises(HTTPException) as caught:
        await config.update_config(request({key: "fiction-token"}), current_user={"role": "admin"})
    assert caught.value.status_code == 422 and rig.owned.write_calls == 0
    rig.invalidate.assert_not_called()
