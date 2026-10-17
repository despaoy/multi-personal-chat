"""Configuration failures report their actual stage without leaking private causes."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from api import config
from inference import model_manager


@pytest.fixture
def rig(monkeypatch):
    refresh = Mock()
    state = SimpleNamespace(selected="vllm", saved={}, threads=[])

    def select(provider):
        state.selected = provider.value
        return True

    def status():
        state.threads.append(threading.get_ident())
        return {"currentProvider": state.selected}

    def save(update):
        state.saved.update(update)

    manager = SimpleNamespace(
        set_provider=Mock(side_effect=select),
        get_status=Mock(side_effect=status),
        get_current_provider=Mock(return_value=SimpleNamespace(_refresh_db_config=refresh)),
    )
    state.manager, state.refresh = manager, refresh
    state.acquire = Mock(return_value=manager)
    state.save = Mock(side_effect=save)
    state.invalidate = Mock()
    monkeypatch.setattr(model_manager, "get_model_manager", state.acquire)
    monkeypatch.setattr(config, "db", SimpleNamespace(update_config=state.save))
    monkeypatch.setattr(config, "invalidate_config_cache", state.invalidate)
    return state


def request(body=None):
    return SimpleNamespace(json=AsyncMock(return_value=body))


@pytest.mark.asyncio
async def test_status_failure_is_safe_and_preserves_original_cause(rig, caplog):
    failure = OSError("synthetic-private-status-details")
    rig.manager.get_status.side_effect = failure
    with pytest.raises(HTTPException) as caught:
        await config.get_model_status(current_user={"role": "admin"})
    assert caught.value.status_code == 503 and caught.value.__cause__ is failure
    assert "synthetic-private" not in caught.value.detail + caplog.text


@pytest.mark.asyncio
async def test_status_uses_bounded_executor_and_preserves_success_body(rig):
    result = await config.get_model_status(current_user={"role": "admin"})
    assert result == {"success": True, "status": {"currentProvider": "vllm"}}
    assert rig.threads and all(item != threading.get_ident() for item in rig.threads)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [None, [], "synthetic-private-body", {}, {"provider": "unknown"}])
async def test_invalid_request_cannot_acquire_manager_or_change_state(rig, body):
    with pytest.raises(HTTPException) as caught:
        await config.set_model_provider(request(body), current_user={"role": "admin"})
    assert caught.value.status_code == 400 and "synthetic-private" not in caught.value.detail
    rig.acquire.assert_not_called()
    rig.save.assert_not_called()
    rig.invalidate.assert_not_called()


@pytest.mark.asyncio
async def test_malformed_json_is_a_request_error(rig):
    incoming = request()
    failure = json.JSONDecodeError("synthetic-private-json", "synthetic-private-input", 0)
    incoming.json.side_effect = failure
    with pytest.raises(HTTPException) as caught:
        await config.set_model_provider(incoming, current_user={"role": "admin"})
    assert caught.value.status_code == 400 and caught.value.__cause__ is failure
    assert "synthetic-private" not in caught.value.detail
    rig.acquire.assert_not_called()


@pytest.mark.asyncio
async def test_disallowed_provider_does_not_persist_or_select_another_provider(rig):
    failure = ValueError("synthetic-private-disallowed")
    rig.manager.set_provider.side_effect = failure
    with pytest.raises(HTTPException) as caught:
        await config.set_model_provider(request({"provider": "mock"}), current_user={"role": "admin"})
    assert caught.value.status_code == 400 and caught.value.__cause__ is failure
    assert rig.selected == "vllm"
    rig.save.assert_not_called()
    rig.invalidate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage,switched,persisted",
    [
        ("acquire", False, False),
        ("selection", None, False),
        ("persistence", True, None),
        ("cache", True, True),
        ("refresh", True, True),
        ("status", True, True),
    ],
)
async def test_faults_report_confirmed_and_unconfirmed_partial_state(rig, caplog, stage, switched, persisted):
    failure = OSError("synthetic-private-operation-details")
    operations = {
        "acquire": rig.acquire,
        "selection": rig.manager.set_provider,
        "persistence": rig.save,
        "cache": rig.invalidate,
        "refresh": rig.refresh,
        "status": rig.manager.get_status,
    }
    operations[stage].side_effect = failure
    with pytest.raises(HTTPException) as caught:
        await config.set_model_provider(request({"provider": "openai_compat"}), current_user={"role": "admin"})
    error = caught.value
    assert error.status_code == 503 and error.__cause__ is failure
    expected_stage = {
        "acquire": "provider_selection",
        "selection": "provider_selection",
        "persistence": "config_persistence",
        "cache": "config_cache",
        "refresh": "runtime_config",
        "status": "model_status",
    }[stage]
    assert error.detail["stage"] == expected_stage
    assert error.detail["providerSwitched"] is switched and error.detail["configPersisted"] is persisted
    assert "synthetic-private" not in str(error.detail) + caplog.text
    assert rig.manager.set_provider.call_count == int(stage != "acquire")
    assert rig.selected == ("vllm" if stage in {"acquire", "selection"} else "openai_compat")
    assert bool(rig.saved) == (persisted is True)
    if stage in {"acquire", "selection", "persistence"}:
        rig.invalidate.assert_not_called()
    if stage not in {"refresh", "status"}:
        rig.refresh.assert_not_called()


@pytest.mark.asyncio
async def test_persistence_acknowledgement_failure_is_not_claimed_as_unchanged(rig):
    failure = OSError("synthetic-private-commit-acknowledgement")

    def save_then_fail(update):
        rig.saved.update(update)
        raise failure

    rig.save.side_effect = save_then_fail
    with pytest.raises(HTTPException) as caught:
        await config.set_model_provider(request({"provider": "ollama"}), current_user={"role": "admin"})
    assert caught.value.detail["configPersisted"] is None
    assert rig.saved == {"modelProvider": "ollama"} and rig.selected == "ollama"
    assert caught.value.__cause__ is failure
    rig.manager.set_provider.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["openai_compat", "ollama"])
async def test_switch_success_keeps_response_and_each_required_operation(rig, name):
    result = await config.set_model_provider(request({"provider": name}), current_user={"role": "admin"})
    assert result == {"success": True, "message": f"已切换到提供商: {name}", "status": {"currentProvider": name}}
    assert rig.saved == {"modelProvider": name} and rig.selected == name
    rig.manager.set_provider.assert_called_once()
    rig.save.assert_called_once_with({"modelProvider": name})
    rig.invalidate.assert_called_once()
    assert rig.refresh.call_count == int(name == "openai_compat")
    assert rig.threads and all(item != threading.get_ident() for item in rig.threads)


@pytest.mark.asyncio
async def test_http_partial_failure_preserves_503_and_safe_state(rig):
    rig.save.side_effect = OSError("synthetic-private-http-storage")
    app = FastAPI()
    app.include_router(config.router)
    app.dependency_overrides[config.get_current_admin] = lambda: {"role": "admin"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://unit") as client:
        response = await client.put("/api/model/provider", json={"provider": "ollama"})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["providerSwitched"] and detail["configPersisted"] is None
    assert "synthetic-private" not in response.text
