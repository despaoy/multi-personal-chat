"""Retired rule dispatch and surviving compatibility operations remain explicit."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from provider_fixtures import cloud_connection

from api import generate
from api import router as compatibility
from db.database import SQLiteDB
from db.schemas import MessageRequest
from inference import model_manager


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{"enabled": True}, {"mode": "rule"}, {"mode": "intent"}])
async def test_retired_live_routing_cannot_be_reenabled(body, monkeypatch):
    from db.schemas import RouterConfigUpdate
    writer = Mock(side_effect=AssertionError("Unsupported routing must not change historical configuration"))
    monkeypatch.setattr(compatibility, "_execute_sql", writer)
    with pytest.raises(HTTPException) as error:
        await compatibility.update_router_config(RouterConfigUpdate(**body), current_user={"isAdmin": True})
    assert error.value.status_code == 409
    writer.assert_not_called()


@pytest.mark.asyncio
async def test_preserved_configuration_is_marked_as_historical(monkeypatch):
    monkeypatch.setattr(compatibility, "_execute_sql", AsyncMock(return_value=[{"value": json.dumps({"enabled": False})}]))
    result = await compatibility.get_router_config(current_user={"isAdmin": True})
    assert result["success"] and result["routingAvailable"] is False and result["config"]["enabled"] is False


@pytest.fixture
def records(monkeypatch, tmp_path):
    database = SQLiteDB(tmp_path / "compat.sqlite")
    monkeypatch.setattr(compatibility, "db", database)
    from db import database as module
    scan = Mock(return_value={"fixture-adapter": str(tmp_path / "fixture-adapter")})
    monkeypatch.setattr(module, "refresh_lora_dir_map", scan)
    return database, scan


@pytest.mark.asyncio
async def test_empty_scan_and_unchecked_adapter_are_normal_results(records):
    database, scan = records
    first = await compatibility.list_adapters(current_user={"isAdmin": True})
    assert first["success"] and first["total"] == 1 and first["adapters"][0]["compatibility"] is None
    scan.return_value = {}
    assert await compatibility.list_adapters(current_user={"isAdmin": True}) == {"success": True, "adapters": [], "total": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("checks", "not-json"), ("checks", "[]"), ("warnings", "{}"), ("errors", "null")])
async def test_stored_report_fault_does_not_remove_failed_fields_and_return_success(records, field, value):
    database, _ = records
    values = dict(name="fixture-adapter", checks='{"base_model":true}', warnings="[]", errors="[]")
    values[field] = value
    database.execute_sql(
        "INSERT INTO adapter_compatibility (adapter_name, checked_at, compatible, checks, warnings, errors) "
        "VALUES (:name, '2026-10-14T01:00:00+00:00', 1, :checks, :warnings, :errors)", values)
    with pytest.raises(HTTPException) as error:
        await compatibility.list_adapters(current_user={"isAdmin": True})
    assert error.value.status_code == 503 and error.value.__cause__ is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["scan", "storage"])
async def test_read_failure_never_becomes_empty_adapter_list(records, monkeypatch, stage):
    database, scan = records
    failure = OSError("fiction-private-adapter-read-error")
    if stage == "scan":
        scan.side_effect = failure
    else:
        monkeypatch.setattr(database, "execute_sql", Mock(side_effect=failure))
    with pytest.raises(HTTPException) as error:
        await compatibility.list_adapters(current_user={"isAdmin": True})
    assert error.value.status_code == 503 and error.value.__cause__ is failure
    assert "fiction-private" not in str(error.value.detail)


@pytest.fixture
def checker(monkeypatch):
    from inference import adapter_checker
    report = SimpleNamespace(compatible=True, checks={"base_model": True, "rank": True}, warnings=[], errors=[])
    check = Mock(return_value=report)
    monkeypatch.setattr(adapter_checker, "AdapterChecker", lambda: SimpleNamespace(check_adapter=check))
    return report, check


@pytest.mark.asyncio
@pytest.mark.parametrize("compatible", [True, False])
async def test_success_and_incompatibility_both_store_the_actual_report(records, checker, compatible):
    database, _ = records
    report, check = checker
    report.compatible = compatible
    report.errors = [] if compatible else ["base_model_mismatch"]
    result = await compatibility.check_adapter_compat("fixture-adapter", current_user={"isAdmin": True})
    assert result["success"] and result["compatible"] is compatible and result["checks"] == report.checks
    rows = database.execute_sql("SELECT * FROM adapter_compatibility")
    assert len(rows) == 1 and bool(rows[0]["compatible"]) is compatible
    assert json.loads(rows[0]["errors"]) == report.errors
    check.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("write_result", ["exception", 0, None])
async def test_completed_check_with_unconfirmed_storage_returns_partial_failure(records, checker, monkeypatch, write_result):
    database, _ = records
    _, check = checker
    failure = OSError("fiction-private-report-storage")
    writer = Mock(side_effect=failure) if write_result == "exception" else Mock(return_value=write_result)
    monkeypatch.setattr(database, "execute_sql", writer)
    with pytest.raises(HTTPException) as error:
        await compatibility.check_adapter_compat("fixture-adapter", current_user={"isAdmin": True})
    assert error.value.status_code == 503 and error.value.detail["stage"] == "adapter_compatibility_record"
    assert error.value.detail["checkCompleted"] is True and error.value.detail["recordSaved"] is None
    assert "fiction-private" not in str(error.value.detail)
    check.assert_called_once()
    if write_result == "exception":
        assert error.value.__cause__ is failure


@pytest.mark.asyncio
async def test_missing_adapter_is_not_checked_or_created(records, checker):
    _, _ = records
    _, check = checker
    with pytest.raises(HTTPException) as error:
        await compatibility.check_adapter_compat("absent-adapter", current_user={"isAdmin": True})
    assert error.value.status_code == 404
    check.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_checker_failure_or_cancel_never_creates_a_report(records, checker, monkeypatch, cancel):
    database, _ = records
    _, check = checker
    failure = asyncio.CancelledError() if cancel else OSError("fiction-checker-fault")
    check.side_effect = failure
    with pytest.raises(asyncio.CancelledError if cancel else HTTPException) as error:
        await compatibility.check_adapter_compat("fixture-adapter", current_user={"isAdmin": True})
    if not cancel:
        assert error.value.__cause__ is failure
    assert database.execute_sql("SELECT id FROM adapter_compatibility") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["base", "explicit_adapter"])
async def test_legacy_rule_config_cannot_choose_a_persona_from_a_name(monkeypatch, selection):
    manager = SimpleNamespace(_current_provider=SimpleNamespace(value="openai_compat"),
        set_lora_adapter=Mock(), set_provider=Mock(),
        get_status=lambda: {"currentProvider": "openai_compat", "providers": {"openai_compat": {"modelName": "fixture-model"}}})
    connection = cloud_connection("fixture-model")
    manager.get_current_provider = lambda: connection
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    legacy = {"enabled": True, "mode": "rule", "persona_keywords": {"hutao": ["胡桃"]}, "persona_adapters": {"hutao": "hutao"}}
    store = SimpleNamespace(config={"useKnowledgeBase": False, "lora_router_config": json.dumps(legacy)},
        loras=[{"id": "fiction-hutao", "name": "hutao", "status": "inactive"}])
    monkeypatch.setattr(generate, "db", store)
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "circuit_breaker_registry", None)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: asyncio.Semaphore(1))
    operation = AsyncMock(return_value=("3张票，每张2元，总额6元。", False, {"modelInvoked": True}))
    monkeypatch.setattr(generate, "_generate_with_retrieval", operation)
    req = MessageRequest(message="只是讨论胡桃这个名字，不要求切换角色。完整题设：3张票，每张2元，计算总额。",
        loraName="hutao" if selection == "explicit_adapter" else "")
    if selection == "base":
        response = await generate._generate_reply_impl(req, persist_message=False, record_invocation=False)
        assert response.reply == "3张票，每张2元，总额6元。" and operation.await_args.args[1] == "default"
    else:
        with pytest.raises(HTTPException) as error:
            await generate._generate_reply_impl(req, persist_message=False, record_invocation=False)
        assert error.value.status_code == 422
        operation.assert_not_awaited()
    manager.set_provider.assert_not_called()
