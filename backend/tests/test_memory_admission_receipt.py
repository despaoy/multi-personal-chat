"""Unaccepted memory jobs have their own terminal reason, never another job's state."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import UserScope
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

MESSAGE = "请记住，我的专业是物理学，12月5日14点在二楼办理登记；须本人携身份证和书面确认，未确认不得付款。"


@pytest.mark.parametrize("interface", ["supplied_receipt", "waiter"])
@pytest.mark.parametrize("reason", ["closed", "disabled", "non_user_source", "write_gate", "capacity"])
async def test_rejected_job_has_exact_terminal_reason_and_never_runs_model(tmp_path, monkeypatch, interface, reason):
    repository = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "source.sqlite"))
    capture = AsyncMock(wraps=repository.capture_source)
    monkeypatch.setattr(repository, "capture_source", capture)
    entered, release = asyncio.Event(), asyncio.Event()

    async def complete(messages):
        entered.set()
        await release.wait()
        return '{"memories":[]}'

    completion = SimpleNamespace(complete=AsyncMock(side_effect=complete), close=AsyncMock())
    worker = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(reason != "disabled", "unused", "fixture", queue_size=1), completion=completion
    )
    active_receipt = None
    values = dict(
        repository=repository,
        character_id="role",
        user_scope=UserScope("test", "admission", "bob", "bob", "private"),
        message=MESSAGE,
        rule_hints=(),
        source_message_id="rejected-source",
    )
    try:
        if reason == "closed":
            await worker.shutdown(timeout=1)
        elif reason == "non_user_source":
            values["source_type"] = "assistant"
        elif reason == "write_gate":
            values["message"] = "请只分析这些完整事实：我的专业是物理学。不要新增、替换或删除记忆。"
        elif reason == "capacity":
            active_receipt = asyncio.get_running_loop().create_future()
            assert worker.schedule(
                **{
                    **values,
                    "source_message_id": "accepted-source",
                    "user_scope": UserScope("test", "admission", "alice", "alice", "private"),
                },
                immediate=True,
                receipt=active_receipt,
            )
            await asyncio.wait_for(entered.wait(), timeout=2)
        skipped = worker.status.skipped
        if interface == "supplied_receipt":
            receipt = asyncio.get_running_loop().create_future()
            assert worker.schedule(**values, receipt=receipt) is False
            assert receipt.done()
            result = receipt.result()
        else:
            result = await worker.schedule_and_wait(**values)
        assert result == dict(status="not_scheduled", reason=reason, stage="admission", accepted=0, persisted=0)
        assert worker.status.skipped == skipped + 1
        assert worker.status.last_outcome == (reason if reason in {"closed", "capacity"} else "skipped")
        assert capture.await_count == completion.complete.await_count == int(reason == "capacity")
        assert await repository.list_memory_records("role", values["user_scope"]) == []
        if active_receipt is not None:
            assert not active_receipt.done()
            release.set()
            accepted = await asyncio.wait_for(asyncio.shield(active_receipt), timeout=2)
            assert accepted["source_message_id"] == "accepted-source" and accepted["source_capture"] == "recorded"
            assert accepted["status"] == "no_change" and accepted["persisted"] == 0
            assert result["reason"] == "capacity" and result["status"] == "not_scheduled"
    finally:
        release.set()
        if not worker._closed:
            await worker.shutdown(timeout=1)


async def test_rejected_job_does_not_resurrect_cancelled_caller_receipt():
    completion = SimpleNamespace(complete=AsyncMock(), close=AsyncMock())
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(False, "unused", "fixture"), completion=completion)
    receipt = asyncio.get_running_loop().create_future()
    receipt.cancel()
    try:
        assert (
            worker.schedule(
                repository=object(),
                character_id="role",
                user_scope=UserScope("test", "admission", "bob", "bob", "private"),
                message=MESSAGE,
                rule_hints=(),
                source_message_id="cancelled",
                receipt=receipt,
            )
            is False
        )
        assert receipt.cancelled()
        completion.complete.assert_not_awaited()
    finally:
        await worker.shutdown(timeout=1)
