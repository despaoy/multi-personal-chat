"""Changed completion deadline and its task lifecycle; no provider calls."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from services.turn_completion import TurnCompletionRuntime


@pytest.mark.asyncio
async def test_response_deadline_retains_exact_write_and_later_stage(monkeypatch):
    from api import generate as gen
    from db.schemas import MessageRequest
    from services import turn_completion

    runtime = TurnCompletionRuntime()
    monkeypatch.setattr(turn_completion, "get_turn_completion_runtime", lambda: runtime)
    monkeypatch.setattr(gen, "_DB_WRITE_TIMEOUT", 0.005)
    release = asyncio.Event()
    seen = []

    async def complete(prepared, turn, reply, *, source_message_id):
        seen.append((turn.message, turn.received_at, reply, source_message_id))
        await release.wait()
        seen.append("captured_then_scheduled")
        return SimpleNamespace(source_capture="recorded")

    at = datetime.now(timezone.utc)
    request = MessageRequest(message="我喜欢青绿色油墨做木版套色。", sourceMessageId="owned-original")
    request._source_received_at = at
    warning = await gen._complete_character_turn(
        SimpleNamespace(character_id="role"),
        request,
        "original-answer",
        character_service=SimpleNamespace(complete_turn=complete),
    )
    assert "尚未确认" in warning and "保存失败" not in warning
    assert runtime.active == 1 and seen == [(request.message, at, "original-answer", "owned-original")]
    release.set()
    await runtime.shutdown(timeout=1)
    assert seen[-1] == "captured_then_scheduled" and len(seen) == 2
    assert runtime.completed == 1 and runtime.cancelled == runtime.failed == runtime.active == 0


@pytest.mark.asyncio
async def test_caller_cancellation_propagates_without_losing_owned_write():
    runtime = TurnCompletionRuntime()
    started, release = asyncio.Event(), asyncio.Event()
    finished = []

    async def work():
        started.set()
        await release.wait()
        finished.append("accepted-write")

    caller = asyncio.create_task(runtime.run(work, timeout=1))
    await started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert runtime.active == 1 and not finished
    release.set()
    await runtime.shutdown(timeout=1)
    assert finished == ["accepted-write"] and runtime.cancelled == 0


@pytest.mark.asyncio
async def test_capacity_rejects_before_creating_new_work():
    runtime = TurnCompletionRuntime(capacity=1)
    release = asyncio.Event()
    with pytest.raises(TimeoutError):
        await runtime.run(release.wait, timeout=0.005)
    factory_calls = []

    def forbidden():
        factory_calls.append(True)
        return release.wait()

    with pytest.raises(RuntimeError, match="unavailable"):
        await runtime.run(forbidden, timeout=1)
    assert not factory_calls and runtime.active == 1
    release.set()
    await runtime.shutdown(timeout=1)


@pytest.mark.asyncio
async def test_late_error_is_consumed_and_releases_capacity():
    runtime = TurnCompletionRuntime(capacity=1)
    release = asyncio.Event()

    async def work():
        await release.wait()
        raise RuntimeError("synthetic-late-failure")

    with pytest.raises(TimeoutError):
        await runtime.run(work, timeout=0.005)
    release.set()
    await runtime.shutdown(timeout=1)
    assert runtime.failed == 1 and runtime.active == runtime.cancelled == 0


@pytest.mark.asyncio
async def test_shutdown_deadline_cancels_and_refuses_new_tasks():
    runtime = TurnCompletionRuntime()
    cancelled = asyncio.Event()

    async def work():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with pytest.raises(TimeoutError):
        await runtime.run(work, timeout=0.005)
    await runtime.shutdown(timeout=0.005)
    assert cancelled.is_set() and runtime.cancelled == 1 and runtime.active == 0
    with pytest.raises(RuntimeError, match="unavailable"):
        await runtime.run(work, timeout=1)


@pytest.mark.asyncio
async def test_shutdown_keeps_dependencies_open_for_followup():
    runtime = TurnCompletionRuntime()
    release = asyncio.Event()
    dependency = dict(open=True)
    stages = []

    async def work():
        await release.wait()
        assert dependency["open"]
        stages.append("semantic-job-enqueued")

    with pytest.raises(TimeoutError):
        await runtime.run(work, timeout=0.005)
    closing = asyncio.create_task(runtime.shutdown(timeout=1))
    await asyncio.sleep(0)
    assert runtime.closed and not closing.done()
    release.set()
    await closing
    dependency["open"] = False
    assert stages == ["semantic-job-enqueued"] and runtime.completed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["source", "owner"])
async def test_late_sqlite_capture_cannot_restore_deleted_source(tmp_path, scope):
    from db.database import SQLiteDB

    runtime = TurnCompletionRuntime()
    database = SQLiteDB(tmp_path / "late.sqlite")
    fields = ("role", "web", "late-guard", "owner", "private", "owner")
    at = datetime.now(timezone.utc) - timedelta(seconds=1)
    body = "我喜欢青绿色油墨做木版套色。"
    receipt = database.reserve_memory_source(*fields, source_message_id="late-original", body=body, observed_at=at)
    assert receipt["status"] == "pending" and receipt["observed_at"] is not None
    stamp = datetime.fromisoformat(receipt["observed_at"])
    release = asyncio.Event()
    captured = []

    async def work():
        await release.wait()
        captured.append(
            await asyncio.to_thread(
                database.capture_memory_source, *fields, source_message_id="late-original", body=body, observed_at=stamp
            )
        )

    with pytest.raises(TimeoutError):
        await runtime.run(work, timeout=0.005)
    if scope == "source":
        database.erase_unlinked_memory_sources(*fields, source_message_ids=("late-original",))
    else:
        database.clear_character_memories(*fields)
    release.set()
    await runtime.shutdown(timeout=1)
    assert captured == ["revoked" if scope == "source" else "stale"]
    assert database.list_memory_sources(*fields) == []
    database.close_connection()


@pytest.mark.asyncio
async def test_global_shutdown_does_not_reopen_during_dependency_teardown(monkeypatch):
    from services import turn_completion

    runtime = TurnCompletionRuntime()
    monkeypatch.setattr(turn_completion, "_runtime", runtime)
    await turn_completion.shutdown_turn_completions()
    assert turn_completion.get_turn_completion_runtime() is runtime and runtime.closed
    with pytest.raises(RuntimeError, match="unavailable"):
        await turn_completion.get_turn_completion_runtime().run(asyncio.Event().wait, timeout=1)
    assert runtime.active == 0


@pytest.mark.asyncio
async def test_explicit_start_reopens_only_drained_lifecycle(monkeypatch):
    from services import turn_completion

    runtime = TurnCompletionRuntime()
    monkeypatch.setattr(turn_completion, "_runtime", runtime)
    await turn_completion.shutdown_turn_completions()
    turn_completion.start_turn_completions()
    reopened = turn_completion.get_turn_completion_runtime()
    assert reopened is not runtime and not reopened.closed
    done = []

    async def work():
        done.append(True)

    await reopened.run(work, timeout=1)
    await turn_completion.shutdown_turn_completions()
    assert done == [True] and reopened.closed and reopened.active == 0
