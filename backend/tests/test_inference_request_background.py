"""Durable and queued owners detach request scope while keeping logging context."""

import asyncio
from contextvars import ContextVar

from services import delivery_memory as dm
from services.turn_completion import TurnCompletionRuntime
from test_delivery_memory import deliver, generated

from db.database import SQLiteDB
from inference.request_context import InferenceRequestContext, get_request_context, use_request_context
from infra.concurrency_control import InferenceRuntime

OTHER_CONTEXT = ContextVar("background_scope_test_log_context", default="outside")


async def test_delivery_owner_drops_request_but_keeps_other_context(tmp_path, monkeypatch):
    parent_binding = get_request_context()
    bound = InferenceRequestContext("openai_compat", object(), lambda text: len(text))
    database = SQLiteDB(tmp_path / "background-owner.sqlite")
    key, owner, stored = generated(database)
    deliver(database, key, owner, stored)
    runtime = TurnCompletionRuntime(1)
    monkeypatch.setattr(dm, "get_turn_completion_runtime", lambda _owner_database=None: runtime)
    entered = asyncio.Event()
    release = asyncio.Event()
    observed = []
    worker = None

    async def observe_job(self, record, job_runtime, slot):
        assert record["receipt_key"] == key and job_runtime is runtime
        assert slot.state == "reserved"
        observed.append((get_request_context(), OTHER_CONTEXT.get()))
        entered.set()
        # Only the semantic body is controlled. The durable owner still polls
        # real SQLite, obtains a reservation, and executes normal cleanup.
        await release.wait()

    monkeypatch.setattr(dm.DeliveryMemoryWorker, "_process", observe_job)
    other_token = OTHER_CONTEXT.set("synthetic-delivery-log-context")
    try:
        with use_request_context(bound):
            worker = dm.DeliveryMemoryWorker(database)
            await asyncio.wait_for(entered.wait(), 1)
            assert get_request_context() is bound
        assert get_request_context() is parent_binding
        assert observed == [(None, "synthetic-delivery-log-context")]
    finally:
        release.set()
        if worker is not None:
            await worker.shutdown(timeout=1)
        await runtime.shutdown(timeout=1)
        OTHER_CONTEXT.reset(other_token)
        database.close_connection()
    assert worker.task.done() and runtime.reserved == runtime.active == 0


async def test_queue_owners_detach_submitter_and_actual_job_binds_its_own_scope(monkeypatch):
    monkeypatch.setenv("INFERENCE_WORKERS", "1")
    monkeypatch.setenv("INFERENCE_QUEUE_TIMEOUT", "5")
    runtime = InferenceRuntime()
    outer = InferenceRequestContext("openai_compat", object(), len)
    job = InferenceRequestContext("vllm", None, len)
    observations = {}
    maintained = asyncio.Event()
    worker, maintenance = runtime._worker, runtime._maintenance_loop

    async def observe_worker(index):
        observations["worker"] = (get_request_context(), OTHER_CONTEXT.get())
        await worker(index)

    async def observe_maintenance():
        observations["maintenance"] = (get_request_context(), OTHER_CONTEXT.get())
        maintained.set()
        await maintenance()

    async def generate_job():
        assert get_request_context() is None
        with use_request_context(job):
            await asyncio.sleep(0)
            assert get_request_context() is job
        assert get_request_context() is None
        return "completed-owned-job"

    monkeypatch.setattr(runtime, "_worker", observe_worker)
    monkeypatch.setattr(runtime, "_maintenance_loop", observe_maintenance)
    token = OTHER_CONTEXT.set("preserved-queue-log-context")
    try:
        with use_request_context(outer):
            assert await runtime.submit(generate_job, session_id="synthetic-queue-session", priority=10) == "completed-owned-job"
            await asyncio.wait_for(maintained.wait(), 1)
            assert get_request_context() is outer
        assert get_request_context() is None
        assert observations == {
            "worker": (None, "preserved-queue-log-context"),
            "maintenance": (None, "preserved-queue-log-context"),
        }
    finally:
        await runtime.shutdown()
        OTHER_CONTEXT.reset(token)
