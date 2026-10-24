from __future__ import annotations

import asyncio
import threading
import time

import pytest

from app.readiness import ReadinessProbe, ReadinessProbeTimeout


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_database_and_model_refresh():
    calls = {"database": 0, "model": 0}

    def database_check():
        calls["database"] += 1
        time.sleep(0.03)

    async def model_check():
        calls["model"] += 1
        await asyncio.sleep(0.03)
        return True

    probe = ReadinessProbe(
        database_check=database_check,
        model_check=model_check,
        model_required=True,
    )
    try:
        results = await asyncio.gather(*(probe.get() for _ in range(50)))
    finally:
        await probe.shutdown()

    assert calls == {"database": 1, "model": 1}
    assert all(result["ready"] for result in results)
    assert all(result["details"]["rag"] == "not_probed" for result in results)


@pytest.mark.asyncio
async def test_waiter_timeout_does_not_cancel_shared_refresh():
    calls = 0
    release = asyncio.Event()

    async def database_check():
        nonlocal calls
        calls += 1
        await release.wait()
        return True

    probe = ReadinessProbe(
        database_check=database_check,
        check_timeout=1.0,
        wait_timeout=0.02,
    )
    try:
        with pytest.raises(ReadinessProbeTimeout):
            await probe.get()

        release.set()
        await asyncio.sleep(0)
        result = await probe.get()
    finally:
        await probe.shutdown()

    assert result["ready"] is True
    assert calls == 1


@pytest.mark.asyncio
async def test_sync_timeout_keeps_bounded_probe_slot_until_work_finishes():
    calls = 0
    clock = [0.0]
    release = threading.Event()

    def database_check():
        nonlocal calls
        calls += 1
        release.wait(timeout=1.0)

    probe = ReadinessProbe(
        database_check=database_check,
        failure_ttl=0.01,
        check_timeout=0.02,
        wait_timeout=0.1,
        clock=lambda: clock[0],
    )
    try:
        first = await probe.get()
        assert first["details"]["database"] == "BlockingWorkTimeout"

        clock[0] += 1.0
        second = await probe.get()
        assert second["details"]["database"] == "BlockingWorkTimeout"

        clock[0] += 1.0
        third = await probe.get()
        assert third["details"]["database"] == "BlockingWorkRejected"
        assert calls == 2
    finally:
        release.set()
        await probe.shutdown()


@pytest.mark.asyncio
async def test_success_and_failure_use_different_cache_ttls():
    clock = [0.0]
    healthy = [True]
    calls = 0

    async def database_check():
        nonlocal calls
        calls += 1
        return healthy[0]

    probe = ReadinessProbe(
        database_check=database_check,
        success_ttl=5.0,
        failure_ttl=1.0,
        clock=lambda: clock[0],
    )
    try:
        assert (await probe.get())["ready"] is True
        healthy[0] = False
        clock[0] = 4.0
        assert (await probe.get())["ready"] is True
        assert calls == 1

        clock[0] = 6.0
        assert (await probe.get())["ready"] is False
        assert calls == 2

        healthy[0] = True
        clock[0] = 6.5
        assert (await probe.get())["ready"] is False
        assert calls == 2

        clock[0] = 7.1
        assert (await probe.get())["ready"] is True
        assert calls == 3
    finally:
        await probe.shutdown()

@pytest.mark.asyncio
async def test_database_and_model_sync_checks_have_independent_capacity():
    calls = {"database": 0, "model": 0}

    def database_check():
        calls["database"] += 1
        time.sleep(0.03)
        return True

    def model_check():
        calls["model"] += 1
        time.sleep(0.03)
        return True

    probe = ReadinessProbe(
        database_check=database_check,
        model_check=model_check,
        model_required=True,
        check_timeout=0.2,
        wait_timeout=0.3,
    )
    try:
        result = await probe.get()
    finally:
        await probe.shutdown()

    assert result["ready"] is True
    assert result["deps"] == {"database": True, "model": True}
    assert calls == {"database": 1, "model": 1}

@pytest.mark.asyncio
async def test_application_close_waits_for_readiness_cancellation_cleanup():
    from app.main import _close_resource

    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    finished = asyncio.Event()

    async def check():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            finished.set()

    probe = ReadinessProbe(database_check=check, check_timeout=5, wait_timeout=5)
    waiter = asyncio.create_task(probe.get())
    closer = None
    try:
        await asyncio.wait_for(started.wait(), 1)
        closer = asyncio.create_task(_close_resource("readiness", probe, "shutdown"))
        await asyncio.wait_for(cleaning.wait(), 1)
        # Give the original synchronous close enough loop turns to return.
        for _ in range(10):
            await asyncio.sleep(0)
        assert not closer.done(), "shutdown returned before readiness cleanup finished"
        release.set()
        await asyncio.wait_for(closer, 1)
        assert finished.is_set() and probe._refresh_task.done()
        with pytest.raises(asyncio.CancelledError):
            await waiter
    finally:
        release.set()
        await asyncio.gather(waiter, *([closer] if closer else []), return_exceptions=True)
