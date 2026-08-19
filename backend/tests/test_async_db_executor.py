"""Async DB executor should not block lightweight async work."""

from __future__ import annotations

import asyncio
import time

import pytest


@pytest.mark.asyncio
async def test_slow_db_call_does_not_block_lightweight_async():
    from infra.db_executor import run_db

    async def light_ping():
        return "pong"

    def slow_db():
        time.sleep(0.2)
        return "db-done"

    slow_task = asyncio.create_task(run_db(slow_db))
    await asyncio.sleep(0.01)
    assert await light_ping() == "pong"
    assert await slow_task == "db-done"
