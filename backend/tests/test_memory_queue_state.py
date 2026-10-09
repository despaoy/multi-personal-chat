"""Capacity follows real buffered, queued and processing jobs."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import UserScope


async def test_capacity_survives_buffer_flush_processing_and_release(monkeypatch):
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "test", queue_size=2, idle_seconds=3600),
        completion=SimpleNamespace(close=AsyncMock()))
    started, release = asyncio.Event(), asyncio.Event()
    processed = []
    async def process(job):
        started.set()
        await release.wait()
        processed.append(job.source_message_id)
    monkeypatch.setattr(worker, "_process_job", process)
    values = dict(repository=object(), character_id="test", user_scope=UserScope("web", "test", "u", "u", "private"),
                  message="我喜欢红茶", rule_hints=[])
    try:
        assert worker.schedule(**values, immediate=False, source_message_id="buffered")
        assert worker.status.buffered == worker._inflight == 1
        assert worker.schedule(**values, immediate=True, source_message_id="hot")
        assert worker.status.queued == worker._inflight == 2
        assert not worker.schedule(**values, immediate=True, source_message_id="refused")
        await asyncio.wait_for(started.wait(), 1)
        assert worker.status.processing == worker.status.queued == 1
        assert worker._inflight == 2
        release.set()
        assert await worker.flush_memory(timeout=1)
        assert worker._inflight == 0
        assert processed == ["buffered", "hot"]
        assert worker.schedule(**values, immediate=True, source_message_id="next")
        assert await worker.flush_memory(timeout=1)
        assert processed == ["buffered", "hot", "next"]
    finally:
        release.set()
        await worker.shutdown(timeout=1)
    assert worker._inflight == 0
