"""Own accepted turn writes beyond the response deadline until shutdown."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

logger = logging.getLogger(__name__)


class TurnCompletionRuntime:
    """Bound task ownership; caller timeout does not cancel an accepted write."""

    def __init__(self, capacity: int = 128):
        if capacity < 1:
            raise ValueError("Completion capacity must be positive")
        self.loop = asyncio.get_running_loop()
        self.capacity = capacity
        self.closed = False
        self._tasks: set[asyncio.Task[Any]] = set()
        self.completed = self.failed = self.cancelled = 0

    @property
    def active(self) -> int:
        return sum(not task.done() for task in self._tasks)

    def _finished(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            self.cancelled += 1
        elif (error := task.exception()) is not None:
            self.failed += 1
            # No prompt, owner identity, or provider details in diagnostics.
            logger.warning("角色回写任务结束：%s", type(error).__name__)
        else:
            self.completed += 1

    async def run(self, work: Callable[[], Coroutine[Any, Any, Any]], *, timeout: float) -> Any:
        if asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("Completion runtime belongs to another loop")
        if self.closed or self.active >= self.capacity:
            raise RuntimeError("Completion runtime unavailable")
        task = self.loop.create_task(work(), name="character-turn-completion")
        self._tasks.add(task)
        task.add_done_callback(self._finished)
        # Both response timeout and caller cancellation leave the exact task
        # owned here. No second capture, receipt, generation, or retry is made.
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    async def shutdown(self, timeout: float = 35.0) -> None:
        if asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("Completion runtime belongs to another loop")
        self.closed = True
        pending = {task for task in self._tasks if not task.done()}
        if pending:
            _, pending = await asyncio.wait(pending, timeout=max(0.0, timeout))
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        # Deliver done callbacks before reporting a fully drained registry.
        await asyncio.sleep(0)


_runtime: TurnCompletionRuntime | None = None


def get_turn_completion_runtime() -> TurnCompletionRuntime:
    global _runtime
    loop = asyncio.get_running_loop()
    if _runtime is None or (_runtime.loop is not loop and not _runtime.active):
        _runtime = TurnCompletionRuntime()
    return _runtime


def start_turn_completions() -> None:
    global _runtime
    runtime = get_turn_completion_runtime()
    if runtime.closed:
        if runtime.active:
            raise RuntimeError("Completion runtime still draining")
        _runtime = TurnCompletionRuntime()


async def shutdown_turn_completions() -> None:
    runtime = _runtime
    if runtime is not None:
        # Retain the closed registry through dependency teardown. A request
        # finishing generation later must not silently reopen write intake.
        await runtime.shutdown()
