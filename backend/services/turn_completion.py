"""Bound generation-to-completion ownership before expensive stateful work."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

logger = logging.getLogger(__name__)


class CompletionUnavailable(RuntimeError):
    """No completion capacity is available before stateful generation."""


class CompletionReservation:
    """One runtime-owned slot, released or consumed exactly once."""

    def __init__(self, runtime: TurnCompletionRuntime):
        self.runtime = runtime
        self.state = "reserved"

    def release(self) -> None:
        if self.state == "reserved":
            self.runtime._reservations.discard(self)
            self.state = "released"


class TurnCompletionRuntime:
    """Own accepted completions and their pre-generation reservations."""

    def __init__(self, capacity: int = 128):
        if capacity < 1:
            raise ValueError("Completion capacity must be positive")
        self.loop = asyncio.get_running_loop()
        self.capacity = capacity
        self.closed = False
        self._tasks: set[asyncio.Task[Any]] = set()
        self._reservations: set[CompletionReservation] = set()
        self.completed = self.failed = self.cancelled = 0

    @property
    def active(self) -> int:
        return sum(not task.done() for task in self._tasks)

    @property
    def reserved(self) -> int:
        return len(self._reservations)

    def reserve(self) -> CompletionReservation:
        if asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("Completion runtime belongs to another loop")
        if self.closed or self.active + self.reserved >= self.capacity:
            raise CompletionUnavailable("Completion runtime unavailable")
        slot = CompletionReservation(self)
        self._reservations.add(slot)
        return slot

    def _finished(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            self.cancelled += 1
        elif (error := task.exception()) is not None:
            self.failed += 1
            logger.warning("角色回写任务结束：%s", type(error).__name__)
        else:
            self.completed += 1

    async def run(
        self,
        work: Callable[[], Coroutine[Any, Any, Any]],
        *,
        timeout: float | None,
        reservation: CompletionReservation | None = None,
    ) -> Any:
        if asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("Completion runtime belongs to another loop")
        if self.closed:
            raise CompletionUnavailable("Completion runtime unavailable")
        slot = reservation if reservation is not None else self.reserve()
        if slot.runtime is not self or slot.state != "reserved" or slot not in self._reservations:
            raise RuntimeError("Invalid completion reservation")
        try:
            task = self.loop.create_task(work(), name="character-turn-completion")
        except BaseException:
            slot.release()
            raise
        # No await between consuming the reservation and owning its task.
        self._reservations.remove(slot)
        slot.state = "consumed"
        self._tasks.add(task)
        task.add_done_callback(self._finished)
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    async def shutdown(self, timeout: float = 35.0) -> None:
        if asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("Completion runtime belongs to another loop")
        self.closed = True
        for slot in tuple(self._reservations):
            slot.release()
        pending = {task for task in self._tasks if not task.done()}
        if pending:
            _, pending = await asyncio.wait(pending, timeout=max(0.0, timeout))
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.sleep(0)


_runtime: TurnCompletionRuntime | None = None


def get_turn_completion_runtime() -> TurnCompletionRuntime:
    global _runtime
    loop = asyncio.get_running_loop()
    if _runtime is None or (_runtime.loop is not loop and not _runtime.active and not _runtime.reserved):
        _runtime = TurnCompletionRuntime()
    return _runtime


def start_turn_completions() -> None:
    global _runtime
    runtime = get_turn_completion_runtime()
    if runtime.closed:
        if runtime.active or runtime.reserved:
            raise RuntimeError("Completion runtime still draining")
        _runtime = TurnCompletionRuntime()


async def shutdown_turn_completions() -> None:
    runtime = _runtime
    if runtime is not None:
        await runtime.shutdown()
