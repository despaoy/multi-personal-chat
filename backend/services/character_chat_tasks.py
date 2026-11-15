"""Application-owned waits: client disconnects do not cancel admitted chat jobs."""

import asyncio

from db.errors import RevisionConflictError
from inference.request_context import detached_request_context
from infra.concurrency_control import InferenceQueueFull


class CharacterChatTasks:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.pending = {}
        self.closed = False

    async def run(self, key, identity, factory):
        if self.closed:
            raise InferenceQueueFull("chat task owner is closed")
        existing = self.pending.get(key)
        if existing is not None:
            previous_identity, task = existing
            if previous_identity != identity:
                raise RevisionConflictError()
        else:
            if len(self.pending) >= self.capacity:
                raise InferenceQueueFull("chat task owner is full")
            task = asyncio.create_task(factory(), context=detached_request_context())
            self.pending[key] = (identity, task)

            def finished(completed):
                if self.pending.get(key, (None, None))[1] is completed:
                    self.pending.pop(key, None)
                # Retrieve failures even when every HTTP waiter disconnected.
                if not completed.cancelled():
                    completed.exception()

            task.add_done_callback(finished)
        return await asyncio.shield(task)

    async def shutdown(self):
        self.closed = True
        tasks = [task for _, task in self.pending.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.pending.clear()
