"""Temporal dispatch and execution for in-app character messages."""

import asyncio
from contextlib import suppress
from time import time

from temporalio import activity
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import ApplicationError, WorkflowAlreadyStartedError

from repositories.character_chat_proactive import CharacterChatProactiveRepository
from workflows.chat_proactive import ChatProactiveWorkflow


class ChatProactiveActivities:
    def __init__(self, database):
        self.repository = CharacterChatProactiveRepository(database)

    @activity.defn(name="deliver_character_chat_proactive")
    async def deliver(self) -> int:
        try:
            return await asyncio.to_thread(self.repository.deliver_due)
        except Exception:
            pass
        raise ApplicationError("Character message delivery unavailable", type="ChatProactiveUnavailable")


async def dispatch_proactive(database, client, task_queue, minute):
    if not await asyncio.to_thread(CharacterChatProactiveRepository(database).has_due, time()):
        return
    with suppress(WorkflowAlreadyStartedError):
        await client.start_workflow(
            ChatProactiveWorkflow.run, id=f"character-chat-proactive-{task_queue}-{minute}",
            task_queue=task_queue, id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
        )
