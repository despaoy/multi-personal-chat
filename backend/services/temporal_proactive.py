"""Temporal dispatch and execution for in-app character messages."""

import asyncio
import logging
from contextlib import suppress
from time import time

from temporalio import activity
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import ApplicationError, WorkflowAlreadyStartedError

from repositories.character_chat_proactive import CharacterChatProactiveRepository
from repositories.character_chat_push import CharacterChatPushRepository
from services.character_web_push import WebPushConfig
from workflows.chat_proactive import ChatProactiveWorkflow

logger = logging.getLogger(__name__)


class ChatProactiveActivities:
    def __init__(self, database):
        self.repository = CharacterChatProactiveRepository(database)
        self.push = CharacterChatPushRepository(database)

    @activity.defn(name="deliver_character_chat_proactive")
    async def deliver(self) -> int:
        try:
            delivered = await asyncio.to_thread(self.repository.deliver_due)
        except Exception:
            delivered = None
        if delivered is None:
            raise ApplicationError("Character message delivery unavailable", type="ChatProactiveUnavailable")
        try:
            if sender := WebPushConfig.from_env():
                await asyncio.to_thread(self.push.deliver, sender)
        except Exception as exc:
            logger.warning("Browser reminder sweep unavailable type=%s", type(exc).__name__)
        return delivered


async def dispatch_proactive(database, client, task_queue, minute):
    now = time()
    if (
        not await asyncio.to_thread(CharacterChatProactiveRepository(database).has_due, now)
        and (WebPushConfig.from_env() is None
             or not await asyncio.to_thread(CharacterChatPushRepository(database).has_due, now))
    ):
        return
    with suppress(WorkflowAlreadyStartedError):
        await client.start_workflow(
            ChatProactiveWorkflow.run, id=f"character-chat-proactive-{task_queue}-{minute}",
            task_queue=task_queue, id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
        )
