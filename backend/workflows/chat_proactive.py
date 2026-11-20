"""Run a durable sweep without putting session contents in workflow history."""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy


@workflow.defn
class ChatProactiveWorkflow:
    @workflow.run
    async def run(self) -> int:
        return await workflow.execute_activity(
            "deliver_character_chat_proactive", start_to_close_timeout=timedelta(minutes=2),
            schedule_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=5)),
        )
