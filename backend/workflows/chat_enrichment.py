"""Deterministic orchestration containing identifiers and status values only."""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError


@workflow.defn
class ChatEnrichmentWorkflow:
    @workflow.run
    async def run(self, job: dict) -> str:
        try:
            return await workflow.execute_activity(
                "enrich_character_chat", job,
                start_to_close_timeout=timedelta(minutes=5),
                schedule_to_close_timeout=timedelta(minutes=20),
                retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=5)),
            )
        except ActivityError:
            # Persist terminal failure even if the database is temporarily unavailable.
            # The finalizer retries independently and carries no private model payload.
            return await workflow.execute_activity(
                "fail_character_chat_enrichment", job,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(initial_interval=timedelta(seconds=5)),
            )
