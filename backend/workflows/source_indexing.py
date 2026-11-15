"""Source indexing orchestration; history contains identifiers, never original text."""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError


@workflow.defn
class SourceIndexWorkflow:
    @workflow.run
    async def run(self, job: dict) -> str:
        try:
            return await workflow.execute_activity(
                'index_character_sources', job,
                start_to_close_timeout=timedelta(minutes=15),
                schedule_to_close_timeout=timedelta(hours=1),
                retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=5)),
            )
        except ActivityError:
            return await workflow.execute_activity(
                'fail_character_source_index', job,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(initial_interval=timedelta(seconds=5)),
            )
