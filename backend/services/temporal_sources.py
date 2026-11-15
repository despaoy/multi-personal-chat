"""Durable source outbox delivery and private-payload-free Temporal activities."""

import asyncio
from contextlib import suppress

from temporalio import activity
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import ApplicationError, WorkflowAlreadyStartedError

from db.errors import RevisionConflictError
from repositories.character_library import CharacterNotFoundError
from repositories.character_sources import CharacterSourceRepository
from services.character_source_indexing import publish_configured_source_index
from workflows.source_indexing import SourceIndexWorkflow


class SourceIndexActivities:
    def __init__(self, database):
        self.database = database

    @activity.defn(name='index_character_sources')
    async def index(self, job: dict) -> str:
        try:
            repo = CharacterSourceRepository(self.database)
            corpus = await asyncio.to_thread(repo.get, job['owner'], job['corpus_id'])
            if corpus['index_attempt'] != job['attempt'] or corpus['index_status'] == 'failed':
                return 'obsolete'
            if corpus['index_status'] == 'completed':
                return 'completed'
            await publish_configured_source_index(job['owner'], job['corpus_id'], database=self.database,
                                                  expected_attempt=job['attempt'])
            return 'completed'
        except (CharacterNotFoundError, RevisionConflictError):
            return 'obsolete'
        except Exception:
            pass
        raise ApplicationError('Source indexing unavailable', type='SourceIndexUnavailable')

    @activity.defn(name='fail_character_source_index')
    async def fail(self, job: dict) -> str:
        try:
            repo = CharacterSourceRepository(self.database)
            await asyncio.to_thread(repo.set_index_job_status, job['owner'], job['corpus_id'], job['attempt'], 'failed')
            return 'failed'
        except (CharacterNotFoundError, RevisionConflictError):
            return 'obsolete'
        except Exception:
            pass
        raise ApplicationError('Source indexing status unavailable', type='SourceIndexStatusUnavailable')


async def dispatch_sources(database, client, task_queue):
    repo = CharacterSourceRepository(database)
    jobs = await asyncio.to_thread(repo.pending_index_jobs)
    for job in jobs:
        with suppress(WorkflowAlreadyStartedError):
            await client.start_workflow(
                SourceIndexWorkflow.run, dict(job), id=f"character-source-index-{job['corpus_id']}-{job['attempt']}",
                task_queue=task_queue, id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
            )
        # A fast worker may finish, fail, or observe revocation before this acknowledgement.
        with suppress(CharacterNotFoundError, RevisionConflictError):
            await asyncio.to_thread(repo.set_index_job_status, job['owner'], job['corpus_id'], job['attempt'], 'submitted')
    return len(jobs)
