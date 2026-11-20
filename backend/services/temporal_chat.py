"""Temporal activity and transactional outbox dispatch adapters."""

import asyncio
import os
from contextlib import suppress
from functools import partial

from temporalio import activity
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import ApplicationError, WorkflowAlreadyStartedError

from infra.environment import read_bool
from repositories.character_chat_enrichment import CharacterChatEnrichmentRepository
from services.character_chat_enrichment import process_enrichment
from services.character_chat_memory import enrich_state
from services.character_chat_relationship import enrich_relationship
from services.character_chat_summary import enrich_summary
from services.character_chat_vectors import build_memory_vectors
from workflows.chat_enrichment import ChatEnrichmentWorkflow


class ChatEnrichmentActivities:
    def __init__(self, database):
        self.database = database

    @activity.defn(name="enrich_character_chat")
    async def enrich(self, job: dict) -> str:
        try:
            return await process_enrichment(
                job["turn_id"], database=self.database, expected_revision=job["expected_revision"],
                enrich=partial(enrich_state, database=self.database),
                relationship=partial(enrich_relationship, database=self.database),
                summary=partial(enrich_summary, database=self.database),
                embeddings=build_memory_vectors if read_bool(os.environ, "CHARACTER_CHAT_VECTORS_ENABLED", False) else None,
            )
        except Exception:
            # Do not persist SQL parameters, model payloads or private exceptions in history.
            pass
        raise ApplicationError("Background enrichment unavailable", type="ChatEnrichmentUnavailable")

    @activity.defn(name="fail_character_chat_enrichment")
    async def fail(self, job: dict) -> str:
        try:
            repo = CharacterChatEnrichmentRepository(self.database)
            return await asyncio.to_thread(repo.fail, job["turn_id"], job["expected_revision"])
        except Exception:
            pass
        raise ApplicationError("Background status unavailable", type="ChatEnrichmentStatusUnavailable")


async def dispatch_enrichment(database, client, task_queue):
    repo = CharacterChatEnrichmentRepository(database)
    jobs = await asyncio.to_thread(repo.pending)
    submitted = 0
    for job in jobs:
        turn_id = job["turn_id"]
        revision = job["expected_revision"]
        if not await asyncio.to_thread(repo.keep_active, turn_id, revision):
            continue
        # Covers the crash window after Temporal accepts but before DB acknowledgement.
        with suppress(WorkflowAlreadyStartedError):
            await client.start_workflow(
                ChatEnrichmentWorkflow.run, dict(job), id=f"character-enrichment-{turn_id}-{revision}",
                task_queue=task_queue, id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
            )
        await asyncio.to_thread(repo.submitted, turn_id, revision)
        submitted += 1
    return submitted
