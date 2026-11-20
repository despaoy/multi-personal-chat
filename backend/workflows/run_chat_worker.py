"""Run the chat enrichment worker and dispatch durable database outbox entries."""

import asyncio
import logging
import os
from functools import partial
from time import time

from temporalio.client import Client
from temporalio.worker import Worker

from app.env import load_backend_env
from infra.environment import read_bool
from infra.resource_pool import HttpClientPool
from services.character_chat import complete_with_provider
from services.temporal_chat import ChatEnrichmentActivities, dispatch_enrichment
from services.temporal_proactive import ChatProactiveActivities, dispatch_proactive
from services.temporal_sources import SourceIndexActivities, dispatch_sources
from workflows.chat_enrichment import ChatEnrichmentWorkflow
from workflows.chat_proactive import ChatProactiveWorkflow
from workflows.source_indexing import SourceIndexWorkflow

logger = logging.getLogger(__name__)


async def run(database, client, task_queue, http_client_pool=None):
    activities = ChatEnrichmentActivities(database)
    sources = SourceIndexActivities(database)
    proactive = ChatProactiveActivities(database, partial(
        complete_with_provider, database=database, http_client_pool=http_client_pool, max_reply_tokens=192,
    ))
    proactive_minute = None
    async with Worker(
        client, task_queue=task_queue, workflows=[ChatEnrichmentWorkflow, SourceIndexWorkflow, ChatProactiveWorkflow],
        activities=[activities.enrich, activities.fail, sources.index, sources.fail, proactive.deliver],
        max_concurrent_activities=4,
    ):
        while True:
            try:
                await dispatch_enrichment(database, client, task_queue)
            except Exception as exc:
                logger.warning("Enrichment dispatch unavailable: %s", type(exc).__name__)
            if read_bool(os.environ, 'CHARACTER_SOURCE_INDEXING_ENABLED', False):
                try:
                    await dispatch_sources(database, client, task_queue)
                except Exception as exc:
                    logger.warning('Source dispatch unavailable: %s', type(exc).__name__)
            minute = int(time() // 60)
            if minute != proactive_minute and read_bool(os.environ, "CHARACTER_CHAT_PROACTIVE_ENABLED", False):
                try:
                    await dispatch_proactive(database, client, task_queue, minute)
                    proactive_minute = minute
                except Exception as exc:
                    logger.warning("Proactive dispatch unavailable: %s", type(exc).__name__)
            await asyncio.sleep(2)


async def main():
    load_backend_env()
    from db.adapter import db
    from infra.environment import read_bool

    if hasattr(db, "init"):
        await asyncio.to_thread(db.init)
    pool = HttpClientPool()
    try:
        client = await Client.connect(
            os.getenv("TEMPORAL_ADDRESS", "127.0.0.1:7233"),
            namespace=os.getenv("TEMPORAL_NAMESPACE", "default"),
            tls=read_bool(os.environ, "TEMPORAL_TLS", False),
            api_key=os.getenv("TEMPORAL_API_KEY") or None,
        )
        await run(db, client, os.getenv("TEMPORAL_CHAT_TASK_QUEUE", "character-chat-enrichment"), pool)
    finally:
        try:
            await pool.close()
        finally:
            close = getattr(db, "close", None) or getattr(db, "close_connection", None)
            if close:
                await asyncio.to_thread(close)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
