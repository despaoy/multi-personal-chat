"""Activity implementation: reload private inputs; never return them to workflow history."""

import asyncio

from repositories.character_chat_enrichment import CharacterChatEnrichmentRepository


async def process_enrichment(turn_id, *, database, enrich, relationship, expected_revision=None, summary=None, embeddings=None):
    repo = CharacterChatEnrichmentRepository(database)
    loaded = await asyncio.to_thread(repo.load, turn_id, expected_revision)
    if loaded is None:
        return "inactive"
    session, turn = loaded
    history = await asyncio.to_thread(repo.chats.turns, session["owner_user_id"], session["id"])
    state = await enrich(turn["message"], session, turn["request_id"])
    state = await relationship(turn["message"], turn["reply"], {**session, "state": state}, turn["request_id"])
    if summary:
        state = await summary({**session, "state": state}, [*session["baseline"]["history"], *history], turn["request_id"])
    if any(state.get(field) == "failed" for field in ("memory_processing", "relationship_processing", "summary_processing")):
        raise RuntimeError("Auxiliary model processing failed")
    vector_batch = await asyncio.to_thread(embeddings, state) if embeddings else None
    applied = await asyncio.to_thread(repo.commit, session, turn, state, vector_batch)
    return "completed" if applied else "superseded"
