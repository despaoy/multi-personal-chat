"""Explicit real-extension capability check in an isolated PostgreSQL cluster."""

import asyncio
import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.mark.skipif(os.getenv("RUN_PGVECTOR_INTEGRATION") != "1", reason="isolated pgvector required")
def test_pgvector_loads_and_ranks_cosine_neighbors():
    async def exercise():
        engine = create_async_engine(os.environ["DATABASE_URL"])
        try:
            async with engine.begin() as connection:
                await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                version = await connection.scalar(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
                assert version == "0.8.7"
                await connection.execute(text("CREATE TEMP TABLE vector_probe (id INTEGER, embedding vector(3)) ON COMMIT DROP"))
                await connection.execute(text("INSERT INTO vector_probe VALUES (1,'[1,0,0]'), (2,'[0,1,0]')"))
                result = await connection.execute(text("SELECT id FROM vector_probe ORDER BY embedding <=> CAST(:query AS vector)"),
                                                  {"query": "[0.9,0.1,0]"})
                assert list(result.scalars()) == [1, 2]
        finally:
            await engine.dispose()
    asyncio.run(exercise())


@pytest.mark.skipif(os.getenv("RUN_PGVECTOR_INTEGRATION") != "1", reason="isolated pgvector required")
def test_private_memory_vectors_cannot_survive_correction_or_cross_accounts():
    from character.library import CharacterSettings
    from db.errors import RevisionConflictError
    from db.pg_database import PgDatabase, SyncPgAdapter
    from repositories.character_chat import CharacterChatRepository
    from repositories.character_chat_vectors import CharacterChatVectorRepository
    from repositories.character_library import CharacterNotFoundError

    db = SyncPgAdapter(PgDatabase(os.environ["DATABASE_URL"]))
    try:
        chats = CharacterChatRepository(db)
        vectors = CharacterChatVectorRepository(db)
        char = chats.characters.create("vector-owner", CharacterSettings(name="旅人", persona="谨慎", greeting="你好"))
        session = chats.create("vector-owner", char["id"], "向量验收")
        session = chats.manage_memory("vector-owner", session["id"], 1, content="喜欢茶")
        memory_id = session["state"]["memories"][0]["id"]
        vectors.replace(session, "test-embedding-v1", {memory_id: [1.0, 0.0, 0.0]})
        assert vectors.search("vector-owner", session["id"], "test-embedding-v1", [1, 0, 0])[0]["content"] == "喜欢茶"
        assert vectors.search("vector-owner", session["id"], "other-model", [1, 0, 0]) == []
        with pytest.raises(CharacterNotFoundError):
            vectors.search("other-owner", session["id"], "test-embedding-v1", [1, 0, 0])
        current = chats.manage_memory("vector-owner", session["id"], session["revision"], memory_id=memory_id, content="喜欢咖啡")
        assert vectors.search("vector-owner", session["id"], "test-embedding-v1", [1, 0, 0]) == []
        with pytest.raises(RevisionConflictError):
            vectors.replace(session, "test-embedding-v1", {memory_id: [1, 0, 0]})
        vectors.replace(current, "test-embedding-v1", {memory_id: [0, 1, 0]})
        assert vectors.search("vector-owner", session["id"], "test-embedding-v1", [0, 1, 0])[0]["content"] == "喜欢咖啡"
        from sqlalchemy.exc import DBAPIError

        from repositories.character_chat_enrichment import CharacterChatEnrichmentRepository

        turn = chats.commit_turn(current, "enrich", "继续聊", "好的", "test", enqueue_enrichment=True)
        jobs = CharacterChatEnrichmentRepository(db)
        loaded, source = jobs.load(turn["id"])
        with pytest.raises(DBAPIError):
            jobs.commit(loaded, source, loaded["state"], ("test-embedding-v1", {memory_id: [1e300, 0, 0]}))
        assert chats.get("vector-owner", session["id"])["revision"] == loaded["revision"]
        assert jobs.load(turn["id"]) is not None
        assert jobs.commit(loaded, source, loaded["state"], ("test-embedding-v1", {memory_id: [0, 1, 0]}))
        current = chats.get("vector-owner", session["id"])
        assert vectors.search("vector-owner", session["id"], "test-embedding-v1", [0, 1, 0])[0]["content"] == "喜欢咖啡"
        import json
        from functools import partial

        from services.character_chat import CharacterChatService
        from services.character_chat_vectors import select_memories

        class Provider:
            model_id = "controlled"
            model_fingerprint = "weights-v1"
            dimension = 3

            def embed_texts(self, texts):
                return [[0, 1, 0] if "咖啡" in text else [1, 0, 0] for text in texts]

            def embed_query(self, text):
                return [0, 1, 0]

        current = chats.manage_memory("vector-owner", session["id"], current["revision"], content="喜欢看书")

        async def complete(request, _session, _request):
            context = json.loads(request.character_context.conversation_reference_context)
            assert [memory["content"] for memory in context["memories"]] == ["喜欢咖啡"]
            return "为你准备咖啡", "controlled"

        chat = CharacterChatService(db, complete, retrieve=partial(select_memories, database=db, provider=Provider(), limit=1))
        asyncio.run(chat.generate("vector-owner", session["id"], revision=current["revision"], request_id="retrieved", message="喝什么？"))
        current = chats.get("vector-owner", session["id"])
        assert len(current["state"]["memories"]) == 2  # Retrieval selection never truncates saved memory.
        chats.delete_sessions("vector-owner", [{"id": session["id"], "revision": current["revision"]}])
        assert db.execute_sql("SELECT memory_id FROM character_chat_memory_vectors WHERE session_id=:session", {"session": session["id"]}) == []
    finally:
        db.close()
