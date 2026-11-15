"""Opt-in acceptance against the isolated real PostgreSQL instance."""

import asyncio
import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from character.library import CharacterSettings
from db.errors import RevisionConflictError
from repositories.character_chat import CharacterChatRepository
from repositories.character_chat_enrichment import CharacterChatEnrichmentRepository
from repositories.character_chat_generation import CharacterChatGenerationRepository, GenerationInProgressError

pytestmark = pytest.mark.skipif(os.getenv("RUN_CHARACTER_PG_INTEGRATION") != "1", reason="isolated PostgreSQL required")


@pytest.fixture
def database():
    from db.pg_database import PgDatabase, SyncPgAdapter

    db = SyncPgAdapter(PgDatabase(os.environ["DATABASE_URL"]))
    yield db
    db.close()


def test_postgres_chat_generation_enrichment_retry_and_erasure(database):
    repo = CharacterChatRepository(database)
    char = repo.characters.create("pg-owner", CharacterSettings(name="旅人", persona="谨慎", greeting="你好"))
    session = repo.create("pg-owner", char["id"], "验收")
    jobs = CharacterChatGenerationRepository(database)
    token = jobs.claim(session, "request", "喜欢茶")
    turn = repo.commit_turn(session, "request", "喜欢茶", "知道了", "test", generation_token=token, enqueue_enrichment=True)
    assert jobs.get("pg-owner", session["id"], "request")["status"] == "completed"
    enrichment = CharacterChatEnrichmentRepository(database)
    assert enrichment.fail(turn["id"], 2) == "failed"
    enrichment.retry("pg-owner", session["id"], 3)
    loaded, source = enrichment.load(turn["id"], 4)
    assert enrichment.commit(loaded, source, loaded["state"])
    assert enrichment.fail(turn["id"], 2) == "inactive"
    assert repo.list("pg-owner", char["id"])[0]["id"] == session["id"]
    assert any(item["id"] == session["id"] for item in repo.list("pg-owner"))
    assert repo.list("other-owner") == []
    repo.delete_sessions("pg-owner", [{"id": session["id"], "revision": 5}])
    assert database.execute_sql("SELECT turn_id FROM character_chat_enrichment_jobs WHERE session_id=:session",
                                {"session": session["id"]}) == []


def test_postgres_concurrent_generation_claim_has_one_winner(database):
    repo = CharacterChatRepository(database)
    char = repo.characters.create("pg-owner", CharacterSettings(name="旅人", persona="谨慎", greeting="你好"))
    session = repo.create("pg-owner", char["id"], "并发")
    jobs = CharacterChatGenerationRepository(database)

    def claim(request):
        try:
            return jobs.claim(session, request, "你好")
        except (RevisionConflictError, GenerationInProgressError):
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(value is not None for value in pool.map(claim, ["one", "two"])) == 1


def test_postgres_memory_rewind_branch_and_atomic_delete(database):
    repo = CharacterChatRepository(database)
    char = repo.characters.create("branch-owner", CharacterSettings(name="旅人", persona="谨慎", greeting="你好"))
    session = repo.create("branch-owner", char["id"], "原会话")
    session = repo.manage_memory("branch-owner", session["id"], 1, content="喜欢茶")
    memory_id = session["state"]["memories"][0]["id"]
    turn = repo.commit_turn(session, "request", "去海边", "好", "test")
    session = repo.get("branch-owner", session["id"])
    branch = repo.create("branch-owner", char["id"], "分支", source_session_id=session["id"],
                         source_revision=session["revision"], source_turn_id=turn["id"], hypothesis="去山上")
    session = repo.manage_memory("branch-owner", session["id"], session["revision"],
                                memory_id=memory_id, content="喜欢咖啡")
    session = repo.retract("branch-owner", session["id"], turn["id"], session["revision"])
    assert session["state"]["memories"][0]["content"] == "喜欢咖啡"
    assert repo.get("branch-owner", branch["id"])["state"]["memories"][0]["content"] == "喜欢茶"
    with pytest.raises(RevisionConflictError):
        repo.delete_sessions("branch-owner", [{"id": session["id"], "revision": session["revision"]},
                                             {"id": branch["id"], "revision": 999}])
    assert repo.get("branch-owner", session["id"])["revision"] == session["revision"]
    repo.delete_sessions("branch-owner", [{"id": session["id"], "revision": session["revision"]}])
    assert repo.get("branch-owner", branch["id"])["baseline"]["history"][0]["message"] == "去海边"


def test_postgres_legacy_generation_upgrade_preserves_receipt():
    import sqlalchemy as sa
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy.ext.asyncio import create_async_engine

    def migrate(connection):
        # Reconstruct the actual 021 shape, not today's ORM create_all schema.
        connection.execute(sa.text("CREATE SCHEMA migration_fixture"))
        connection.execute(sa.text("SET LOCAL search_path TO migration_fixture"))
        connection.execute(sa.text("CREATE TABLE character_chat_sessions (id TEXT PRIMARY KEY)"))
        connection.execute(sa.text("CREATE TABLE character_chat_turns (id TEXT PRIMARY KEY)"))
        modules = []
        for name in ("021_character_chat_generation", "022_character_chat_draft", "023_character_chat_enrichment"):
            path = Path(__file__).parents[1] / "alembic" / "versions" / (name + ".py")
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.op = Operations(MigrationContext.configure(connection))
            modules.append(module)
        modules[0].upgrade()
        assert "draft_text" not in {column["name"] for column in sa.inspect(connection).get_columns("character_chat_generations")}
        connection.execute(sa.text("INSERT INTO character_chat_sessions (id) VALUES ('legacy-session')"))
        connection.execute(sa.text("INSERT INTO character_chat_generations "
            "(session_id,request_id,payload_json,status,lease_token,lease_until,attempt,turn_id) "
            "VALUES ('legacy-session','legacy-request','[]','completed','token',0,1,'legacy-turn')"))
        for module in modules[1:]:
            module.upgrade()
            module.upgrade()
        row = connection.execute(sa.text("SELECT * FROM character_chat_generations")).mappings().one()
        assert row["request_id"] == "legacy-request" and row["turn_id"] == "legacy-turn"
        assert row["status"] == "completed" and row["draft_text"] == "" and row["draft_revision"] == 0
        for module in reversed(modules[1:]):
            module.downgrade()
        assert connection.execute(sa.text("SELECT turn_id FROM character_chat_generations")).scalar_one() == "legacy-turn"

    async def exercise():
        engine = create_async_engine(os.environ["DATABASE_URL"])
        try:
            async with engine.connect() as connection:
                transaction = await connection.begin()
                try:
                    await connection.run_sync(migrate)
                finally:
                    await transaction.rollback()
        finally:
            await engine.dispose()

    asyncio.run(exercise())
