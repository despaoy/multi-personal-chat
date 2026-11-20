"""Opt-in acceptance against the isolated real PostgreSQL instance."""

import asyncio
import importlib.util
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time

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


def test_postgres_proactive_delivery_respects_session_consent_and_is_idempotent(database):
    from repositories.character_chat_proactive import CharacterChatProactiveRepository

    repo = CharacterChatProactiveRepository(database)
    owner = f"proactive-{uuid.uuid4().hex}"
    character = repo.chats.characters.create(owner, CharacterSettings(name="fixture", persona="fixture", greeting="hi"))
    session = repo.chats.create(owner, character["id"], "proactive")
    identity = session["id"]
    repo.set_policy(owner, identity, 0, enabled=True, cadence_hours=24, timezone="UTC", quiet_start=None, quiet_end=None)
    repo.schedule(owner, identity, 1, "id", "event", "test event", 101, now=100)
    with ThreadPoolExecutor(2) as pool:
        assert sum(pool.map(lambda _: repo.deliver_due(now=101), range(2))) == 1
    assert repo.view(owner, identity)["jobs"][0]["status"] == "sent"
    repo.set_policy(owner, identity, 1, enabled=True, cadence_hours=24, timezone="UTC", quiet_start=None, quiet_end=None,
                    inactivity_hours=72)
    assert repo._policy(identity)["inactivity_due_at"] is None
    repo.chats.commit_turn(session, "reply", "hi", "hello", "test")
    due = repo._policy(identity)["inactivity_due_at"]
    assert repo.deliver_due(now=due - 1) == 0
    assert repo.deliver_due(now=due) == 1
    assert repo.inbox(owner)[0]["kind"] == "inactivity"
    for number in (1, 2):
        send_at = due + number * 86400
        repo.schedule(owner, identity, 2, f"fiction-{number}", "fiction", f"scene-{number}", send_at, now=due)
        assert repo.deliver_due(now=send_at) == 1
    page = repo.fiction_history(owner, identity, limit=1)
    assert page["next_before"] == "fiction-2"
    earlier = repo.fiction_history(owner, identity, before=page["next_before"], limit=1)
    assert [entry["id"] for entry in earlier["entries"]] == ["fiction-1"]
    assert earlier["next_before"] is None
    session = repo.chats.get(owner, identity)
    repo.chats.delete_sessions(owner, [{"id": identity, "revision": session["revision"]}])
    assert repo.inbox(owner) == []
    assert repo.deliver_due(now=200) == 0


def test_postgres_legacy_copy_can_preview_and_confirm_an_author_update(database):
    from db.legacy_character_publications import backfill_legacy_copy_baselines
    from repositories.character_library import CharacterLibraryRepository

    suffix = uuid.uuid4().hex
    author, recipient = f'author-{suffix}', f'recipient-{suffix}'
    library = CharacterLibraryRepository(database)
    old = CharacterSettings(name='old', persona='x', greeting='x')
    source = library.create(author, old)
    copy = library.create(recipient, old)
    database.execute_sql(
        'UPDATE character_library SET source_character_id=:source,source_revision=1 WHERE id=:id',
        {'source': source['id'], 'id': copy['id']},
    )
    backfill_legacy_copy_baselines(database.execute_sql)
    library.update(author, source['id'], 1, CharacterSettings(name='new', persona='x', greeting='x'))
    library.publish(author, source['id'], 2, True)

    plan = library.source_update(recipient, copy['id'])
    args = (recipient, copy['id'], 1, 2, plan['publication_id'], plan['local_corpus_id'], {})
    preview = library.preview_source_update(*args)
    assert preview['sources'] == []
    assert library.import_source_update(*args, preview['preview_id'])['settings']['name'] == 'new'


def test_postgres_browser_reminder_claim_has_one_winner(database):
    from test_character_chat_push import device_keys

    from repositories.character_chat_proactive import CharacterChatProactiveRepository
    from repositories.character_chat_push import CharacterChatPushRepository

    repo = CharacterChatProactiveRepository(database)
    push = CharacterChatPushRepository(database)
    owner = f"push-{uuid.uuid4().hex}"
    character = repo.chats.characters.create(owner, CharacterSettings(name="fixture", persona="fixture", greeting="hi"))
    session = repo.chats.create(owner, character["id"], "push")
    identity = session["id"]
    repo.set_policy(owner, identity, 0, enabled=True, cadence_hours=24, timezone="UTC", quiet_start=None, quiet_end=None)
    device = push.register(owner, identity, "https://fcm.googleapis.com/fcm/send/fixture", *device_keys())
    now = time()
    repo.schedule(owner, identity, 1, "push", "event", "test event", now + 1, now=now)
    assert repo.deliver_due(now=now + 1) == 1
    sent = []

    class Sender:
        def send(self, endpoint, public_key, auth, tag):
            sent.append(tag)
            return 201

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda _: push.deliver(Sender(), now=now + 1), range(2)))
    assert len(sent) == 1
    assert not push.has_due(now + 1)
    assert push.registered(owner, identity) == [device["id"]]
    session = repo.chats.get(owner, identity)
    repo.chats.delete_sessions(owner, [{"id": identity, "revision": session["revision"]}])
    assert not push.has_due(now + 1)


def test_postgres_sourced_proactive_job_is_cancelled_on_history_change(database):
    from repositories.character_chat_proactive import CharacterChatProactiveRepository

    repo = CharacterChatProactiveRepository(database)
    owner = f"source-outreach-{uuid.uuid4().hex}"
    character = repo.chats.characters.create(owner, CharacterSettings(name="fixture", persona="fixture", greeting="hi"))
    session = repo.chats.create(owner, character["id"], "outreach")
    identity = session["id"]
    topic = "我们下次再聊这份方案"
    turn = repo.chats.commit_turn(session, "retry", topic, "好", "fixture")
    repo.set_policy(owner, identity, 0, enabled=True, cadence_hours=24, timezone="UTC", quiet_start=None, quiet_end=None)
    candidate, = repo.candidates(owner, identity)
    assert candidate["source_turn_id"] == turn["id"]
    repo.schedule(owner, identity, 1, "later", "follow_up", topic, 101, now=100, source_turn_id=turn["id"])
    current = repo.chats.get(owner, identity)
    repo.chats.retract(owner, identity, turn["id"], current["revision"])
    assert repo.deliver_due(now=101) == 0
    assert repo.view(owner, identity)["jobs"] == []


def test_postgres_concurrent_source_editions_have_distinct_sequence_numbers(database, monkeypatch):
    from repositories import character_sources
    from repositories.character_sources import CharacterSourceRepository

    sources = CharacterSourceRepository(database)
    owner = f"edition-{uuid.uuid4().hex}"
    character = sources.characters.create(owner, CharacterSettings(name="fixture", persona="fixture", greeting="hi"))
    identity = character["id"]
    monkeypatch.setattr(character_sources, "timestamp", lambda: "2026-11-20T12:00:00+00:00")
    original = sources.publish(owner, identity, 1, [{"id": "source", "title": "source", "content": "original"}])
    with ThreadPoolExecutor(2) as pool:
        versions = list(pool.map(
            lambda value: sources.publish(owner, identity, 1, [{"id": "source", "title": "source", "content": value}]),
            ("first", "second"),
        ))
    expected = database.execute_sql(
        "SELECT id,edition_number FROM character_source_corpora WHERE character_id=:id ORDER BY edition_number DESC",
        {"id": identity},
    )
    assert [row["edition_number"] for row in expected] == [3, 2, 1]
    assert [item["id"] for item in sources.list(owner, identity, 1)] == [row["id"] for row in expected]
    assert expected[-1]["id"] == original["id"]
    assert {item["id"] for item in versions} == {row["id"] for row in expected[:2]}


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
    receipts = "SELECT session_id FROM character_chat_erasure_receipts WHERE session_id=:id"
    assert database.execute_sql(receipts, {"id": session["id"]}) == []
    repo.delete_sessions("branch-owner", [{"id": session["id"], "revision": session["revision"]}])
    assert database.execute_sql(receipts, {"id": session["id"]}) == [{"session_id": session["id"]}]
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
