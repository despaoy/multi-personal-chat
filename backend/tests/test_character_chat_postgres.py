"""Opt-in acceptance against the isolated real PostgreSQL instance."""

import asyncio
import importlib.util
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
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


def test_postgres_concurrent_preset_installation_keeps_one_pending_edition(database):
    from character.kamimaho_preset import PRESET_KEY, SETTINGS, SOURCES
    from repositories.character_sources import CharacterSourceRepository

    owner = uuid.uuid4().hex
    sources = CharacterSourceRepository(database)
    with ThreadPoolExecutor(2) as pool:
        created = list(pool.map(lambda _: sources.install_preset(owner, PRESET_KEY, SETTINGS, SOURCES), range(2)))
    assert created[0]["id"] == created[1]["id"]
    assert len(sources.characters.list(owner)) == 1
    edition, = sources.list(owner, created[0]["id"], 1)
    assert edition["index_status"] == "pending"
    assert len(sources.get(owner, edition["id"])["sources"]) == 3


def test_postgres_concurrent_import_confirmation_creates_one_character(database):
    from repositories.character_library import CharacterLibraryRepository

    repo = CharacterLibraryRepository(database)
    owner = uuid.uuid4().hex
    settings = CharacterSettings(name="导入角色", persona="谨慎", greeting="你好")
    with ThreadPoolExecutor(2) as pool:
        records = list(pool.map(lambda _: repo.import_package(owner, settings, "same-confirmation"), range(2)))
    assert records[0]["id"] == records[1]["id"]
    assert len(repo.list(owner)) == 1


def test_postgres_closing_character_share_rejects_old_confirmations(database):
    from repositories.character_library import CharacterLibraryRepository

    repo = CharacterLibraryRepository(database)
    owner = uuid.uuid4().hex
    character = repo.create(owner, CharacterSettings(name="角色", persona="谨慎", greeting="你好"))["id"]

    first_preview = repo.publish_preview(owner, character, 1)
    first = repo.publish(owner, character, 1, True, preview_id=first_preview["preview_id"])
    repo.publish(owner, character, 1, False)
    with pytest.raises(RevisionConflictError):
        repo.publish(owner, character, 1, True, preview_id=first_preview["preview_id"])
    assert repo.get(owner, character)["share_token"] is None

    new_preview = repo.publish_preview(owner, character, 1)
    new = repo.publish(owner, character, 1, True, preview_id=new_preview["preview_id"])
    assert new["publication_id"] == first["publication_id"]
    assert new["share_token"] != first["share_token"]


def test_postgres_production_startup_requires_current_migration_head():
    from sqlalchemy import text

    from db.pg_database import PgDatabase
    from db.schema_readiness import require_current_migrations

    async def check():
        pg = PgDatabase(os.environ["DATABASE_URL"], require_migrations=True)
        try:
            await pg.init()
            async with pg.engine.connect() as connection:
                async with connection.begin() as transaction:
                    await connection.execute(text("UPDATE alembic_version SET version_num='fixture_stale'"))
                    with pytest.raises(RuntimeError, match="schema does not match"):
                        await connection.run_sync(require_current_migrations)
                    await transaction.rollback()
                await connection.run_sync(require_current_migrations)
        finally:
            await pg.engine.dispose()

    asyncio.run(check())


def test_postgres_fiction_rendering_keeps_one_valid_lease(database):
    from repositories.character_chat_proactive import CharacterChatProactiveRepository

    repo = CharacterChatProactiveRepository(database)
    owner = f"fiction-{uuid.uuid4().hex}"
    character = repo.chats.characters.create(owner, CharacterSettings(name="fixture", persona="fixture", greeting="hi"))
    session = repo.chats.create(owner, character["id"], "fiction")
    identity = session["id"]
    repo.set_policy(owner, identity, 0, enabled=True, cadence_hours=24, timezone="UTC", quiet_start=None, quiet_end=None)
    repo.schedule(owner, identity, 1, "id", "fiction", "test fiction", 101, now=100)
    barrier = Barrier(2)

    def claim(_):
        barrier.wait(timeout=5)
        return repo.claim_fiction(now=101)

    with ThreadPoolExecutor(2) as pool:
        expired, = [lease for lease in pool.map(claim, range(2)) if lease is not None]
    current = repo.claim_fiction(now=191)
    assert current is not None and not repo.finish_fiction(expired, "expired body", now=192)
    assert repo.finish_fiction(current, "new body", now=192)
    assert not repo.finish_fiction(current, "duplicate body", now=192)
    entry, = repo.fiction_history(owner, identity)["entries"]
    assert entry["content"] == "【角色虚构日常】new body"


def test_postgres_proactive_delivery_respects_session_consent_and_is_idempotent(database):
    from repositories.character_chat_proactive import CharacterChatProactiveRepository

    repo = CharacterChatProactiveRepository(database)
    owner = f"proactive-{uuid.uuid4().hex}"
    character = repo.chats.characters.create(owner, CharacterSettings(name="fixture", persona="fixture", greeting="hi"))
    session = repo.chats.create(owner, character["id"], "proactive")
    identity = session["id"]
    repo.set_policy(owner, identity, 0, enabled=True, cadence_hours=24, timezone="UTC", quiet_start=None, quiet_end=None)
    repo.schedule(owner, identity, 1, "id", "event", "test event", 101, now=100)
    repo.schedule(owner, identity, 1, "cancel", "event", "different event", 102, now=100)
    assert repo.change_schedule(owner, identity, "cancel", 102, None) == {"status": "cancelled"}
    assert repo.change_schedule(owner, identity, "id", 101, 102, now=100) == {"status": "pending"}
    assert repo.deliver_due(now=101) == 0
    with ThreadPoolExecutor(2) as pool:
        assert sum(pool.map(lambda _: repo.deliver_due(now=102), range(2))) == 1
    assert repo.view(owner, identity)["jobs"][0]["status"] == "sent"
    repo.set_policy(owner, identity, 1, enabled=True, cadence_hours=24, timezone="UTC", quiet_start=None, quiet_end=None,
                    inactivity_hours=72)
    assert repo._policy(identity)["inactivity_due_at"] is None
    repo.chats.commit_turn(session, "reply", "hi", "hello", "test")
    due = repo._policy(identity)["inactivity_due_at"]
    assert repo.deliver_due(now=due - 1) == 0
    generations = CharacterChatGenerationRepository(database)
    busy = repo.chats.get(owner, identity)
    generations.claim(busy, "busy", "fixture", lease_seconds=73 * 3600)
    assert not repo.has_due(due)
    assert repo.deliver_due(now=due) == 0
    repo.chats.update_relationship(owner, identity, busy["revision"], "familiar", "手动修改")
    assert generations.get(owner, identity, "busy")["status"] == "superseded"
    assert repo.has_due(due)
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


def test_postgres_browser_reminder_claim_has_one_winner(database, monkeypatch):
    from test_character_chat_push import device_keys

    from repositories import character_chat_push
    from repositories.character_chat_proactive import CharacterChatProactiveRepository
    from repositories.character_chat_push import CharacterChatPushRepository

    repo = CharacterChatProactiveRepository(database)
    push = CharacterChatPushRepository(database)
    monkeypatch.setattr(character_chat_push, "_SCAN_PAGE", 1)
    owner = f"push-{uuid.uuid4().hex}"
    character = repo.chats.characters.create(owner, CharacterSettings(name="fixture", persona="fixture", greeting="hi"))
    session = repo.chats.create(owner, character["id"], "push")
    identity = session["id"]
    repo.set_policy(owner, identity, 0, enabled=True, cadence_hours=24, timezone="UTC", quiet_start=None, quiet_end=None)
    endpoints = {f"https://fcm.googleapis.com/fcm/send/fixture-{number}" for number in (1, 2)}
    devices = {push.register(owner, identity, endpoint, *device_keys())["id"] for endpoint in endpoints}
    now = time()
    repo.schedule(owner, identity, 1, "push", "event", "test event", now + 1, now=now)
    assert repo.deliver_due(now=now + 1) == 1
    sent = []

    class Sender:
        def send(self, endpoint, public_key, auth, tag):
            sent.append(endpoint)
            return 201

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda _: push.deliver(Sender(), now=now + 1), range(2)))
    assert len(sent) == 2 and set(sent) == endpoints
    assert not push.has_due(now + 1)
    assert set(push.registered(owner, identity)) == devices
    session = repo.chats.get(owner, identity)
    repo.chats.delete_sessions(owner, [{"id": identity, "revision": session["revision"]}])
    assert not push.has_due(now + 1)


def test_postgres_push_result_cannot_delete_subscription_after_successful_retry(database):
    from test_character_chat_push import device_keys

    from repositories.character_chat_proactive import CharacterChatProactiveRepository
    from repositories.character_chat_push import CharacterChatPushRepository

    outreach = CharacterChatProactiveRepository(database)
    push = CharacterChatPushRepository(database)
    owner = f"push-lease-{uuid.uuid4().hex}"
    role = outreach.chats.characters.create(owner, CharacterSettings(name="fixture", persona="fixture", greeting="hi"))
    session = outreach.chats.create(owner, role["id"], "提醒")
    identity = session["id"]
    outreach.set_policy(owner, identity, 0, enabled=True, cadence_hours=24, timezone="UTC", quiet_start=None, quiet_end=None)
    device = push.register(owner, identity, "https://fcm.googleapis.com/fcm/send/fixture", *device_keys())["id"]
    now = time()
    outreach.schedule(owner, identity, 1, "push", "event", "fixture", now + 1, now=now)
    assert outreach.deliver_due(now=now + 1) == 1

    class Accepted:
        def send(self, *_):
            return 201

    class Older:
        def send(self, *_):
            push.deliver(Accepted(), now=now + 61)
            return 410

    push.deliver(Older(), now=now + 1)
    assert push.registered(owner, identity) == [device]
    assert [item["id"] for item in outreach.inbox(owner)] == ["push"]


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
    assert enrichment.keep_active(turn["id"], 2)
    assert enrichment.fail(turn["id"], 2) == "failed"
    enrichment.retry("pg-owner", session["id"], 3)
    assert not enrichment.keep_active(turn["id"], 2)
    assert enrichment.keep_active(turn["id"], 4)
    loaded, source = enrichment.load(turn["id"], 4)
    assert enrichment.commit(loaded, source, loaded["state"])
    assert not enrichment.keep_active(turn["id"], 4)
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
    current = repo.update_relationship("pg-owner", session["id"], session["revision"], "familiar", "手动修改")
    fresh = jobs.claim(current, "fresh", "继续聊天")
    assert repo.commit_turn(current, "fresh", "继续聊天", "好", "test", generation_token=fresh)["active"] == 1


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


def test_postgres_path_can_reuse_content_while_last_existing_branch_is_deleted(database):
    repo = CharacterChatRepository(database)
    owner = uuid.uuid4().hex
    character = repo.characters.create(owner, CharacterSettings(name="旅人", persona="谨慎", greeting="你好"))
    source = repo.create(owner, character["id"], "来源")
    turn = repo.commit_turn(source, "request", "去海边", "好", "test")
    source = repo.get(owner, source["id"])

    def create_branch():
        return repo.create(owner, character["id"], "分支", source_session_id=source["id"],
                           source_revision=source["revision"], source_turn_id=turn["id"])

    old = create_branch()
    sync = Barrier(2, timeout=10)

    def remove_old():
        sync.wait()
        repo.delete_sessions(owner, [{"id": old["id"], "revision": old["revision"]}])

    def create_new():
        sync.wait()
        return create_branch()

    with ThreadPoolExecutor(2) as pool:
        deletion = pool.submit(remove_old)
        creation = pool.submit(create_new)
        deletion.result(timeout=20)
        new = creation.result(timeout=20)

    assert new["baseline"]["history"][0]["message"] == "去海边"
    assert database.execute_sql(
        "SELECT session_id FROM character_chat_history_path WHERE owner_user_id=:owner", {"owner": owner},
    ) == [{"session_id": new["id"]}]
    assert len(database.execute_sql(
        "SELECT node_key FROM character_chat_history_nodes WHERE owner_user_id=:owner", {"owner": owner},
    )) == 1
    repo.delete_sessions(owner, [{"id": new["id"], "revision": new["revision"]}])
    assert database.execute_sql(
        "SELECT node_key FROM character_chat_history_nodes WHERE owner_user_id=:owner", {"owner": owner},
    ) == []


def test_postgres_legacy_history_migration_preserves_body_and_path_local_state():
    import sqlalchemy as sa
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy.ext.asyncio import create_async_engine

    def migrate(connection):
        connection.execute(sa.text("CREATE SCHEMA history_migration_fixture"))
        connection.execute(sa.text("SET LOCAL search_path TO history_migration_fixture"))
        connection.execute(sa.text(
            "CREATE TABLE character_chat_sessions (id TEXT PRIMARY KEY,owner_user_id TEXT NOT NULL,baseline_json TEXT NOT NULL)"
        ))
        name = "037_character_chat_history"
        path = Path(__file__).parents[1] / "alembic" / "versions" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.op = Operations(MigrationContext.configure(connection))
        source = {"id": "turn", "session_id": "deleted-source", "message": "旧正文", "reply": "旧回覆",
                  "evidence": {"sources": [{"title": "旧资料"}]}}
        original = {
            "a": {"history": [{**source, "state_after": {"memories": ["红茶"]}}], "state": {}},
            "b": {"history": [{**source, "state_after": {"memories": ["绿茶"]}}], "state": {}},
            "c": {"history": [source], "state": {}},
        }
        connection.execute(
            sa.text("INSERT INTO character_chat_sessions VALUES (:id,'owner',:baseline)"),
            [{"id": key, "baseline": json.dumps(value, ensure_ascii=False)} for key, value in original.items()],
        )
        module.upgrade()
        module.upgrade()
        assert connection.execute(sa.text("SELECT count(*) FROM character_chat_history_nodes")).scalar_one() == 1
        assert connection.execute(sa.text("SELECT count(*) FROM character_chat_history_path")).scalar_one() == 3
        connection.execute(sa.text(
            "INSERT INTO character_chat_history_nodes SELECT 'wrong-account',node_key,origin_session_id,payload_json "
            "FROM character_chat_history_nodes"
        ))
        with pytest.raises(sa.exc.IntegrityError), connection.begin_nested():
            connection.execute(sa.text(
                "INSERT INTO character_chat_history_path (session_id,position,owner_user_id,node_key) "
                "SELECT 'a',1,owner_user_id,node_key FROM character_chat_history_nodes WHERE owner_user_id='wrong-account'"
            ))
        module.downgrade()
        rows = connection.execute(sa.text("SELECT id,baseline_json FROM character_chat_sessions"))
        assert {key: json.loads(value) for key, value in rows} == original

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
        for name in ("021_character_chat_generation", "022_character_chat_draft", "023_character_chat_enrichment",
                     "039_generation_request_revision"):
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
            "VALUES ('legacy-session','legacy-request','[7,\"fixture\",null]','completed','token',0,1,'legacy-turn')"))
        for module in modules[1:]:
            module.upgrade()
            module.upgrade()
        row = connection.execute(sa.text("SELECT * FROM character_chat_generations")).mappings().one()
        assert row["request_id"] == "legacy-request" and row["turn_id"] == "legacy-turn"
        assert row["status"] == "completed" and row["draft_text"] == "" and row["draft_revision"] == 0
        assert row["request_revision"] == 7
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
