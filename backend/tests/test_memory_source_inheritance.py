"""Successors retain only real parent links, atomically on SQLite and PostgreSQL."""

import os
import uuid
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.models import MemoryItem, UserScope
from db.database import SQLiteDB
from db.memory_source import ClaimSourceRevokedError, source_identity, source_scope


@pytest.fixture(scope="module", params=["sqlite", "pg"])
def source_database(request, tmp_path_factory):
    if request.param == "sqlite":
        database = SQLiteDB(tmp_path_factory.mktemp("source-inherit") / "records.sqlite")
    else:
        url = os.environ.get("SOURCE_INHERIT_PG_URL")
        if url is None:
            pytest.skip("Dedicated PostgreSQL test endpoint is not configured")
        assert "/stage3_stage35_address_guards?" in url and "port=25433" in url
        from db.pg_database import PgDatabase, SyncPgAdapter

        database = SyncPgAdapter(PgDatabase(url))
    yield database
    (database.close if hasattr(database, "_pg") else database.close_connection)()


async def arrange(database):
    repo = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("web", "source-inherit", "synthetic-" + uuid.uuid4().hex, "old-session", "private")
    new_scope = replace(scope, conversation_id="new-session")
    for owner, sid, body in [
        (scope, "old", "合成旧来源：只有周末才去虚构工坊。"),
        (new_scope, "new", "合成补充来源：还必须材料齐全才去虚构工坊。"),
    ]:
        assert (
            await repo.capture_source(
                "fixture-role", owner, source_message_id=sid, body=body, observed_at=datetime.now(timezone.utc)
            )
            == "recorded"
        )
    old = await repo.append_claim(
        "fixture-role",
        scope,
        MemoryItem("", "user_fact", "只有周末去虚构工坊", 0.9),
        memory_key="fixture-condition",
        scope_level="user_character",
        source_message_id="old",
        source_message_ids=("old",),
        evidence=("合成旧来源：只有周末才去虚构工坊。",),
    )
    return repo, scope, new_scope, old


async def successor(repo, scope, old, source_ids, *, relation="SUPERSEDE"):
    return await repo.append_claim(
        "fixture-role",
        scope,
        MemoryItem("", "user_fact", "去虚构工坊需周末且材料齐全", 0.9),
        memory_key="fixture-condition",
        scope_level="user_character",
        relation_type=relation,
        supersedes_memory_id=old["id"],
        source_message_id="new",
        source_message_ids=source_ids,
        evidence=("合成旧来源：只有周末才去虚构工坊。", "合成补充来源：还必须材料齐全才去虚构工坊。"),
    )


async def test_successor_preserves_actual_old_link_without_expanding_raw_speech_scope(source_database):
    repo, scope, new_scope, old = await arrange(source_database)
    current = await successor(repo, new_scope, old, ("old", "new"))
    pairs = ((current["id"], "old"), (current["id"], "new"))
    revisions = await repo.linked_source_revisions("fixture-role", new_scope, claim_sources=pairs)
    assert {row["source_message_id"] for row in revisions} == {"old", "new"}
    assert all(set(row) == {"memory_id", "source_message_id", "observed_at", "body_sha256"} for row in revisions)
    raw = await repo.linked_source_receipts("fixture-role", new_scope, claim_sources=pairs)
    assert [row["source_message_id"] for row in raw] == ["new"]
    assert [row["source_message_id"] for row in await repo.list_sources("fixture-role", new_scope)] == ["new"]
    assert (await repo.get_memory_record(old["id"], "fixture-role", new_scope))["status"] == "superseded"
    assert (
        await repo.linked_source_revisions(
            "fixture-role", replace(new_scope, sender_id="foreign-owner"), claim_sources=pairs
        )
        == []
    )


async def test_replacement_does_not_implicitly_inherit_unused_old_sources(source_database):
    repo, _scope, new_scope, old = await arrange(source_database)
    current = await successor(repo, new_scope, old, ("new",))
    revisions = await repo.linked_source_revisions(
        "fixture-role", new_scope, claim_sources=((current["id"], "old"), (current["id"], "new"))
    )
    assert [row["source_message_id"] for row in revisions] == ["new"]


@pytest.mark.parametrize("kind", ["unbound_same_owner", "foreign_owner", "missing"])
async def test_unbound_source_fails_without_committing_successor_or_target_change(source_database, kind):
    repo, _scope, new_scope, old = await arrange(source_database)
    if kind != "missing":
        owner = new_scope if kind == "unbound_same_owner" else replace(new_scope, sender_id="other-" + uuid.uuid4().hex)
        assert (
            await repo.capture_source(
                "fixture-role",
                owner,
                source_message_id="unbound",
                body="完整合成但未关联前版的另一条原话。",
                observed_at=datetime.now(timezone.utc),
            )
            == "recorded"
        )
    before = await repo.list_memory_records("fixture-role", new_scope, limit=None, include_inactive=True)
    with pytest.raises(ValueError, match="unambiguously linked"):
        await successor(repo, new_scope, old, ("old", "new", "unbound"))
    assert await repo.list_memory_records("fixture-role", new_scope, limit=None, include_inactive=True) == before
    assert (await repo.get_memory_record(old["id"], "fixture-role", new_scope))["status"] == "active"


def mutate(database, sql, params):
    if hasattr(database, "_pg"):
        from sqlalchemy import text

        async def apply():
            async with database._pg.async_session() as session:
                await session.execute(text(sql), params)
                await session.commit()

        database._run(apply())
    else:
        connection = database._get_connection()
        connection.execute(sql, params)
        connection.commit()


async def test_revoked_parent_source_aborts_successor_transaction(source_database):
    repo, scope, new_scope, old = await arrange(source_database)
    identity = source_identity(
        source_scope(
            "fixture-role",
            scope.platform,
            scope.adapter,
            scope.sender_id,
            scope.conversation_type,
            scope.conversation_id,
        ),
        "old",
    )
    mutate(
        source_database,
        "UPDATE memory_sources SET state='revoked',body=NULL WHERE source_key=:key",
        dict(key=identity["source_key"]),
    )
    before = await repo.list_memory_records("fixture-role", new_scope, limit=None, include_inactive=True)
    with pytest.raises(ClaimSourceRevokedError, match="inherited"):
        await successor(repo, new_scope, old, ("old", "new"))
    assert await repo.list_memory_records("fixture-role", new_scope, limit=None, include_inactive=True) == before


async def test_ambiguous_same_id_origins_cannot_be_collapsed_into_one_inherited_grant(source_database):
    repo, _scope, new_scope, old = await arrange(source_database)
    assert (
        await repo.capture_source(
            "fixture-role",
            new_scope,
            source_message_id="old",
            body="同一所有者另一会话复用编号的不同合成原话。",
            observed_at=datetime.now(timezone.utc),
        )
        == "recorded"
    )
    identity = source_identity(
        source_scope(
            "fixture-role",
            new_scope.platform,
            new_scope.adapter,
            new_scope.sender_id,
            new_scope.conversation_type,
            new_scope.conversation_id,
        ),
        "old",
    )
    mutate(
        source_database,
        "INSERT INTO memory_source_links (memory_id,source_key) VALUES (:memory,:key)",
        dict(memory=old["id"], key=identity["source_key"]),
    )
    before = await repo.list_memory_records("fixture-role", new_scope, limit=None, include_inactive=True)
    with pytest.raises(ValueError, match="unambiguously linked"):
        await successor(repo, new_scope, old, ("old", "new"))
    assert await repo.list_memory_records("fixture-role", new_scope, limit=None, include_inactive=True) == before


async def test_merge_retains_declared_parent_dependencies(source_database):
    repo, _scope, new_scope, old = await arrange(source_database)
    current = await successor(repo, new_scope, old, ("old", "new"), relation="MERGE")
    pairs = ((current["id"], "old"), (current["id"], "new"))
    revisions = await repo.linked_source_revisions("fixture-role", new_scope, claim_sources=pairs)
    assert {row["source_message_id"] for row in revisions} == {"old", "new"}
    assert (await repo.get_memory_record(old["id"], "fixture-role", new_scope))["status"] == "superseded"
