"""Exact pair receipts: no unrelated source crowding or identity reassignment."""

from datetime import datetime, timezone

import pytest

from db import memory_source
from db.database import SQLiteDB
from db.memory_source_receipts import linked_receipt_plan

SCOPE = dict(
    character_id="role",
    platform="test",
    adapter="pairs",
    sender_id="user",
    conversation_type="private",
    conversation_id="room",
)


def fixture_db(tmp_path):
    db = SQLiteDB(tmp_path / "pairs.sqlite")
    stamp = datetime.now(timezone.utc)
    for sid, body in [("a", "我的专业是信息管理与信息系统。"), ("b", "我的姓名是阿黎。")]:
        db.capture_memory_source(**SCOPE, source_message_id=sid, body=body, observed_at=stamp)
    a = db.append_character_memory_claim(
        **SCOPE,
        memory_type="user_fact",
        memory_key="major",
        content="专业",
        source_message_id="a",
        observed_at=stamp.isoformat(),
    )
    b = db.append_character_memory_claim(
        **SCOPE,
        memory_type="user_fact",
        memory_key="name",
        content="姓名",
        source_message_id="b",
        observed_at=stamp.isoformat(),
    )
    return db, a["id"], b["id"]


def test_exact_pairs_preserve_associations_not_cartesian(tmp_path):
    db, a, b = fixture_db(tmp_path)
    assert db.linked_memory_source_receipts(**SCOPE, claim_sources=((a, "b"), (b, "a"))) == []
    rows = db.linked_memory_source_receipts(**SCOPE, claim_sources=((a, "a"), (b, "b")))
    assert {(r["memory_id"], r["source_message_id"]) for r in rows} == {(a, "a"), (b, "b")}


def test_duplicate_requested_pairs_return_once(tmp_path):
    db, a, b = fixture_db(tmp_path)
    rows = db.linked_memory_source_receipts(**SCOPE, claim_sources=((a, "a"), (a, "a")))
    assert len(rows) == 1 and rows[0]["memory_id"] == a


def test_wrong_owner_role_and_conversation_cannot_grant_receipt(tmp_path):
    db, a, b = fixture_db(tmp_path)
    for change in [
        dict(sender_id="other"),
        dict(character_id="other"),
        dict(conversation_type="group", conversation_id="other"),
    ]:
        assert db.linked_memory_source_receipts(**(SCOPE | change), claim_sources=((a, "a"),)) == []


def test_recorded_but_unlinked_source_does_not_grant_claim_authority(tmp_path):
    db, a, b = fixture_db(tmp_path)
    db.capture_memory_source(
        **SCOPE,
        source_message_id="unlinked",
        body="我的专业是信息管理与信息系统。",
        observed_at=datetime.now(timezone.utc),
    )
    assert db.linked_memory_source_receipts(**SCOPE, claim_sources=((a, "unlinked"),)) == []


def test_revoked_source_is_not_returned_even_if_pair_requested(tmp_path):
    db, a, b = fixture_db(tmp_path)
    conn = db._get_connection()
    conn.execute("UPDATE memory_sources SET state='revoked',body=NULL WHERE source_message_id='a'")
    conn.commit()
    assert db.linked_memory_source_receipts(**SCOPE, claim_sources=((a, "a"),)) == []


def test_owner_fence_removes_stale_receipt(tmp_path):
    db, a, b = fixture_db(tmp_path)
    conn = db._get_connection()
    scope = memory_source.owner_scope(**SCOPE)
    conn.execute(
        "UPDATE memory_source_fences SET revoked_before=? WHERE owner_key=?",
        ("2999-01-01T00:00:00+00:00", scope["owner_key"]),
    )
    conn.commit()
    assert db.linked_memory_source_receipts(**SCOPE, claim_sources=((a, "a"),)) == []


@pytest.mark.parametrize("pairs", [((0, "a"),), (("1", "a"),), ((1, 3),), tuple((i + 1, "s") for i in range(201))])
def test_invalid_or_oversized_pair_request_rejected(pairs):
    with pytest.raises(ValueError):
        next(linked_receipt_plan(memory_source.source_scope(**SCOPE), pairs))


def test_empty_request_has_no_sql():
    with pytest.raises(StopIteration):
        next(linked_receipt_plan(memory_source.source_scope(**SCOPE), ()))


def test_two_hundred_pairs_use_bound_values_and_no_global_topk():
    sql, params = next(
        linked_receipt_plan(memory_source.source_scope(**SCOPE), tuple((i + 1, "source-" + str(i)) for i in range(200)))
    )
    assert "LIMIT" not in sql and "source-199" not in sql and len(params) < 500
    assert sql.count("CAST(:memory") == 200 and sql.count("CAST(:source") == 200


@pytest.mark.asyncio
async def test_more_than_two_hundred_receipts_are_read_in_exact_batches(tmp_path):
    import json

    from character.memory_service import CharacterMemoryService
    from character.models import UserScope
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    db = SQLiteDB(tmp_path / "batch.sqlite")
    stamp = datetime.now(timezone.utc)
    source_ids = tuple("source-" + str(i) for i in range(201))
    body = "我的专业是信息管理与信息系统。"
    for sid in source_ids:
        db.capture_memory_source(**SCOPE, source_message_id=sid, body=body, observed_at=stamp)
    claim = db.append_character_memory_claim(
        **SCOPE,
        memory_type="user_fact",
        memory_key="fact_信息管理与信息系统",
        content="专业",
        source_message_id=source_ids[0],
        source_message_ids_json=json.dumps(source_ids),
        evidence_json=json.dumps([body]),
        observed_at=stamp.isoformat(),
    )
    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute("BEGIN IMMEDIATE")
    memory_source.run_sqlite(cursor, memory_source.lock_owner(memory_source.owner_scope(**SCOPE)))
    scope = memory_source.source_scope(**SCOPE)
    for sid in source_ids[1:]:
        memory_source.run_sqlite(
            cursor, memory_source.link_plan(memory_source.source_identity(scope, sid), claim["id"])
        )
    conn.commit()
    batches = []

    class TracedRepository(DatabaseCharacterMemoryRepository):
        async def linked_source_receipts(self, *args, **kwargs):
            batches.append(len(kwargs["claim_sources"]))
            return await super().linked_source_receipts(*args, **kwargs)

    us = UserScope(
        SCOPE["platform"], SCOPE["adapter"], SCOPE["sender_id"], SCOPE["conversation_id"], SCOPE["conversation_type"]
    )
    items, count, trace = await CharacterMemoryService(
        TracedRepository(db), semantic_enabled=False
    ).recall_with_diagnostics("role", us, "我的专业是什么？", for_contextual_selection=True, reference_time=stamp)
    assert (
        batches == [200, 1] and trace["source_receipt_pairs_requested"] == trace["source_receipt_pairs_returned"] == 201
    )
    assert trace["field_presence"]["major"] is True and items[0].memory_key == "user_major"
    assert body in items[0].evidence
