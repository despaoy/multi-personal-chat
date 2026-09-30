"""Source lifecycle parity on disposable SQLite and PostgreSQL, without a model."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier


def exercise(db):
    results = []
    base = dict(character_id="role", platform="test", adapter="source-probe", sender_id="alice",
                conversation_type="private", conversation_id="room")

    def scope(case, **changes):
        return dict(base, sender_id=case) | changes

    def capture(s, message="same-id", body="我目前在唐山轮岗，到月底结束，之后回苏州。", at=None):
        return db.capture_memory_source(**s, source_message_id=message, body=body,
                                        observed_at=at or (datetime.now(timezone.utc) - timedelta(seconds=1)))

    def fresh_capture(s, **kwargs):
        # Deterministic new receipt after a deletion, independent of coarse OS clocks.
        return capture(s, at=datetime.now(timezone.utc) + timedelta(seconds=1), **kwargs)

    def claim(s, key="work", message="same-id", **kw):
        return db.append_character_memory_claim(**s, memory_type="user_fact", memory_key=key,
                                                content=key, source_message_id=message, **kw)

    s = scope("no-candidate")
    assert capture(s) == "recorded"
    assert db.list_character_memory_claims(**s) == []
    assert "到月底结束" in db.list_memory_sources(**s)[0]["body"]
    results.append("source_without_claim")

    for level in ("conversation", "user_character", "user_global"):
        s = scope("scope-" + level)
        at = datetime.now(timezone.utc)
        assert capture(s, body="我叫阿黎，住在海棠路。", at=at) == "recorded"
        sibling = claim(s, key="name", scope_level=level)
        target = claim(s, key="address", scope_level=level)
        assert db.list_memory_sources(**(s | {"conversation_id": "elsewhere"})) == []
        assert db.list_memory_sources(**(s | {"character_id": "other"})) == []
        assert db.erase_character_memories(**s, memory_id=target["id"]) == 1
        assert db.list_memory_sources(**s) == []
        assert [row["id"] for row in db.list_character_memory_claims(**s)] == [sibling["id"]]
        assert capture(s, body="我叫阿黎，住在海棠路。", at=at) == "stale"
        assert fresh_capture(s, body="我叫阿黎，住在海棠路。") == "revoked"
        assert fresh_capture(s, message="new-id", body="我现在说的是新内容。") == "recorded"
        results.append("sibling_erase_" + level)

    # An anchor written by the rule path precedes delayed semantic capture.
    s = scope("late-capture")
    old_receipt = datetime.now(timezone.utc) - timedelta(seconds=1)
    target = claim(s)
    assert db.delete_character_memory(target["id"], **s)
    assert capture(s, at=old_receipt) == "stale"
    assert fresh_capture(s) == "revoked"
    assert capture(s, message="queued-without-claim", at=old_receipt) == "stale"
    results.append("late_capture_and_queued_without_claim")

    s = scope("clear-source-only")
    old_receipt = datetime.now(timezone.utc)
    assert capture(s, at=old_receipt) == "recorded"
    assert db.clear_character_memories(**s) == 0
    assert db.list_memory_sources(**s) == []
    assert capture(s, at=old_receipt) == "stale"
    assert fresh_capture(s) == "revoked"
    results.append("clear_without_claim")

    s = scope("lineage")
    capture(s, message="first")
    first = claim(s, message="first")
    capture(s, message="next")
    claim(s, message="next", relation_type="SUPERSEDE", supersedes_memory_id=first["id"])
    capture(s, message="unrelated")
    independent = claim(s, key="independent", message="unrelated")
    assert db.erase_character_memories(**s, memory_id=first["id"]) == 2
    assert db.list_memory_sources(**s) == []  # Raw layer fenced, independent fact retained.
    assert [r["id"] for r in db.list_character_memory_claims(**s)] == [independent["id"]]
    results.append("lineage_and_independent_source")

    s = scope("isolation")
    for field, value in (("sender_id", "different-user"), ("character_id", "other-role"),
                         ("conversation_id", "other-room"), ("adapter", "different-adapter"),
                         ("platform", "different-platform")):
        other = s | {field: value}
        capture(other, body="朋友说：我住在那里，不是我自己的住址。")
        assert db.list_memory_sources(**s) == []
    capture(s)
    target = claim(s)
    assert db.erase_character_memories(**(s | {"sender_id": "intruder"}), memory_id=target["id"]) == 0
    assert len(db.list_memory_sources(**s)) == 1
    results.append("exact_identity_and_unauthorized_erase")

    s = scope("immutable")
    at = datetime.now(timezone.utc)
    assert capture(s, body="前半句，最后才说明不是我。", at=at) == "recorded"
    assert capture(s, body="前半句，最后才说明不是我。", at=at) == "recorded"
    assert capture(s, body="前半句", at=at) == "conflict"
    assert db.list_memory_sources(**s)[0]["body"].endswith("不是我。")
    assert db.list_memory_sources(**s, source_message_ids=[]) == []
    assert db.list_memory_sources(**s, source_message_ids=["absent"]) == []
    assert len(db.list_memory_sources(**s, source_message_ids=["same-id"])) == 1
    results.append("immutable_complete_text_and_batched_reads")

    for legacy in (False, True):
        s = scope("unlinked-" + str(legacy))
        capture(s, message="primary", body="我住在海棠路。")
        capture(s, message="no-candidate", body="我叫阿黎，住在海棠路。")
        elsewhere = s | {"character_id": "other-role", "conversation_id": "other-room"}
        capture(elsewhere, body="我住在海棠路。")
        writer = db.add_or_update_character_memory if legacy else db.append_character_memory_claim
        target = writer(**s, memory_type="user_fact", memory_key="address", content="海棠路",
                        source_message_id="primary")
        kept = claim(s, key="name", message="no-candidate")
        assert db.erase_character_memories(**s, memory_id=target["id"]) == 1
        assert db.list_memory_sources(**s) == []
        assert db.list_memory_sources(**s, source_message_ids=["no-candidate"]) == []
        assert db.list_memory_sources(**elsewhere) == []
        assert [r["id"] for r in db.list_character_memory_claims(**s)] == [kept["id"]]
        assert fresh_capture(s, message="new-after-erasure", body="这是新说的话。") == "recorded"
        assert [r["source_message_id"] for r in db.list_memory_sources(**s)] == ["new-after-erasure"]
        results.append("unlinked_fence_legacy_" + str(legacy))

    # Both possible database lock orders must end with no visible source text.
    for iteration in range(12):
        s = scope("race-" + str(iteration))
        target = claim(s)
        at = datetime.now(timezone.utc)
        start = Barrier(2)

        def delayed_capture(start=start, s=s, at=at):
            start.wait(timeout=10)
            return capture(s, at=at)

        def erase(start=start, s=s, target=target):
            start.wait(timeout=10)
            return db.erase_character_memories(**s, memory_id=target["id"])

        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = pool.submit(delayed_capture), pool.submit(erase)
            assert a.result(timeout=20) in {"recorded", "stale", "revoked"}
            assert b.result(timeout=20) == 1
        assert db.list_memory_sources(**s) == []
        assert fresh_capture(s) == "revoked"
    results.append("concurrent_capture_erase_12")
    s = scope("late-claim-commit")
    receipt = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    db.clear_character_memories(**s)
    try:
        claim(s, observed_at=receipt)
    except ValueError as exc:
        assert "predates erasure" in str(exc)
    else:
        raise AssertionError("In-flight old receipt restored a claim")
    assert db.list_character_memory_claims(**s) == []
    results.append("late_claim_commit_fenced")
    for legacy in (False, True):
        s = scope('revoked-retry-' + str(legacy))
        capture(s)
        target = claim(s)
        db.erase_character_memories(**s, memory_id=target['id'])
        writer = db.add_or_update_character_memory if legacy else db.append_character_memory_claim
        for extra in ({},) if legacy else ({}, dict(observed_at=datetime.now(timezone.utc).isoformat())):
            try:
                writer(**s, memory_type='user_fact', memory_key='retry', content='cannot restore',
                       source_message_id='same-id', **extra)
            except ValueError as exc:
                assert 'identity was revoked' in str(exc)
            else:
                raise AssertionError('Revoked identity restored a claim')
        assert db.list_character_memory_claims(**s) == []
    results.append('revoked_identity_retries')
    s = scope('history-grant')
    for message, body in [('removed', '原来的住处。'), ('kept', '无关的新话题。')]:
        capture(s, message=message, body=body)
        db.add_message(dict(platform=s['platform'], adapter=s['adapter'], senderId=s['sender_id'],
            characterId=s['character_id'], conversationType=s['conversation_type'],
            conversationId=s['conversation_id'], sessionType='private', sessionId=s['conversation_id'],
            sourceMessageId=message, message=body, reply='答：' + body, createdAt='2026-09-27T12:00:00'))
    target = claim(s, message='removed')
    db.erase_character_memories(**s, memory_id=target['id'])
    history = db.list_conversation_history(s['platform'], s['adapter'], s['sender_id'],
        s['conversation_type'], s['conversation_id'], character_id=s['character_id'])
    assert [row['content'] for row in history] == ['无关的新话题。', '答：无关的新话题。']
    results.append('revoked_history_projection')
    return results


def main():
    from db.database import SQLiteDB
    from evaluation.conversation_source_probe import ROOT, verify_cluster

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args()
    socket = args.socket.resolve()
    if (not socket.is_relative_to(ROOT) or not socket.parent.name.startswith(("r106pg.", "r107pg.", "r108pg.", "r110pg."))
            or not (socket / ".s.PGSQL.25433").exists()):
        raise ValueError("Only disposable r106 PostgreSQL cluster accepted")
    url = "postgresql+asyncpg://boot@/postgres?host=" + str(socket) + "&port=25433"
    asyncio.run(verify_cluster(url, socket.parent / "data"))
    os.environ["DATABASE_URL"] = url
    from db.pg_database import PgDatabase, SyncPgAdapter

    pg = SyncPgAdapter(PgDatabase(url))
    try:
        result = dict(sqlite=exercise(SQLiteDB(socket.parent / "sources.sqlite")), postgres=exercise(pg),
                      model_calls=0, runtime_retrieval_enabled=False, production_modified=False)
        result["search_sqlite"] = exercise_search(SQLiteDB(socket.parent / "search.sqlite"))
        result["search_postgres"] = exercise_search(pg)
        result['source_erase_sqlite'] = exercise_source_erasure(SQLiteDB(socket.parent / 'source-erase.sqlite'))
        result['source_erase_postgres'] = exercise_source_erasure(pg)
        assert result['source_erase_sqlite'] == result['source_erase_postgres']
        result["windows_sqlite"] = exercise_windows(SQLiteDB(socket.parent / "windows.sqlite"))
        result["windows_postgres"] = exercise_windows(pg)
        assert result["windows_sqlite"] == result["windows_postgres"]
        assert result["search_sqlite"] == result["search_postgres"]
        assert result["sqlite"] == result["postgres"]
        output = socket.parent / "result.json"
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("SOURCE_PARITY_OK " + str(output), flush=True)
    finally:
        pg.close()


def exercise_search(db):
    scope = dict(character_id="role", platform="test", adapter="search-parity", sender_id="user",
                 conversation_type="private", conversation_id="room")
    old = datetime(2020, 1, 1, tzinfo=timezone.utc)
    db.capture_memory_source(**scope, source_message_id="old", body="我在图书馆工作，后来离职了。", observed_at=old)
    for i in range(260):
        db.capture_memory_source(**scope, source_message_id=str(i), body="我今天买了一杯咖啡。", observed_at=old)
    db.capture_memory_source(**(scope | {"character_id": "other"}), source_message_id="foreign",
                             body="我在图书馆工作。", observed_at=old)
    rows = db.search_memory_sources(**scope, query="图书馆工作")
    assert [row["source_message_id"] for row in rows] == ["old"]
    for long_query in ('背景说明。' * 500 + '图书馆工作',
                       ' '.join(f'term{i}' for i in range(600)) + ' 图书馆工作'):
        assert [row['source_message_id'] for row in db.search_memory_sources(**scope, query=long_query)] == ['old']
    claim = db.append_character_memory_claim(**scope, memory_type="user_fact", memory_key="job",
                                              content="图书馆工作", source_message_id="old")
    assert [row["source_message_id"] for row in db.linked_memory_sources(**scope, memory_ids=(claim["id"],))] == ["old"]
    assert db.linked_memory_sources(**(scope | {"character_id": "other"}), memory_ids=(claim["id"],)) == []
    db.erase_character_memories(**scope, memory_id=claim["id"])
    assert db.search_memory_sources(**scope, query="图书馆工作") == []
    assert db.search_memory_sources(**scope, query="咖啡") == []  # unlinked pre-erasure sources also fenced
    return dict(old_source_beyond_260=True, exact_scope=True, claim_link_scope=True,
                linked_and_unlinked_erasure=True, long_query_complete_terms=True)


def exercise_source_erasure(db):
    from db.memory_source import ClaimSourceRevokedError
    from db.source_erasure import SourceClaimConflict

    base = dict(character_id='role', platform='test', adapter='source-erasure', sender_id='user',
                conversation_type='private', conversation_id='room')
    for source in ('remove', 'keep'):
        db.capture_memory_source(**base, source_message_id=source, body=source,
                                 observed_at=datetime.now(timezone.utc))
    assert db.erase_unlinked_memory_sources(**base, source_message_ids=('remove',)) == 1
    assert db.erase_unlinked_memory_sources(**base, source_message_ids=('remove',)) == 0
    assert [r['source_message_id'] for r in db.list_memory_sources(**base)] == ['keep']
    assert db.search_memory_sources(**base, query='remove') == []
    for index in range(12):
        scope = base | dict(sender_id=f'race-{index}')
        db.capture_memory_source(**scope, source_message_id='race', body='race-source',
                                 observed_at=datetime.now(timezone.utc))
        barrier = Barrier(2)

        def write(scope=scope, barrier=barrier):
            barrier.wait(timeout=10)
            try:
                db.append_character_memory_claim(**scope, memory_type='user_fact', memory_key='race',
                                                  content='race', source_message_id='race')
                return 'saved'
            except ClaimSourceRevokedError:
                return 'revoked'

        def erase(scope=scope, barrier=barrier):
            barrier.wait(timeout=10)
            try:
                return db.erase_unlinked_memory_sources(**scope, source_message_ids=('race',))
            except SourceClaimConflict:
                return 'linked'

        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = pool.submit(write), pool.submit(erase)
            result = a.result(timeout=20), b.result(timeout=20)
        assert result in {('saved', 'linked'), ('revoked', 1)}
        claims = db.list_character_memory_claims(**scope)
        sources = db.list_memory_sources(**scope)
        assert (len(claims), len(sources)) == ((1, 1) if result[0] == 'saved' else (0, 0))
    return dict(independent_source=True, idempotent=True, preserved_unrelated=True,
                search_purged=True, concurrent_claim_delete=12)


def exercise_windows(db):
    scope = dict(character_id="role", platform="test", adapter="window-parity", sender_id="user",
                 conversation_type="private", conversation_id="room")
    at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for i in range(260):
        db.capture_memory_source(**scope, source_message_id=f"tie-{i:03}", body="相同时刻的原话。", observed_at=at)
    db.capture_memory_source(**(scope | {"character_id": "other"}), source_message_id="tie-150a",
                             body="另一角色的原话。", observed_at=at)
    window, = db.memory_source_windows(**scope, source_message_ids=("tie-150",), radius=2)
    assert sorted(row["source_message_id"] for row in window["rows"]) == [f"tie-{i}" for i in range(148, 153)]
    assert db.memory_source_windows(**(scope | {"character_id": "other"}), source_message_ids=("tie-150",)) == []
    db.clear_character_memories(**scope)
    assert db.memory_source_windows(**scope, source_message_ids=("tie-150",)) == []
    return dict(indexed_ties=True, scope_isolation=True, erasure_fence=True)


if __name__ == "__main__":
    main()
