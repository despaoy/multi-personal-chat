"""Real two-engine mutation tests, restricted to a disposable r95 cluster."""

import argparse
import asyncio
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from db.database import SQLiteDB
from evaluation.conversation_source_probe import ROOT, verify_cluster


def exercise(db):
    scope = dict(character_id="role", platform="test", adapter="test", sender_id="user",
                 conversation_type="private", conversation_id="room")

    def write(key, **kwargs):
        return db.append_character_memory_claim(
            **scope, memory_type="user_fact", memory_key=key, content="independent fact", **kwargs,
        )

    results = []
    for relation in ("SUPERSEDE", "MERGE", "RETRACT"):
        for stale_kind in ("inactive", "older"):
            key = relation + stale_kind
            first = write(key, observed_at="2026-09-20T12:00:00+00:00")
            if stale_kind == "inactive":
                write(key, relation_type="SUPERSEDE", supersedes_memory_id=first["id"],
                      observed_at="2026-09-20T13:00:00+00:00")
            before = db.list_character_memory_claims(**scope, limit=None, include_inactive=True)
            try:
                write(key, relation_type=relation, supersedes_memory_id=first["id"],
                      observed_at="2026-09-20T11:00:00+00:00")
            except ValueError as exc:
                expected = "memory target changed" if stale_kind == "inactive" else "memory observation precedes target"
                assert expected in str(exc), str(exc)
            else:
                raise AssertionError("obsolete mutation accepted")
            after = db.list_character_memory_claims(**scope, limit=None, include_inactive=True)
            assert before == after, "rejected transaction changed storage"
            results.append(dict(case=key, unchanged=True))

    # Different successor keys bypass a per-key revision lock; the common
    # target row must still be protected. Repeated to exercise real contention.
    for iteration in range(20):
        original = write(f"root-{iteration}")

        def change(index, iteration=iteration, original=original):
            try:
                write(f"successor-{iteration}-{index}", relation_type="SUPERSEDE",
                      supersedes_memory_id=original["id"])
                return True
            except ValueError as exc:
                assert "memory target changed" in str(exc)
                return False

        with ThreadPoolExecutor(max_workers=4) as pool:
            accepted = sum(pool.map(change, range(4)))
        assert accepted == 1, accepted
        results.append(dict(case=f"concurrent-{iteration}", accepted=accepted))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args()
    socket = args.socket.resolve()
    if (not socket.is_relative_to(ROOT) or not socket.parent.name.startswith("r95pg.")
            or not (socket / ".s.PGSQL.25433").exists()):
        raise ValueError("Only disposable r95 cluster accepted")
    url = "postgresql+asyncpg://boot@/postgres?host=" + str(socket) + "&port=25433"
    asyncio.run(verify_cluster(url, socket.parent / "data"))
    os.environ["DATABASE_URL"] = url
    from db.pg_database import PgDatabase, SyncPgAdapter

    pg = SyncPgAdapter(PgDatabase(url))
    try:
        result = dict(sqlite=exercise(SQLiteDB(socket.parent / "claims.sqlite")), postgres=exercise(pg))
        assert result["sqlite"] == result["postgres"]
        backend = Path(__file__).resolve().parents[1]
        result["manifest"] = dict(model_calls=0, production_modified=False, http_e2e=False,
            sources={name: hashlib.sha256((backend / name).read_bytes()).hexdigest()
                     for name in ("db/database.py", "db/pg_database.py", "db/memory_claim_guard.py",
                                  "evaluation/memory_claim_conflict_probe.py")})
        output = socket.parent / "result.json"
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("CONFLICT_PARITY_OK " + str(output), flush=True)
    finally:
        pg.close()


if __name__ == "__main__":
    main()
