"""Warm source-query index comparison on an isolated real PG cluster/SQLite.

No production DB accepted; no claim about generation or system-wide p95.
"""
import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from character.context_builder import build_user_scope
from db.conversation_indexes import SOURCE_INDEX_STATEMENTS
from db.conversation_source import TurnCursor, turn_source_query
from db.database import SQLiteDB
from evaluation.conversation_source_probe import ROOT, verify_cluster

INSERT = '''INSERT INTO messages
    (id, "sessionId", "sessionType", platform, adapter, "senderId", "characterId",
     "conversationType", "conversationId", "createdAt", message, reply, "traceId")
    VALUES (:id, :session, :kind, :platform, :adapter, :sender, :character,
            :kind, :conversation, :created, :message, :reply, :trace)'''
RECEIPT = '''INSERT INTO integration_receipts (receipt_key, owner, status, response, expires_at)
             VALUES (:trace, :trace, :status, '', 0)'''
INDEX_NAMES = ('idx_messages_private_source', 'idx_messages_local_source')


def corpus(noise, target=400):
    rows = []
    for index in range(target + noise):
        kind = 'private' if index % 2 == 0 else 'group'
        rows.append(dict(id=index + 1, session='session' + str(index), kind=kind, platform='web',
            adapter='audit', sender='alice' if index < target else 'other' + str(index % 1000),
            character='role', conversation='room',
            created='2026-09-01T10:00:00' if index < target else '2026-09-26T10:00:00',
            message='独立合成原话' + str(index), reply='合成回复', trace='trace' + str(index),
            status='delivery_failed' if index % 17 == 0 else 'delivered'))
    return rows


def queries():
    user = build_user_scope('web', 'audit', 'alice', 'room', 'private')
    group = build_user_scope('web', 'audit', 'alice', 'room', 'group')
    missing = build_user_scope('web', 'audit', 'missing', 'room', 'private')
    result = dict(private=turn_source_query(user, 'role', limit=40),
        group=turn_source_query(group, 'role', limit=40),
        missing=turn_source_query(missing, 'role', limit=40))
    original = '("createdAt" < :before_at OR ("createdAt" = :before_at AND id < :before_id))'
    base, params = result['private']
    params = params | dict(before_at='2026-09-01T10:00:00', before_id=150)
    query = base.replace(' ORDER BY ', ' AND ' + original + ' ORDER BY ')
    result['older'] = query, params
    result['runtime'] = turn_source_query(user, 'role', limit=40,
        before=TurnCursor(params['before_at'], params['before_id']))
    result['older_tuple'] = (query.replace(original, '("createdAt", id) < (:before_at, :before_id)'), params)
    earlier = query.replace(original, '"createdAt" < :before_at')
    same_time = query.replace(original, '("createdAt" = :before_at AND id < :before_id)')
    result['older_split'] = ('SELECT * FROM (SELECT * FROM (' + same_time
        + ') AS same_time UNION ALL SELECT * FROM (' + earlier
        + ') AS earlier) AS candidates ORDER BY "createdAt" DESC, id DESC LIMIT :limit', params)
    return result


def digest(rows):
    return hashlib.sha256(json.dumps([dict(row) for row in rows], ensure_ascii=False,
        sort_keys=True).encode()).hexdigest()


def sqlite_comparison(path, rows):
    if Path(path).exists():
        raise ValueError('Performance fixture must use a new SQLite file')
    database = SQLiteDB(path)
    conn = database._get_connection()
    # This fixture was just created by the guarded CLI (or tmp_path tests).
    # Remove only these two fixture indexes for a genuine unindexed baseline.
    for name in INDEX_NAMES:
        conn.execute('DROP INDEX IF EXISTS ' + name)
    conn.executemany(INSERT, rows)
    conn.executemany(RECEIPT, rows)
    conn.commit()
    result = {}
    for phase in ('before', 'after'):
        if phase == 'after':
            for statement in SOURCE_INDEX_STATEMENTS:
                conn.execute(statement)
            conn.commit()
        conn.execute('ANALYZE')
        result[phase] = {}
        for name, (query, params) in queries().items():
            conn.execute(query, params).fetchall()  # warmup, excluded
            timings = []
            for _ in range(5):
                started = time.perf_counter()
                found = conn.execute(query, params).fetchall()
                timings.append(time.perf_counter() - started)
            plan = [dict(row) for row in conn.execute('EXPLAIN QUERY PLAN ' + query, params)]
            result[phase][name] = dict(seconds=timings, median=statistics.median(timings),
                plan=plan, rows=len(found), digest=digest(found))
    for name in queries():
        assert result['before'][name]['digest'] == result['after'][name]['digest']
    for phase in result:
        assert result[phase]['older']['digest'] == result[phase]['older_tuple']['digest']
        assert result[phase]['older']['digest'] == result[phase]['older_split']['digest']
        assert result[phase]['older']['digest'] == result[phase]['runtime']['digest']
    return result


async def pg_comparison(url, rows):
    from sqlalchemy import text

    from db.pg_database import PgDatabase

    database = PgDatabase(url)
    try:
        await database.init()
        async with database.engine.begin() as conn:
            if (await conn.execute(text('SELECT COUNT(*) FROM messages'))).scalar_one():
                raise ValueError('Performance fixture must use an empty isolated database')
            for name in INDEX_NAMES:
                await conn.execute(text('DROP INDEX IF EXISTS ' + name))
            for start in range(0, len(rows), 1000):
                await conn.execute(text(INSERT), rows[start:start+1000])
                await conn.execute(text(RECEIPT), rows[start:start+1000])
        result = {}
        for phase in ('before', 'after'):
            async with database.engine.begin() as conn:
                if phase == 'after':
                    for statement in SOURCE_INDEX_STATEMENTS:
                        await conn.execute(text(statement))
                await conn.execute(text('ANALYZE messages'))
                await conn.execute(text('ANALYZE integration_receipts'))
            result[phase] = {}
            async with database.engine.connect() as conn:
                for name, (query, params) in queries().items():
                    await conn.execute(text(query), params)  # warmup, excluded
                    timings = []
                    for _ in range(5):
                        started = time.perf_counter()
                        found = (await conn.execute(text(query), params)).mappings().all()
                        timings.append(time.perf_counter() - started)
                    plan = (await conn.execute(text('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) ' + query), params)).scalar_one()
                    result[phase][name] = dict(seconds=timings, median=statistics.median(timings),
                        plan=plan, rows=len(found), digest=digest(found))
        for name in queries():
            assert result['before'][name]['digest'] == result['after'][name]['digest']
        for phase in result:
            assert result[phase]['older']['digest'] == result[phase]['older_tuple']['digest']
            assert result[phase]['older']['digest'] == result[phase]['older_split']['digest']
            assert result[phase]['older']['digest'] == result[phase]['runtime']['digest']
        return result
    finally:
        await database.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--noise', type=int, default=30000)
    parser.add_argument('--target', type=int, default=400)
    args = parser.parse_args()
    socket = args.socket.resolve()
    if (not socket.is_relative_to(ROOT) or not socket.parent.name.startswith('r89pg.')
            or not (socket / '.s.PGSQL.25433').exists()
            or args.output.resolve().parent != socket.parent or not 1000 <= args.noise <= 100000
            or not 400 <= args.target <= 100000):
        raise ValueError('Expected bounded fixture and isolated cluster')
    url = 'postgresql+asyncpg://boot@/postgres?host=' + str(socket) + '&port=25433'
    asyncio.run(verify_cluster(url, socket.parent / 'data'))
    os.environ['DATABASE_URL'] = url
    args.output.mkdir(exist_ok=False)
    rows = corpus(args.noise, args.target)
    result = dict(sqlite=sqlite_comparison(args.output / 'source.sqlite', rows),
                  postgres=asyncio.run(pg_comparison(url, rows)))
    for phase in ('before', 'after'):
        for name in queries():
            assert result['sqlite'][phase][name]['digest'] == result['postgres'][phase][name]['digest']
    (args.output / 'performance.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    sources = [Path(__file__), Path(turn_source_query.__code__.co_filename),
               Path(__file__).resolve().parents[1] / 'db/conversation_indexes.py']
    (args.output / 'manifest.json').write_text(json.dumps(dict(rows=len(rows),
        receipt_rows=len(rows), target_rows=args.target, noise_rows=args.noise,
        repetitions=5, warmup_excluded=True, model_calls=0,
        production_modified=False, cluster=str(socket.parent),
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}), indent=2), encoding='utf-8')
    print('PERFORMANCE_OK ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
