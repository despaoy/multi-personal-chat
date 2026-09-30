"""Real SQLite/PostgreSQL parity on a new isolated PostgreSQL cluster only.

Verifies cluster path before schema creation. No production credentials or
production DB URL are accepted. Tests speech-source retrieval, not generation.
"""
import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from character.context_builder import build_user_scope
from db.database import SQLiteDB
from repositories.messages import DatabaseMessageRepository

ROOT = Path('/home/boot/lhm/multipersonal-runtime/evaluations')


async def verify_cluster(url, expected):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            result = await connection.execute(text("SELECT current_setting('data_directory')"))
            if Path(result.scalar_one()).resolve() != expected:
                raise ValueError('Refusing writes to unexpected PostgreSQL cluster')
    finally:
        await engine.dispose()


def add(db, message, **changes):
    return db.add_message(dict(sessionId='room', sessionType='private', platform='web', adapter='audit',
        senderId='alice', characterId='role', message=message, reply='角色原话',
        createdAt='2026-09-27T10:00:00') | changes)


async def exercise(db):
    scope = build_user_scope('web', 'audit', 'alice', 'room', 'private')
    for index in range(7):
        add(db, str(index), sessionId='session' + str(index))
    for changes in (dict(senderId='bob'), dict(characterId='other'), dict(characterId=None),
                    dict(platform='other'), dict(adapter='other'), dict(branchId='branch'),
                    dict(sessionType='group'), dict(sessionType='channel')):
        add(db, '不应进入私聊来源', **changes)
    add(db, '未交付', traceId='failed')
    db.integration_receipt('claim', key='receipt', owner='failed', now=0, expires_at=100)
    db.integration_receipt('finish', key='receipt', owner='failed', status='delivery_failed', response='')
    repo = DatabaseMessageRepository(db)
    pages = []
    cursor = None
    while True:
        page = await repo.list_scoped_turns(scope, 'role', limit=2, before=cursor)
        pages.append(page)
        cursor = page.next_cursor
        if cursor is None:
            break
    turns = [turn for page in reversed(pages) for turn in page.turns]
    assert [turn.utterances[0].text for turn in turns] == list(map(str, range(7)))
    assert len({turn.source_id for turn in turns}) == 7
    assert all(turn.utterances[0].speaker_id == 'alice' and turn.utterances[1].speaker_id == 'role'
               for turn in turns)
    for kind in ('group', 'channel'):
        local = build_user_scope('web', 'audit', 'alice', 'room', kind)
        result = await repo.list_scoped_turns(local, 'role')
        assert len(result.turns) == 1
        other = build_user_scope('web', 'audit', 'alice', 'other-room', kind)
        assert not (await repo.list_scoped_turns(other, 'role')).turns
    original_count = db.get_message_count()
    add(db, '不要保存这次修改。', reply='')
    latest = await repo.list_scoped_turns(scope, 'role', limit=1)
    assert latest.turns[0].utterances[0].text == '不要保存这次修改。'
    assert len(latest.turns[0].utterances) == 1
    assert db.get_message_count() == original_count + 1
    db.delete_message(latest.turns[0].source_id)
    assert (await repo.list_scoped_turns(scope, 'role', limit=1)).turns[0].source_id == turns[-1].source_id
    return dict(pages=[asdict(page) for page in pages], cases_checked=[
        'same_time_keyset', 'scope_before_limit', 'speaker_not_subject', 'role_isolation',
        'delivery_filter', 'branch_filter', 'group_channel_isolation', 'privacy_marker_preserved',
        'source_deletion_visible', 'read_does_not_write'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    socket = args.socket.resolve()
    expected = socket.parent / 'data'
    if (not socket.is_relative_to(ROOT) or not socket.parent.name.startswith('r88pg.')
            or not (socket / '.s.PGSQL.25433').exists()
            or args.output.resolve().parent != socket.parent):
        raise ValueError('Only this isolated local cluster is accepted')
    url = 'postgresql+asyncpg://boot@/postgres?host=' + str(socket) + '&port=25433'
    asyncio.run(verify_cluster(url, expected))
    args.output.mkdir(exist_ok=False)
    # pg_database also constructs a module-level adapter at import time.
    # Bind it to the already-verified disposable cluster, never ambient config.
    os.environ['DATABASE_URL'] = url
    from db.pg_database import PgDatabase, SyncPgAdapter

    pg = SyncPgAdapter(PgDatabase(url))
    sqlite = SQLiteDB(args.output / 'source.sqlite')
    try:
        results = {'sqlite': asyncio.run(exercise(sqlite)), 'postgres': asyncio.run(exercise(pg))}
        assert results['sqlite'] == results['postgres']
        (args.output / 'parity.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        backend = Path(__file__).resolve().parents[1]
        paths = [Path(__file__), *(backend / name for name in ('db/conversation_source.py',
            'db/database.py', 'db/pg_database.py', 'repositories/messages.py'))]
        (args.output / 'manifest.json').write_text(json.dumps(dict(
            sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            postgres_real=True, cluster_data=str(expected), production_modified=False,
            model_calls=0, http_e2e=False), indent=2), encoding='utf-8')
        print('PARITY_OK ' + str(args.output), flush=True)
    finally:
        pg.close()


if __name__ == '__main__':
    main()
