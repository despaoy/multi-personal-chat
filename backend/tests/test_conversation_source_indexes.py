from character.context_builder import build_user_scope
from db.conversation_indexes import SOURCE_INDEX_STATEMENTS
from db.conversation_source import TurnCursor, turn_source_query
from db.database import SQLiteDB
from evaluation.conversation_source_performance import INSERT, RECEIPT, corpus


def test_candidate_indexes_preserve_result_and_enable_scoped_search(tmp_path):
    db = SQLiteDB(tmp_path / 'indexes.sqlite')
    conn = db._get_connection()
    indexes = {row['name'] for row in conn.execute('PRAGMA index_list(messages)')}
    assert {'idx_messages_private_source', 'idx_messages_local_source'} <= indexes
    for name in ('idx_messages_private_source', 'idx_messages_local_source'):
        conn.execute('DROP INDEX ' + name)
    rows = corpus(1000)
    conn.executemany(INSERT, rows)
    conn.executemany(RECEIPT, rows)
    conn.commit()
    queries = [turn_source_query(build_user_scope('web', 'audit', 'alice', 'room', kind), 'role', limit=10)
               for kind in ('private', 'group')]
    before = [list(map(tuple, conn.execute(query, params))) for query, params in queries]
    for statement in SOURCE_INDEX_STATEMENTS:
        conn.execute(statement)
        conn.execute(statement)  # installation is idempotent
    conn.execute('ANALYZE')
    after = [list(map(tuple, conn.execute(query, params))) for query, params in queries]
    assert before == after
    for index, (query, params) in enumerate(queries):
        plan = ' '.join(row['detail'] for row in conn.execute('EXPLAIN QUERY PLAN ' + query, params))
        expected = 'idx_messages_private_source' if index == 0 else 'idx_messages_local_source'
        assert expected in plan and 'senderId=?' in plan and 'characterId=?' in plan
        assert 'USE TEMP B-TREE FOR ORDER BY' not in plan


def test_cursor_seeks_same_time_id_and_matches_legacy_across_ranges(tmp_path):
    db = SQLiteDB(tmp_path / 'cursor.sqlite')
    conn = db._get_connection()
    rows = corpus(1000, 2000)
    for index, row in enumerate(rows):
        row['created'] = f'2026-09-{1 + index % 3:02d}T10:00:00'
    conn.executemany(INSERT, rows)
    conn.executemany(RECEIPT, rows)
    conn.commit()
    conn.execute('ANALYZE')
    for kind in ('private', 'group'):
        scope = build_user_scope('web', 'audit', 'alice', 'room', kind)
        for timestamp in ('2026-08-31', '2026-09-02T10:00:00', '2026-10-01'):
            for message_id in (1, 150, 1800, 4000):
                query, params = turn_source_query(scope, 'role', limit=40,
                    before=TurnCursor(timestamp, message_id))
                base, _ = turn_source_query(scope, 'role', limit=40)
                legacy = base.replace(' ORDER BY ', ' AND ("createdAt" < :before_at OR '
                    '("createdAt" = :before_at AND id < :before_id)) ORDER BY ')
                assert list(map(tuple, conn.execute(query, params))) == list(map(tuple, conn.execute(legacy, params)))
                plan = ' '.join(row['detail'] for row in conn.execute('EXPLAIN QUERY PLAN ' + query, params))
                assert 'createdAt=? AND id<?' in plan


def test_migration_is_idempotent_and_matches_orm_indexes(tmp_path):
    import importlib.util
    from pathlib import Path

    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import create_engine, inspect
    from sqlalchemy.schema import CreateIndex

    from db.models import Message

    db = SQLiteDB(tmp_path / 'migration.sqlite')
    db._get_connection().close()
    engine = create_engine('sqlite:///' + str(tmp_path / 'migration.sqlite'))
    path = Path(__file__).resolve().parents[1] / 'alembic/versions/011_conversation_source_indexes.py'
    spec = importlib.util.spec_from_file_location('source_index_migration', path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            migration.downgrade()
            migration.upgrade()
            migration.upgrade()
        names = {item['name'] for item in inspect(conn).get_indexes('messages')}
        assert {'idx_messages_private_source', 'idx_messages_local_source'} <= names
        for index in Message.__table__.indexes:
            if index.name in {'idx_messages_private_source', 'idx_messages_local_source'}:
                existing = conn.exec_driver_sql('SELECT sql FROM sqlite_master WHERE name=?', (index.name,)).scalar_one()
                def normalize(sql):
                    return ''.join(sql.lower().replace('"', '').replace('if not exists', '').split())
                assert normalize(existing) == normalize(str(CreateIndex(index).compile(engine)))
    engine.dispose()
