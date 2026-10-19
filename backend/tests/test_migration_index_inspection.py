"""A failed index inspection stops migration; valid schemas still migrate."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import OperationalError

from db.database import SQLiteDB


def migration(revision):
    filename = {3: '003_add_missing_indexes.py', 4: '004_normalize_redundant_indexes.py'}[revision]
    path = Path(__file__).resolve().parents[1] / 'alembic/versions' / filename
    spec = importlib.util.spec_from_file_location('index_inspection_' + str(revision), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('revision', [3, 4])
@pytest.mark.parametrize('operation', ['upgrade', 'downgrade'])
def test_failed_inspection_stops_before_schema_operations(monkeypatch, revision, operation):
    module = migration(revision)
    failure = OperationalError('synthetic index inspection', {}, OSError('synthetic unavailable'))
    inspector = SimpleNamespace(get_indexes=Mock(side_effect=failure))
    monkeypatch.setattr(module.sa, 'inspect', Mock(return_value=inspector))
    monkeypatch.setattr(module.op, 'get_bind', Mock(return_value=object()))
    create, drop = Mock(), Mock()
    monkeypatch.setattr(module.op, 'create_index', create)
    monkeypatch.setattr(module.op, 'drop_index', drop)
    with pytest.raises(OperationalError) as caught:
        getattr(module, operation)()
    assert caught.value is failure
    create.assert_not_called()
    drop.assert_not_called()


@pytest.mark.parametrize('revision', [3, 4])
def test_real_index_migration_is_idempotent_and_preserves_rows(tmp_path, revision):
    path = tmp_path / 'migration.sqlite'
    database = SQLiteDB(path)
    database.add_message(dict(sessionType='private', sessionId='synthetic', platform='web', adapter='web',
        conversationId='synthetic', conversationType='private', senderId='synthetic-owner', characterId='role',
        message='三张票，每张两元，尚未付款，下雨取消。', reply='共六元，尚未付款，下雨取消。'))
    database.close_connection()
    engine = create_engine('sqlite:///' + str(path))
    module = migration(revision)
    try:
        with engine.begin() as connection:
            before = {table: connection.exec_driver_sql('SELECT * FROM ' + table).all()
                      for table in ['messages', 'conversations']}
            with Operations.context(MigrationContext.configure(connection)):
                module.downgrade()
                module.upgrade()
                module.upgrade()
            if revision == 3:
                assert 'idx_training_tasks_status' in {item['name'] for item in inspect(connection).get_indexes('training_tasks')}
                assert 'idx_knowledge_chunks_documentId' in {item['name'] for item in inspect(connection).get_indexes('knowledge_chunks')}
            else:
                names = {item['name'] for item in inspect(connection).get_indexes('messages')}
                assert 'idx_messages_created_at' in names and 'idx_messages_createdAt' not in names
                assert 'idx_conversations_platform_conversation' not in {
                    item['name'] for item in inspect(connection).get_indexes('conversations')}
            assert before == {table: connection.exec_driver_sql('SELECT * FROM ' + table).all()
                              for table in ['messages', 'conversations']}
    finally:
        engine.dispose()
