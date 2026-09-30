from datetime import datetime, timezone

import pytest

from db.database import SQLiteDB

SCOPE = dict(character_id='role', platform='test', adapter='test', sender_id='user',
             conversation_type='private', conversation_id='room')


@pytest.mark.parametrize('query', ['背景说明。' * 500 + '唯一目标',
    ' '.join(f'term{i}' for i in range(600)) + ' unique_target'], ids=['long_text', 'many_terms'])
def test_long_query_retrieves_tail_terms_without_prefix_selection(tmp_path, query):
    db = SQLiteDB(tmp_path / 'sources.sqlite')
    db.capture_memory_source(**SCOPE, source_message_id='target', body='唯一目标 unique_target',
                             observed_at=datetime.now(timezone.utc))
    rows = db.search_memory_sources(**SCOPE, query=query)
    assert [r['source_message_id'] for r in rows] == ['target']


def test_long_query_still_obeys_scope_and_erasure(tmp_path):
    db = SQLiteDB(tmp_path / 'scope.sqlite')
    for user in ('user', 'other'):
        db.capture_memory_source(**(SCOPE | dict(sender_id=user)), source_message_id=user,
                                 body='unique_target', observed_at=datetime.now(timezone.utc))
    claim = db.append_character_memory_claim(**SCOPE, memory_type='user_fact', memory_key='key',
                                             content='fact', source_message_id='user')
    db.erase_character_memories(**SCOPE, memory_id=claim['id'])
    assert db.search_memory_sources(**SCOPE, query='背景。' * 800 + ' unique_target') == []


def test_long_plan_uses_constant_bound_parameters():
    from db.memory_source import source_scope
    from db.memory_source_search import search_plan

    query = ' '.join(f'term{i}' for i in range(600))
    sql, params = next(search_plan(source_scope(**SCOPE), query))
    assert 'json_each' in sql
    assert len(params) < 20
    assert 'term599' in params['query_terms']
