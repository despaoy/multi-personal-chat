from datetime import datetime, timezone

import pytest

from db.database import SQLiteDB

SCOPE = dict(character_id='role', platform='web', adapter='audit', sender_id='user',
             conversation_type='private', conversation_id='room')


@pytest.mark.parametrize('memory_conversation', ['room', 'user'])
def test_erased_source_turn_not_reinjected_but_chat_record_and_other_turn_survive(tmp_path, memory_conversation):
    db = SQLiteDB(tmp_path / 'history.sqlite')
    scope = SCOPE | dict(conversation_id=memory_conversation)
    for source, body in [('deleted', '我的专业是园艺。'), ('kept', '继续讨论植物分类。')]:
        db.add_message(dict(platform='web', adapter='audit', senderId='user', characterId='role',
            conversationType='private', conversationId='room', sessionType='private', sessionId='room',
            sourceMessageId=source, message=body, reply='回复：' + body, createdAt='2026-09-27T12:00:00'))
        db.capture_memory_source(**scope, source_message_id=source, body=body,
                                 observed_at=datetime.now(timezone.utc))
    claim = db.append_character_memory_claim(**scope, memory_type='user_fact', memory_key='major',
                                             content='用户专业是园艺', source_message_id='deleted')
    db.erase_character_memories(**scope, memory_id=claim['id'])
    history = db.list_conversation_history('web', 'audit', 'user', 'private', 'room', character_id='role')
    assert [row['content'] for row in history] == ['继续讨论植物分类。', '回复：继续讨论植物分类。']
    assert db._get_connection().execute('SELECT COUNT(*) FROM messages').fetchone()[0] == 2


def test_same_external_source_id_in_other_scope_does_not_remove_history():
    from db.history_source_grants import filter_plan
    from db.memory_source import source_identity, source_scope

    rows = [dict(sourceMessageId='same', characterId=role, conversationType='private',
                 conversationId='room', message=role, reply='reply') for role in ('甲', '乙')]
    revoked = source_identity(source_scope(**(SCOPE | dict(character_id='甲'))), 'same')['source_key']
    plan = filter_plan(rows, 'web', 'audit', 'user')
    sql, params = next(plan)
    assert 'source_key IN' in sql and revoked in params.values()
    try:
        plan.send([dict(source_key=revoked)])
    except StopIteration as result:
        assert result.value == [rows[1]]
    else:
        raise AssertionError('filter did not finish')
