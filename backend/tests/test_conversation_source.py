from dataclasses import replace

import pytest

from character.context_builder import build_user_scope
from db.conversation_source import TurnCursor, turn_source_query
from db.database import SQLiteDB
from repositories.messages import DatabaseMessageRepository


def scope(**changes):
    return build_user_scope(**(dict(platform='web', adapter='audit', sender_id='alice',
        conversation_id='room', conversation_type='private') | changes))


def add(db, text, **changes):
    return db.add_message(dict(sessionId='room', sessionType='private', platform='web', adapter='audit',
        senderId='alice', characterId='role', message=text, reply='角色原话',
        createdAt='2026-09-26T10:00:00') | changes)


@pytest.fixture
def db(tmp_path):
    return SQLiteDB(tmp_path / 'turns.sqlite')


def test_speech_speaker_is_not_the_described_fact_subject(db):
    add(db, '朋友说：“我在学陶艺。”', reply='你的职业是建筑师。')
    page = db.list_scoped_conversation_turns(scope(), 'role')
    turn, = page.turns
    assert turn.utterances[0].speaker_id == 'alice'
    assert turn.utterances[0].text == '朋友说：“我在学陶艺。”'
    assert turn.utterances[1].speaker_id == 'role'
    assert turn.utterances[1].role == 'assistant'
    assert turn.utterances[1].text == '你的职业是建筑师。'
    assert not hasattr(turn.utterances[1], 'fact_subject')
    assert page.next_cursor is None


def test_keyset_paging_retains_all_same_timestamp_rows_without_duplicates(db):
    for index in range(7):
        add(db, str(index), sessionId='s' + str(index))
    cursor = None
    pages = []
    while True:
        page = db.list_scoped_conversation_turns(scope(), 'role', limit=2, before=cursor)
        pages.append(page)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    turns = [turn for page in reversed(pages) for turn in page.turns]
    assert [turn.utterances[0].text for turn in turns] == list(map(str, range(7)))
    assert len({turn.source_id for turn in turns}) == 7
    assert [page.older_rows_omitted for page in pages] == [True, True, True, False]


def test_paging_deletion_and_newer_insert_do_not_shift_offset(db):
    for index in range(5):
        add(db, str(index))
    first = db.list_scoped_conversation_turns(scope(), 'role', limit=2)
    # Only disposable SQLite test data is deleted.
    db.delete_message(first.turns[-1].source_id)
    add(db, 'newer')
    second = db.list_scoped_conversation_turns(scope(), 'role', limit=2, before=first.next_cursor)
    assert [row.utterances[0].text for row in second.turns] == ['1', '2']


def test_privacy_markers_and_empty_assistant_are_preserved_without_fact_promotion(db):
    add(db, '不要保存这个修改。', reply='')
    page = db.list_scoped_conversation_turns(scope(), 'role')
    assert len(page.turns[0].utterances) == 1
    assert page.turns[0].utterances[0].text == '不要保存这个修改。'


@pytest.mark.parametrize('limit', [True, 0, 2001, 1.5, '2'])
def test_invalid_bounds_rejected_before_database_query(limit):
    with pytest.raises(ValueError):
        turn_source_query(scope(), 'role', limit=limit)


@pytest.mark.parametrize('cursor', [TurnCursor('', 1), TurnCursor('date', 0),
    TurnCursor('date', True), ('date', 1)])
def test_invalid_cursor_rejected(cursor):
    with pytest.raises(ValueError):
        turn_source_query(scope(), 'role', before=cursor)


def test_cursor_and_identity_values_never_interpolated_into_sql():
    dangerous = "x' OR 1=1 --"
    query, params = turn_source_query(scope(sender_id=dangerous), 'role',
                                     before=TurnCursor(dangerous, 1))
    assert dangerous not in query
    assert params['sender'] == params['before_at'] == dangerous


def test_local_scope_cannot_be_unspecified():
    with pytest.raises(ValueError):
        turn_source_query(replace(scope(), conversation_type='group', conversation_id=''), 'role')


@pytest.mark.asyncio
async def test_repository_exposes_typed_page_and_passes_cursor(db):
    for index in range(3):
        add(db, str(index))
    repo = DatabaseMessageRepository(db)
    first = await repo.list_scoped_turns(scope(), 'role', limit=1)
    second = await repo.list_scoped_turns(scope(), 'role', limit=1, before=first.next_cursor)
    assert first.turns[0].utterances[0].text == '2'
    assert second.turns[0].utterances[0].text == '1'


@pytest.mark.asyncio
async def test_missing_adapter_is_not_claimed_to_be_empty_history():
    with pytest.raises(AttributeError):
        await DatabaseMessageRepository(object()).list_scoped_turns(scope(), 'role')
