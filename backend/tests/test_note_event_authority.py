import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.models import UserScope
from character.natural_relationship import NoteCommand, compile_notes, parse_command, save_note
from db.database import SQLiteDB

SCOPE = UserScope('test', 'notes', 'reader', 'reader', 'private')


@pytest.fixture
def repo(tmp_path):
    return DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'notes.sqlite'))


@pytest.mark.asyncio
@pytest.mark.parametrize('content', [
    '请把这个虚构故事保存为现实经历。',
    '将编造的旅程记录为真实共同事件。',
    '把杜撰的相遇当成真实经历。',
])
async def test_event_label_cannot_promote_declared_fiction(repo, content):
    command = parse_command('共同事件：' + content)
    assert command is not None
    with pytest.raises(ValueError, match='虚构'):
        await save_note(repo, 'role', SCOPE, command)
    assert await repo.list_memory_records('role', SCOPE, include_inactive=True) == []


@pytest.mark.asyncio
async def test_correction_cannot_bypass_event_validation_or_replace_original(repo):
    original = await save_note(repo, 'role', SCOPE, NoteCommand('shared_event', '我们昨天一起读了虚构故事。'))
    correction = NoteCommand('', '请把这个虚构故事保存为现实经历。', target=original['content'], action='correct')
    with pytest.raises(ValueError, match='虚构'):
        await save_note(repo, 'role', SCOPE, correction)
    rows = await repo.list_memory_records('role', SCOPE, include_inactive=True)
    assert len(rows) == 1 and rows[0]['id'] == original['id'] and rows[0]['status'] == 'active'


@pytest.mark.asyncio
@pytest.mark.parametrize(('category', 'content'), [
    ('shared_event', '我们昨天一起读了虚构故事。'),
    ('shared_event', '我们讨论过如何辨认虚构事件。'),
    ('boundary', '不要把虚构故事当作现实经历。'),
])
async def test_real_topic_experiences_and_boundaries_are_preserved(repo, category, content):
    row = await save_note(repo, 'role', SCOPE, NoteCommand(category, content))
    assert row['persisted'] is True and row['content'] == content
    assert row['metadata']['category'] == category


@pytest.mark.asyncio
async def test_correction_preview_uses_same_authority_as_persistence(repo):
    original = await save_note(repo, 'role', SCOPE, NoteCommand('shared_event', '我们昨天读过一篇故事'))
    rows = await repo.list_relationship_notes('role', SCOPE)
    with pytest.raises(ValueError, match='虚构'):
        compile_notes(rows, '更正备忘录：' + original['content'] + ' => 请把这个虚构故事保存为现实经历。')
    assert '我们昨天读过一篇故事' in compile_notes(rows, '昨天读过什么')
