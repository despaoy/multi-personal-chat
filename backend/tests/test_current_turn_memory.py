"""A current assertion can mask stale facts without pretending to save it."""
import pytest

from character.context_builder import build_user_scope
from character.current_turn_memory import shadowed_memory_ids
from character.memory_extractor import extract_memories
from character.memory_service import CharacterMemoryService
from character.rule_memory_writer import write_rule_memory
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository


def record(text, identifier=1):
    item, = extract_memories(text)
    return dict(id=identifier, memory_key=item.memory_key, content=item.content,
                status='active', metadata={'origin': 'rule_v2', 'qualifiers': dict(item.qualifiers)})


@pytest.mark.parametrize('old,new', [
    ('我的专业是数学。', '我的专业是物理。'),
    ('我在天文馆工作。', '我在档案馆工作。'),
    ('我住在北京。', '我住在上海。'),
    ('我只有周末才去游泳。', '更正一下，我只有放假才去游泳。'),
    ('我只有有空才看电影。', '纠正一下：我只有完成工作才看电影。'),
])
def test_typed_updates_mask_exact_slot_only(old, new):
    rows = [record(old), record('我来自南京。', 2)]
    assert shadowed_memory_ids(new, rows) == {'1'}
    assert rows[0]['status'] == 'active'


@pytest.mark.parametrize('new', [
    '我只有放假才去游泳。', '她说更正一下，我只有放假才去游泳。',
    '假设更正一下，我只有放假才去游泳。',
    '不要保存：更正一下，我只有放假才去游泳。',
    '更正一下，我只有放假才去游泳？',
    '更正一下，我只有放假才看电影。',
    '更正一下，我只有放假才去游泳。我只有有空才去游泳。',
])
def test_uncertain_or_other_slot_does_not_mask(new):
    assert not shadowed_memory_ids(new, [record('我只有周末才去游泳。')])


@pytest.mark.parametrize('change', [
    {'status': 'pending'}, {'scope_level': 'user_global'}, {'metadata': {}},
])
def test_unknown_origin_scope_and_lifecycle_not_masked(change):
    row = {**record('我的专业是数学。'), **change}
    assert not shadowed_memory_ids('我的专业是物理。', [row])


def test_ambiguous_active_rows_not_masked():
    rows = [record('我的专业是数学。'), record('我的专业是中文。', 2)]
    assert not shadowed_memory_ids('我的专业是物理。', rows)


@pytest.mark.asyncio
@pytest.mark.parametrize('old,new', [
    ('我的专业是数学。', '我的专业是物理。'),
    ('我只有周末才去游泳。', '更正一下，我只有放假才去游泳。'),
])
async def test_prepare_masks_old_packet_but_does_not_write_or_delete(tmp_path, monkeypatch, old, new):
    from services.character_context import TurnInput, build_character_context_service

    monkeypatch.setenv('CONTEXTUAL_MEMORY_SELECTION_ENABLED', 'false')
    db = SQLiteDB(tmp_path / 'overlay.sqlite')
    repo = DatabaseCharacterMemoryRepository(db)
    scope = build_user_scope('web', 'test', 'u1', 'old-session', 'private')
    item, = extract_memories(old)
    await write_rule_memory(repo, 'tsukiyashiro_kisaki', scope, item, 'old-source')
    service = build_character_context_service(db)
    service._memory_service = CharacterMemoryService(repo, semantic_enabled=False)
    prepared = await service.prepare_turn(
        TurnInput(new, 'web', 'test', 'u1', 'new-session', 'private'), 'tsukiyashiro_kisaki')
    assert not prepared.compiled.memory_packets
    assert old not in prepared.compiled.reference_context
    assert prepared.memory_recall['current_turn_shadowed_ids']
    rows = await repo.list_memory_records('tsukiyashiro_kisaki', scope, limit=None, include_inactive=True)
    assert len(rows) == 1 and rows[0]['status'] == 'active' and rows[0]['content'] == item.content
    # A failed/cancelled generation before complete_turn has not persisted the change.
    memories, _ = await service._memory_service.load_relevant_memories(
        'tsukiyashiro_kisaki', scope, '我的专业和限制有哪些？')
    assert len(memories) == 1 and memories[0].content == item.content


@pytest.mark.asyncio
@pytest.mark.parametrize('contextual,historical', [(True, False), (False, True)])
async def test_mask_precedes_selector_but_preserves_explicit_history(tmp_path, contextual, historical):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'selection.sqlite'))
    scope = build_user_scope('web', 'test', 'u1', 's1', 'private')
    old, = extract_memories('我只有周末才去游泳。')
    await write_rule_memory(repo, 'kisaki', scope, old, 'source')
    trace = {}
    items, count = await CharacterMemoryService(repo, semantic_enabled=False).load_relevant_memories(
        'kisaki', scope, '更正一下，我只有放假才去游泳。',
        for_contextual_selection=contextual, include_historical=historical, diagnostics=trace)
    assert count == 1 and trace['records_read'] == 1
    if historical:
        assert len(items) == 1 and items[0].historical
        assert not trace['current_turn_shadowed_ids']
    else:
        assert not items and trace['current_turn_shadowed_ids']
