import pytest
from test_rule_memory_correctness import save, scope

from character.memory_extractor import extract_memories
from character.models import CompiledCharacterContext, MemoryItem
from db.database import SQLiteDB
from inference.memory_response import render_memory_response
from repositories.character_memory import DatabaseCharacterMemoryRepository


@pytest.mark.parametrize('message,key,value', [
    ('更正一下，我的专业是建筑学，不是物理学。', 'user_major', '建筑学'),
    ('我的专业不是物理学，而是建筑学。', 'user_major', '建筑学'),
    ('我的名字不是周宁，是林安。', 'user_name', '林安'),
    ('我叫林安，不是周宁。', 'user_name', '林安'),
    ('纠正一下，我现在住在苏州，不是长沙。', 'user_residence', '苏州'),
    ('我来自绍兴，而不是宁波。', 'user_origin', '绍兴'),
    ('我在研究所工作，不是学校。', 'user_workplace', '研究所'),
    ('我不是大二，是大三。', 'user_study_stage', '大三'),
])
def test_same_slot_contrast_preserves_full_evidence(message, key, value):
    item, = extract_memories(message)
    assert item.memory_key == key and value in item.content
    assert item.evidence == message
    assert extract_memories(item.evidence) == [item]


@pytest.mark.parametrize('message', [
    '我的专业是建筑学，不是真的。',
    '我叫林安，不是我的真名。',
    '我叫林安，但不是我的真名。',
    '更正一下，我的专业是建筑学，不是事实。',
    '不是我的专业是建筑学，是物理学。',
    '我的专业不是物理学，是建筑学吗？',
    '我的专业可能是建筑学，不是物理学。',
    '假如我的专业是建筑学，不是物理学。',
    '她说我的专业是建筑学，不是物理学。',
    '小说台词：我的专业不是物理学，是建筑学。',
    '“我的专业是建筑学，不是物理学。”',
    '不要记录，我的专业是建筑学，不是物理学。',
    '密码是123，我的专业是建筑学，不是物理学。',
    '我的专业是建筑学，不是物理学，可能明年再换。',
    '我的专业不是物理学。',
    '我的专业是建筑学，不是建筑学。',
])
def test_contrast_does_not_promote_nonassertions(message):
    assert extract_memories(message) == []


@pytest.mark.asyncio
async def test_correction_versions_old_value_and_supports_deterministic_read(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'contrast.db'))
    await save(repo, '我的专业是物理学。', source='old')
    message = '更正一下，我的专业是建筑学，不是物理学。'
    from character.current_turn_memory import shadowed_memory_ids

    before = await repo.list_memory_records('kisaki', scope())
    assert shadowed_memory_ids(message, before) == {str(before[0]['id'])}
    await save(repo, message, source='correction')
    rows = await repo.list_memory_records('kisaki', scope(), include_inactive=True)
    assert len(rows) == 2
    old = next(row for row in rows if '物理学' in row['content'])
    new = next(row for row in rows if '建筑学' in row['content'])
    assert old['status'] == 'superseded' and old['valid_to']
    assert new['status'] == 'active' and new['evidence'] == [message]
    assert new['source_message_ids'] == ['correction']
    item = MemoryItem(new['id'], 'user_fact', new['content'], memory_key='user_major',
                      evidence=tuple(new['evidence']), source_message_ids=tuple(new['source_message_ids']))
    context = CompiledCharacterContext('', '', '', (item.memory_id,), memory_packets=(item,))
    reply = render_memory_response('我的专业是什么？', context)
    assert '建筑学' in reply and '物理学' not in reply
