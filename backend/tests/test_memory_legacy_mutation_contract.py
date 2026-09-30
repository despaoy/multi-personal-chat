import pytest

from character.context_builder import build_user_scope
from character.models import MemoryItem
from repositories.character_memory import DatabaseCharacterMemoryRepository


class LegacyDatabase:
    def __init__(self):
        self.writes = []

    def add_or_update_character_memory(self, *args):
        self.writes.append(args)
        return {'id': 1}


@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [
    {'relation_type': relation} for relation in ('RETRACT', 'PENDING', 'SUPERSEDE', 'MERGE', 'COEXIST')
] + [{'status': status} for status in ('pending', 'retracted', 'superseded', 'archived')]
  + [{'scope_level': 'user_global'}, {'scope_level': 'user_character'},
     {'parent_memory_id': 1}, {'supersedes_memory_id': 1}, {'valid_to': '2026-09-01'},
     {'valid_from': '2026-09-01'}])
async def test_unsupported_lifecycle_is_never_silently_saved_as_active(changes):
    db = LegacyDatabase()
    repo = DatabaseCharacterMemoryRepository(db)
    with pytest.raises(RuntimeError, match='版本'):
        await repo.append_claim('role', build_user_scope('web', 'test', 'user', 'room', 'private'),
            MemoryItem('', 'user_fact', '待处理的旧陈述'), memory_key='key', **changes)
    assert db.writes == []


@pytest.mark.asyncio
async def test_plain_legacy_add_and_noop_still_work():
    db = LegacyDatabase()
    repo = DatabaseCharacterMemoryRepository(db)
    scope = build_user_scope('web', 'test', 'user', 'room', 'private')
    memory = MemoryItem('', 'user_fact', '普通兼容记录')
    added = await repo.append_claim('role', scope, memory, memory_key='key')
    noop = await repo.append_claim('role', scope, memory, memory_key='key', relation_type='NOOP')
    assert added['persisted'] and added['compatibility_fallback']
    assert noop['persisted'] is False and len(db.writes) == 1
