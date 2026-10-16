"""Revision-zero test records are historical data, never a production fallback."""


def seed_legacy_memory(repository, character_id, scope, memory, *, memory_key, source_message_id=None):
    record = repository._database.add_or_update_character_memory(
        character_id, scope.platform, scope.adapter, scope.sender_id,
        scope.conversation_type, scope.conversation_id, memory.memory_type,
        memory_key, memory.content, memory.importance, source_message_id,
    )
    assert record["revision"] == 0
    return record
