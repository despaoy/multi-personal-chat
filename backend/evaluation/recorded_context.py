"""Read archived context metadata without re-enabling retired experiments."""

from character.models import CompiledCharacterContext, MemoryItem


def restore_recorded_context(record):
    data = dict(record)
    data.pop("source_shared_memory_ids", None)
    data.pop("source_reference_backup", None)
    data["memory_packets"] = tuple(MemoryItem(**item) for item in data.get("memory_packets", ()))
    return CompiledCharacterContext(**data)
