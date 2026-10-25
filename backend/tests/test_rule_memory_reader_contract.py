"""Rule writes require the scoped repository reader, never a broad scan."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from character.memory_extractor import extract_memories
from character.models import build_user_scope
from character.rule_memory_writer import write_rule_memory


@pytest.mark.asyncio
@pytest.mark.parametrize("reader_state", ["missing", "failed", "present"])
async def test_rule_write_uses_only_the_scoped_reader(reader_state):
    item = extract_memories("我喜欢咖啡。")[0]
    scope = build_user_scope(
        platform="qq",
        adapter="nonebot",
        sender_id="alice",
        conversation_type="group",
        conversation_id="group",
    )
    repo = SimpleNamespace(list_memory_records=AsyncMock(), append_claim=AsyncMock())
    failure = OSError("scoped reader unavailable")
    if reader_state != "missing":
        repo.find_rule_memory_records = AsyncMock(
            side_effect=failure if reader_state == "failed" else None,
            return_value=[dict(memory_key=item.memory_key, content=item.content)],
        )
    if reader_state == "missing":
        with pytest.raises(AttributeError, match="find_rule_memory_records"):
            await write_rule_memory(repo, "character", scope, item)
    elif reader_state == "failed":
        with pytest.raises(OSError) as caught:
            await write_rule_memory(repo, "character", scope, item)
        assert caught.value is failure
    else:
        assert not await write_rule_memory(repo, "character", scope, item)
    if reader_state != "missing":
        repo.find_rule_memory_records.assert_awaited_once_with(
            "character",
            scope,
            tuple(sorted(set(item.aliases) | {item.memory_key})),
        )
    repo.list_memory_records.assert_not_awaited()
    repo.append_claim.assert_not_awaited()
