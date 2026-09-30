"""Closed memory reads preserve speech without redundant semantic writes."""
import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import UserScope
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "我叫什么名字？", "你还记得我的姓名吗？", "我的专业和年级是什么？",
    "我来自哪里，目前住哪里？", "请告诉我我的工作单位。",
    "我的住址呢？", "我的家乡",
])
async def test_complete_reads_capture_source_without_lookup_or_model(tmp_path, message):
    class SourceOnlyRepository(DatabaseCharacterMemoryRepository):
        async def list_memory_records(self, *args, **kwargs):
            raise AssertionError("Pure read must not load facts for another write")

    class NoModel:
        async def complete(self, messages):
            raise AssertionError("Pure read must not invoke semantic writer")

        async def close(self):
            pass

    database = SQLiteDB(tmp_path / "read.sqlite")
    repo = SourceOnlyRepository(database)
    scope = UserScope("web", "web", "reader", "reader", "private")
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(True, "unused", "unused"), completion=NoModel())
    try:
        assert scheduler.schedule(repository=repo, character_id="role", user_scope=scope,
                                  message=message, rule_hints=[], source_message_id="read")
        assert await scheduler.flush_memory(timeout=3)
        assert scheduler.status.failed == 0
        result = scheduler.status.recent_results[-1]
        assert result["status"] == "source_only"
        assert result["source_capture"] == "recorded"
        assert [row["body"] for row in await repo.list_sources("role", scope)] == [message]
    finally:
        await scheduler.shutdown(timeout=3)


@pytest.mark.asyncio
@pytest.mark.parametrize("message,immediate", [
    ("我叫林青，我住哪里？", None),
    ("我不喝酒，你记得吗？", None),
    ("我搬到外地了，你知道吗？", None),
    ("我的名字不是林青，是林秋。", None),
    ("更正一下，我的名字是什么？", None),
    ("请记住我的工作单位。", False),
    ("忘掉我的住址。", None),
    ("我还有哪些安排？", None),
    ("‘我的专业是什么’是练习台词。", None),
    ("我叫什么名字？", True),
])
async def test_mixed_unknown_and_explicit_writes_keep_semantic_path(tmp_path, message, immediate):
    class Completion:
        calls = 0

        async def complete(self, messages):
            self.calls += 1
            return '{"memories":[]}'

        async def close(self):
            pass

    completion = Completion()
    database = SQLiteDB(tmp_path / "mixed.sqlite")
    repo = DatabaseCharacterMemoryRepository(database)
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(True, "unused", "unused"), completion=completion)
    try:
        assert scheduler.schedule(repository=repo, character_id="role",
                                  user_scope=UserScope("web", "web", "u", "u", "private"),
                                  message=message, rule_hints=[], source_message_id="mixed",
                                  immediate=immediate)
        assert await scheduler.flush_memory(timeout=3)
        assert scheduler.status.failed == 0
        assert completion.calls == 1
    finally:
        await scheduler.shutdown(timeout=3)
