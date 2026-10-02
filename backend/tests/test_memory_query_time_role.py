"""Date ownership must not erase an unrelated current personal lookup."""

from datetime import datetime, timezone

import pytest

from character.memory_extractor import extract_memories
from character.memory_service import CharacterMemoryService, _historical_query_window
from character.models import UserScope
from character.rule_memory_writer import write_rule_memory
from character.source_memory import SourceMemoryService
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

NOW = datetime(2026, 10, 2, 2, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "assertion",
    [
        "我出生于1998年。",
        "我在1998年出生，",
        "我是1998年出生的。",
        "我的出生年份是1998年。",
        "我1998年三月2日出生。",
        "我的出生日期为1998年3月2日。",
    ],
)
def test_complete_birth_assertion_is_not_current_residence_query_period(assertion):
    assert _historical_query_window(assertion + "我的现居地是什么？", NOW) is None


@pytest.mark.parametrize(
    "query,year,month",
    [
        ("我出生于1998年。2015年我当时的住址是什么？", 2015, 1),
        ("我1998年出生。2026年9月我的居住地是什么？", 2026, 9),
        ("我出生于1998年时住在哪里？", 1998, 1),
        ("我1998年出生吗？", 1998, 1),
        ("我出生于1998年。去年我的居住地是什么？", 2025, 1),
    ],
)
def test_actual_historical_question_preserves_its_own_time(query, year, month):
    window = _historical_query_window(query, NOW)
    assert window is not None and window.start.year == year and window.start.month == month


def test_unknown_or_quoted_birth_grammar_remains_conservative():
    for text in ["他说我出生于1998年。我的住址是什么？", "“我出生于1998年”。我的住址是什么？"]:
        assert _historical_query_window(text, NOW).start.year == 1998


async def test_actual_scoped_rule_correction_and_source_links_survive_unrelated_birth_year(tmp_path):
    database = SQLiteDB(tmp_path / "memory.sqlite")
    repo = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("web", "test", "alice", "alice", "private")
    for sid, text, stamp in [
        ("old", "我现在住在宁德。", datetime(2026, 10, 1, 1, tzinfo=timezone.utc)),
        ("new", "更正，我现在住在淮安。", datetime(2026, 10, 2, 1, tzinfo=timezone.utc)),
    ]:
        assert (
            await repo.capture_source("kisaki", scope, source_message_id=sid, body=text, observed_at=stamp)
            == "recorded"
        )
        (item,) = extract_memories(text, reference_time=stamp)
        assert await write_rule_memory(repo, "kisaki", scope, item, sid, observed_at=stamp)
    before = await repo.list_memory_records("kisaki", scope, limit=None, include_inactive=True)
    service = CharacterMemoryService(repo, semantic_enabled=False)
    question = "我出生于1998年。请告诉我，我的现居地是什么？"
    items, _, trace = await service.recall_with_diagnostics(
        "kisaki", scope, question, for_contextual_selection=True, reference_time=NOW
    )
    assert (
        len(items) == 1
        and items[0].memory_key == "user_residence"
        and "淮安" in items[0].content
        and not items[0].historical
        and items[0].source_message_ids == ("new",)
    )
    assert trace["field_presence"]["residence"] is True and trace["selected_count"] == 1
    source = await SourceMemoryService(repo).recall("kisaki", scope, question, memories=items)
    assert source.diagnostics["selected_ids"] == ["new"] and "更正，我现在住在淮安。" in source.context
    other = UserScope("web", "test", "bob", "bob", "private")
    foreign, _, _ = await service.recall_with_diagnostics(
        "kisaki", other, question, for_contextual_selection=True, reference_time=NOW
    )
    assert foreign == () and before == await repo.list_memory_records(
        "kisaki", scope, limit=None, include_inactive=True
    )
