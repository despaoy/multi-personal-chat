"""Independent tasks require a union of their scoped temporal evidence."""

from datetime import datetime, timezone

import pytest

from character.memory_extractor import extract_memories
from character.memory_query_time import personal_time_tasks
from character.memory_service import CharacterMemoryService
from character.models import UserScope
from character.rule_memory_writer import write_rule_memory
from character.source_memory import SourceMemoryService
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository


@pytest.mark.parametrize(
    "query,fields",
    [
        ("2026年9月我的居住地是什么？请告诉我，我的现居地是什么？", [("residence",), ("residence",)]),
        ("我的现居地是什么？2026年9月我的居住地是什么？", [("residence",), ("residence",)]),
        ("2026年9月我的住址是什么？2026年10月我的住址是什么？", [("residence",), ("residence",)]),
        ("去年我的专业是什么？我的当前工作单位是什么？", [("major",), ("workplace",)]),
        ("前年我的姓名是什么？我现在住在哪里？", [("name",), ("residence",)]),
        ("2026年三月我的名字和专业分别是什么？我的当前工作单位是什么？", [("name", "major"), ("workplace",)]),
    ],
)
def test_complete_task_parse_keeps_original_text_and_shared_fields(query, fields):
    tasks = personal_time_tasks(query)
    assert [task.fields for task in tasks] == fields
    assert all(task.query in query for task in tasks)


@pytest.mark.parametrize(
    "query",
    [
        "我在2017年参加培训。我的现居地是什么？",
        "2026年9月。我的现居地是什么？",
        "2026年9月你住哪里？我的现居地是什么？",
        "2026年9月我的室友住哪里？我的现居地是什么？",
        "“2026年9月我的住址是什么”。我的现居地是什么？",
        "那时我的居住地是什么？我的现居地是什么？",
        "2026年9月我的住址是什么？请说明原因。",
    ],
)
def test_unknown_clause_or_other_owner_defers_the_whole_query(query):
    assert personal_time_tasks(query) == ()


@pytest.mark.parametrize(
    "query,override,expected",
    [
        ("2026年9月我的居住地是什么？请告诉我，我的现居地是什么？", None, {"old": True, "new": False}),
        ("请告诉我，我的现居地是什么？2026年9月我的居住地是什么？", None, {"old": True, "new": False}),
        ("2026年9月我的居住地是什么？2026年10月我的居住地是什么？", None, {"old": True, "new": True}),
        ("2026年10月我的居住地是什么？2026年9月我的居住地是什么？", None, {"old": True, "new": True}),
        ("2026年9月我的居住地是什么？我的现居地是什么？", False, {"new": False}),
        ("2026年9月我的居住地是什么？我的现居地是什么？", True, {"old": True}),
    ],
)
async def test_actual_source_rule_versions_task_union_and_override(tmp_path, query, override, expected):
    db = SQLiteDB(tmp_path / "memory.sqlite")
    repo = DatabaseCharacterMemoryRepository(db)
    owner = UserScope("web", "test", "alice", "alice", "private")
    for sid, text, stamp in [
        ("old", "我现在住在德阳。", datetime(2026, 9, 30, 1, tzinfo=timezone.utc)),
        ("new", "更正，我现在住在晋中。", datetime(2026, 10, 1, 1, tzinfo=timezone.utc)),
    ]:
        assert (
            await repo.capture_source("kisaki", owner, source_message_id=sid, body=text, observed_at=stamp)
            == "recorded"
        )
        (item,) = extract_memories(text, reference_time=stamp)
        assert await write_rule_memory(repo, "kisaki", owner, item, sid, observed_at=stamp)
    original = await repo.list_memory_records("kisaki", owner, limit=None, include_inactive=True)
    items, _, trace = await CharacterMemoryService(repo, semantic_enabled=False).recall_with_diagnostics(
        "kisaki",
        owner,
        query,
        reference_time=datetime(2026, 10, 2, 2, tzinfo=timezone.utc),
        for_contextual_selection=True,
        include_historical=override,
    )
    assert {item.source_message_ids[0]: item.historical for item in items} == expected
    assert all(item.status == ("superseded" if item.source_message_ids == ("old",) else "active") for item in items)
    if override is None:
        assert len(trace["personal_time_tasks"]) == 2
        assert all(row["covered_fields"] == ["residence"] for row in trace["personal_time_tasks"])
    else:
        assert not trace["personal_time_tasks"]
    sources = await SourceMemoryService(repo).recall("kisaki", owner, query, memories=items)
    assert set(sources.diagnostics["selected_ids"]) == set(expected)
    other = UserScope("web", "test", "bob", "bob", "private")
    foreign, _, _ = await CharacterMemoryService(repo, semantic_enabled=False).recall_with_diagnostics(
        "kisaki",
        other,
        query,
        reference_time=datetime(2026, 10, 2, 2, tzinfo=timezone.utc),
        for_contextual_selection=True,
    )
    assert not foreign
    assert await repo.list_memory_records("kisaki", owner, limit=None, include_inactive=True) == original


async def test_different_fields_do_not_share_each_others_time_windows(tmp_path):
    db = SQLiteDB(tmp_path / "fields.sqlite")
    repo = DatabaseCharacterMemoryRepository(db)
    owner = UserScope("web", "test", "alice", "alice", "private")
    for sid, body, stamp in [
        ("old-name", "我叫若岚。", datetime(2026, 9, 30, 1, tzinfo=timezone.utc)),
        ("old-work", "我在海泽研究院工作。", datetime(2026, 9, 30, 1, tzinfo=timezone.utc)),
        ("new-name", "更正，我叫清禾。", datetime(2026, 10, 1, 1, tzinfo=timezone.utc)),
        ("new-work", "更正，我在青原研究所工作。", datetime(2026, 10, 1, 1, tzinfo=timezone.utc)),
    ]:
        assert (
            await repo.capture_source("kisaki", owner, source_message_id=sid, body=body, observed_at=stamp)
            == "recorded"
        )
        (item,) = extract_memories(body, reference_time=stamp)
        assert await write_rule_memory(repo, "kisaki", owner, item, sid, observed_at=stamp)
    original = await repo.list_memory_records("kisaki", owner, limit=None, include_inactive=True)
    items, _, trace = await CharacterMemoryService(repo, semantic_enabled=False).recall_with_diagnostics(
        "kisaki",
        owner,
        "2026年9月我的姓名是什么？我的当前工作单位是什么？",
        reference_time=datetime(2026, 10, 2, 2, tzinfo=timezone.utc),
        for_contextual_selection=True,
    )
    assert {item.source_message_ids[0]: item.historical for item in items} == {"old-name": True, "new-work": False}
    assert [task["covered_fields"] for task in trace["personal_time_tasks"]] == [["name"], ["workplace"]]
    assert await repo.list_memory_records("kisaki", owner, limit=None, include_inactive=True) == original
