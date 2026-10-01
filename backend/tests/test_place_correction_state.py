"""Complete place corrections certify the new state without discarding denials."""

from copy import deepcopy
from datetime import timedelta

import pytest
from test_temporal_projection import NOW, OBSERVED, record

from character.context_builder import compile_reference_context
from character.memory_extractor import extract_memories
from character.memory_service import CharacterMemoryService
from character.models import CompiledCharacterContext, MemoryItem, UserScope
from character.temporal_projection import project_temporal_record
from db.database import SQLiteDB
from inference.memory_response import read_memory_fields, render_memory_response
from repositories.character_memory import DatabaseCharacterMemoryRepository


@pytest.mark.parametrize(
    "source,key,content",
    [
        ("我目前住在舟山，已经不住丽水了。", "user_residence", "用户说自己居住在舟山"),
        ("我现在住在舟山了，已经不住丽水了。", "user_residence", "用户说自己居住在舟山"),
        ("我现在住在舟山，不再住在丽水了。", "user_residence", "用户说自己居住在舟山"),
        ("更正一下，我现在住在舟山，我已经不再住在丽水了。", "user_residence", "用户说自己居住在舟山"),
        ("我已经搬到舟山了，已经不住丽水了。", "user_residence", "用户说自己居住在舟山"),
        ("我现在在华辰工作，已经不在启明工作了。", "user_workplace", "用户说自己在华辰工作"),
        ("我目前在华辰上班，我不再在启明上班了。", "user_workplace", "用户说自己在华辰工作"),
    ],
)
def test_complete_place_denial_is_preserved_and_certifies_the_affirmative_state(source, key, content):
    (item,) = extract_memories(source)
    assert item.memory_key == key and item.content == content and item.evidence == source
    assert not item.qualifiers
    row = record(source, "模型的概括不能决定值", key)
    before = deepcopy(row)
    view = project_temporal_record(row)
    assert view["temporal_mode"] == "asserted_state" and view["content"] == content
    assert view["valid_from"] == OBSERVED.isoformat() and view["valid_to"] == ""
    assert view["evidence"] == [source] and row == before


@pytest.mark.parametrize(
    "source",
    [
        "我现在住在舟山，已经不住舟山了。",
        "我现在住在舟山了，已经不住舟山了。",
        "我现在住在舟山，已经不住那里了。",
        "我现在住在舟山，她已经不住丽水了。",
        "我现在住在舟山，已经不在丽水工作了。",
        "我明年住在舟山，已经不住丽水了。",
        "假设我现在住在舟山，已经不住丽水了。",
        "我现在住在舟山，已经不住丽水了，下月搬走。",
        "我现在住在舟山，已经不住丽水了，但这只是小说设想。",
        "我现在住在舟山，已经不住丽水了吗？",
    ],
)
def test_place_correction_never_discards_conflict_owner_or_qualification(source):
    row = record(source, "用户说自己居住在舟山", "user_residence")
    before = deepcopy(row)
    view = project_temporal_record(row)
    assert view["temporal_mode"] == "observation" and source in view["content"]
    assert row == before
    if source in {"我现在住在舟山，已经不住舟山了。", "我现在住在舟山了，已经不住舟山了。"}:
        assert extract_memories(source) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,known",
    [
        ("我目前住在舟山，已经不住丽水了。", True),
        ("我现在住在舟山，已经不住舟山了。", False),
    ],
)
async def test_complete_semantic_correction_reads_status_value_and_original_version_chain(tmp_path, source, known):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "correction.sqlite"))
    scope = UserScope("web", "test", "alice", "alice", "private")
    old = await repo.append_claim(
        "kisaki",
        scope,
        MemoryItem("", "user_fact", "用户说自己居住在丽水", 0.9),
        memory_key="user_residence",
        evidence=("我住在丽水。",),
        source_message_id="old",
        valid_from=(OBSERVED - timedelta(days=1)).isoformat(),
    )
    await repo.capture_source("kisaki", scope, source_message_id="correction", body=source, observed_at=OBSERVED)
    new = await repo.append_claim(
        "kisaki",
        scope,
        MemoryItem("", "user_fact", "模型当前住址概括", 0.9),
        memory_key="user_residence",
        evidence=(source,),
        source_message_id="correction",
        relation_type="SUPERSEDE",
        supersedes_memory_id=old["id"],
        confidence=0.98,
        observed_at=OBSERVED.isoformat(),
        valid_from=OBSERVED.isoformat(),
        valid_to=OBSERVED.isoformat(),
        metadata={
            "temporal_provenance": {"version": 1, "producer": "semantic_memory", "validity_authority": "unverified"}
        },
    )
    before = await repo.list_memory_records("kisaki", scope, include_inactive=True)
    items, _, trace = await CharacterMemoryService(repo, semantic_enabled=False).recall_with_diagnostics(
        "kisaki", scope, "你保存了我的居住地吗？", reference_time=NOW
    )
    assert trace["field_presence"]["residence"] is (True if known else None)
    context = CompiledCharacterContext(
        "",
        "",
        "",
        tuple(item.memory_id for item in items),
        memory_status="available",
        memory_packets=items,
        memory_field_presence=tuple(trace["field_presence"].items()),
    )
    (result,) = read_memory_fields("我现在住哪里？", context)
    assert result.status == ("known" if known else "unverified")
    assert result.value == ("舟山" if known else None)
    text, used = compile_reference_context(items, complete_evidence=True, observation_semantics=True, max_chars=6000)
    assert str(new["id"]) in used and source in text
    if known:
        assert "有你的现居地记录" in render_memory_response("你保存了我的居住地吗？", context)
        assert new["supersedes_memory_id"] == old["id"]
    assert await repo.list_memory_records("kisaki", scope, include_inactive=True) == before
