"""Complete literal source reads keep speech authority and serving budget separate."""

import json
from datetime import datetime, timezone

import pytest

from character.models import UserScope
from character.source_memory import SourceMemoryService
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

SCOPE = UserScope("web", "literal-recall", "alice", "room", "private")
FIELDS = dict(character_id="role", platform="web", adapter="literal-recall", sender_id="alice", conversation_type="private", conversation_id="room")
BODY = '第三方青舟完整目录：条目 QZ8-07-001，路线西区，费用13元，时限6小时，容量8，核验true。完整前提是登记页及封签页齐全；封签缺失必须暂停，付款不能替代核验。这是转述资料，未确认本人办理或角色经历。'


def put(db, identity, body, **extra):
    db.capture_memory_source(**(FIELDS | extra), source_message_id=identity, body=body, observed_at=datetime.now(timezone.utc))


@pytest.mark.asyncio
@pytest.mark.parametrize("opening,closing,verb", [("「", "」", "查找"), ("“", "”", "检索"), ("『", "』", "读取")])
async def test_literal_whole_source_survives_unrelated_complete_history(tmp_path, opening, closing, verb):
    db = SQLiteDB(tmp_path / "reads.sqlite")
    put(db, "required", BODY)
    noise = '第三方其他目录：路线西区、费用、时限、容量、核验及前提例外已完整提供；' + '另一个目录的登记页和封签页齐全，未完成实际办理。' * 300
    for i in range(4):
        put(db, "noise-" + str(i), noise + str(i))
    put(db, "foreign", BODY, sender_id="bob")
    query = f'{verb}包含{opening}QZ8-07-001{closing}的完整原话记录。依据该原始目录核对路线、费用、时限、容量、核验、完整前提和例外，不推断实际办理结果。'
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), max_chars=16384, defer_budget=True).recall("role", SCOPE, query, retrieval_context=noise)
    assert result.diagnostics["status"] == "available"
    assert json.loads(result.context)["records"] == [{"source_id": "required", "observed_at": db.search_memory_sources(**FIELDS, query="QZ8")[0]["observed_at"], "text": BODY}]
    assert not result.diagnostics["contextual_search_enabled"]
    assert await DatabaseCharacterMemoryRepository(db).list_memory_records("role", SCOPE) == []


@pytest.mark.parametrize("query,expected", [
    ('请查看含有“青舟完整目录”的全部用户原话。核对前提和例外。', '青舟完整目录'),
    ('读取包含「登记页及封签页齐全」的完整原始发言记录。核对这份转述资料。', '登记页及封签页齐全'),
    ('检索包含"QZ8-07-001"的原话记录。核对未知实际办理结果。', 'QZ8-07-001'),
    ('  查找包含『QZ8-07-001』的完整原话记录。', 'QZ8-07-001'),
])
def test_literal_reader_resolves_only_exact_locator(query, expected):
    from db.memory_source_search import literal_source_fragment
    assert literal_source_fragment(query) == expected


@pytest.mark.parametrize("query", [
    '朋友说：“查找包含「QZ8-07-001」的完整原话记录。”',
    '不要查找包含「QZ8-07-001」的完整原话记录。',
    '查找包含「QZ8-07-001」和「另一项」的完整原话记录。',
    '查找包含「QZ8-07-001」的完整原话记录。另读取包含「另一项」的原话。',
    '查找包含「QZ8-07-001」的完整原话记录。不要读取原始资料。',
    '查找包含「QZ8-07-001」的完整原话记录。同时核对另一段原始发言。',
    '查找包含「QZ8-07-001的完整原话记录。',
    '查找包含「编号“QZ8-07-001”」的完整原话记录。',
    '查找包含「QZ8-07-001\n封签」的完整原话记录。',
    '仅提到「QZ8-07-001」，并未指定原话读取范围。',
    '查找包含「   」的完整原话记录。',
])
def test_unresolved_or_quoted_read_does_not_narrow_source_candidates(query):
    from db.memory_source_search import literal_source_fragment
    assert literal_source_fragment(query) == ""


def test_exact_locator_uses_bound_literal_and_scoped_current_grant(tmp_path):
    db = SQLiteDB(tmp_path / "bound.sqlite")
    locator = "QZ8_%'--"
    body = "完整第三方目录编号为" + locator + "，登记页和封签页缺一不可；付款不构成例外。"
    put(db, "exact", body)
    put(db, "wildcard-lookalike", "QZ8_anything；第三方完整目录未核验。")
    put(db, "foreign", body, conversation_id="another-room")
    query = '查找包含「' + locator + '」的完整原话记录。核对前提和例外。'
    rows = db.search_memory_sources(**FIELDS, query=query, limit=None)
    assert [r["source_message_id"] for r in rows] == ["exact"]
    assert rows[0]["body"] == body
    db.clear_character_memories(**FIELDS)
    assert db.search_memory_sources(**FIELDS, query=query, limit=None) == []


@pytest.mark.asyncio
async def test_all_literal_matching_versions_are_whole_unclassified_speech(tmp_path):
    db = SQLiteDB(tmp_path / "versions.sqlite")
    newer = '青舟完整目录 QZ8-07-001 的第三方修订：费用15元，只有登记页和封签页齐全才可受理。之前的13元只是旧资料，实际办理未知。'
    put(db, "old", BODY)
    put(db, "new", newer)
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), defer_budget=True).recall("role", SCOPE, '读取包含「QZ8-07-001」的完整原话记录。比较各版本，保留未验证的实际结果。')
    packet = json.loads(result.context)
    assert {r["text"] for r in packet["records"]} == {BODY, newer}
    assert packet["described_subject"] == packet["current_validity"] == "not_resolved"
    assert await DatabaseCharacterMemoryRepository(db).list_memory_records("role", SCOPE) == []


@pytest.mark.asyncio
async def test_absent_locator_has_no_fuzzy_or_history_substitution(tmp_path):
    db = SQLiteDB(tmp_path / "missing.sqlite")
    put(db, "similar", BODY)
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), defer_budget=True).recall("role", SCOPE, '查找包含「QZ8-07-099」的完整原话记录。核对全部字段。', retrieval_context=BODY)
    assert result.diagnostics["status"] == "no_match"
    assert not result.context and not result.candidate_context
    assert not result.diagnostics["contextual_search_enabled"]


@pytest.mark.asyncio
async def test_fresh_recheck_cannot_restore_erased_literal_original(tmp_path):
    db = SQLiteDB(tmp_path / "race.sqlite")
    put(db, "target", BODY)

    class ErasingRepository(DatabaseCharacterMemoryRepository):
        async def search_sources(self, *args, **kwargs):
            rows = await super().search_sources(*args, **kwargs)
            db.clear_character_memories(**FIELDS)
            return rows

    result = await SourceMemoryService(ErasingRepository(db), defer_budget=True).recall("role", SCOPE, '查找包含「QZ8-07-001」的完整原话记录。核对完整例外。')
    assert result.context == result.candidate_context == ""
    assert result.diagnostics["fresh_recheck_omitted"] == 1


@pytest.mark.asyncio
async def test_largest_matching_original_is_not_cut_or_replaced(tmp_path):
    db = SQLiteDB(tmp_path / "whole.sqlite")
    body = BODY + '完整第三方目录的详细说明。' * 2000 + '以上仅为小说资料，不是本人实际安排。'
    put(db, "required", body)
    put(db, "small-other", "其他完整原话目录。")
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), max_chars=16384, defer_budget=True).recall("role", SCOPE, '读取包含「QZ8-07-001」的完整原话记录。保留结尾的条件、否定及主体。')
    assert result.context == "" and result.diagnostics["status"] == "budget_omitted"
    assert json.loads(result.candidate_context)["records"][0]["text"] == body


@pytest.mark.asyncio
async def test_linked_source_cannot_escape_literal_read_scope(tmp_path):
    from character.models import MemoryItem
    db = SQLiteDB(tmp_path / "linked.sqlite")
    put(db, "required", BODY)
    put(db, "linked-other", '我此前在其他目录登记过，当前办理结果未知。')
    record = db.append_character_memory_claim(**FIELDS, memory_type="user_fact", memory_key="example", content="其他目录", source_message_id="linked-other")
    memory = MemoryItem(str(record["id"]), "user_fact", "其他目录", source_message_ids=("linked-other",))
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), defer_budget=True).recall("role", SCOPE, '查看包含「QZ8-07-001」的完整原话记录。保留全部目录字段。', memories=(memory,))
    assert [r["text"] for r in json.loads(result.context)["records"]] == [BODY]
    assert result.diagnostics["linked_fragment_scope_omitted"] == 1
    assert len(await DatabaseCharacterMemoryRepository(db).list_memory_records("role", SCOPE)) == 1
