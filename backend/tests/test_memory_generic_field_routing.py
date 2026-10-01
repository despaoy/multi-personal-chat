"""Generic model labels must not hide independently proven personal fields."""

import json

import pytest

from character.context_builder import build_user_scope
from character.memory_llm import parse_llm_proposals
from character.memory_service import CharacterMemoryService
from character.models import MemoryItem
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,value,query,field",
    [
        ("我叫岚舟。", "岚舟", "我的名字是什么？", "name"),
        (
            "请跨会话记住，我叫岚舟，我的老家在宣城，目前住在丽水，大学专业是环境工程。这四项资料都是我本人的。",
            "岚舟",
            "我的名字是什么？",
            "name",
        ),
        ("我目前住在丽水。", "丽水", "我目前住哪里？", "residence"),
    ],
)
async def test_generic_proposal_is_visible_to_real_personal_field_recall(tmp_path, source, value, query, field):
    raw = dict(
        kind="other_user_fact",
        value=value,
        content="用户明确提到：" + source,
        evidence=source,
        confidence=0.98,
        operation="ADD",
        attributed_to="user",
    )
    (proposal,) = parse_llm_proposals(json.dumps({"memories": [raw]}, ensure_ascii=False), source_message=source)
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "fields.sqlite"))
    scope = build_user_scope(
        platform="web", adapter="test", sender_id="alice", conversation_id="alice", conversation_type="private"
    )
    await repo.append_claim(
        "kisaki",
        scope,
        MemoryItem("", proposal.memory.memory_type, proposal.memory.content, proposal.memory.importance),
        memory_key=proposal.memory.memory_key,
        evidence=(proposal.evidence,),
        source_message_id="source-1",
        confidence=proposal.confidence,
    )
    items, _, trace = await CharacterMemoryService(repo, semantic_enabled=False).recall_with_diagnostics(
        "kisaki", scope, query
    )
    assert trace["field_presence"][field] is True
    assert len(items) == 1 and value in items[0].content
    assert items[0].source_message_ids == ("source-1",)


@pytest.mark.parametrize(
    "source,value,qualifiers",
    [
        ("我来自淮安，我目前住在淮安。", "淮安", {}),
        ("我计划明年住在丽水。", "丽水", {}),
        ("我喜欢红茶，但只有白天才喝。", "红茶", {"condition": "只有白天才喝"}),
    ],
)
def test_ambiguous_or_qualified_generic_source_stays_generic(source, value, qualifiers):
    raw = dict(
        kind="other_user_fact",
        value=value,
        evidence=source,
        confidence=0.98,
        operation="ADD",
        attributed_to="user",
        qualifiers=qualifiers,
    )
    proposals = parse_llm_proposals(json.dumps({"memories": [raw]}, ensure_ascii=False), source_message=source)
    assert proposals
    assert all(p.memory.memory_key.startswith("fact_") for p in proposals)


def test_existing_generic_target_is_not_silently_migrated_to_a_different_key():
    source = "我叫岚舟。"
    target = dict(id="old", memory_key="fact_岚舟", content="用户明确提到：" + source, status="active")
    raw = dict(
        kind="other_user_fact",
        value="岚舟",
        evidence=source,
        confidence=0.98,
        operation="SUPERSEDE",
        target_memory_id="old",
        target_memory_key="fact_岚舟",
        attributed_to="user",
    )
    (proposal,) = parse_llm_proposals(
        json.dumps({"memories": [raw]}, ensure_ascii=False), source_message=source, existing_memories=(target,)
    )
    assert proposal.memory.memory_key == "fact_岚舟" and proposal.target_memory_id == "old"


def test_clipped_hypothesis_cannot_be_promoted_to_a_current_personal_field():
    source = "假设我目前住在丽水，那么通勤会方便。这只是设想，并非我的真实住址。"
    raw = dict(
        kind="other_user_fact",
        value="丽水",
        evidence="我目前住在丽水",
        confidence=0.98,
        operation="ADD",
        attributed_to="user",
    )
    proposals = parse_llm_proposals(json.dumps({"memories": [raw]}, ensure_ascii=False), source_message=source)
    assert not any(p.memory.memory_key == "user_residence" for p in proposals)
