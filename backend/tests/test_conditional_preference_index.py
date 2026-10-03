"""Independent full synthetic unit fixtures, not native model outputs or seeds."""

import json
from types import SimpleNamespace

import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, parse_llm_proposals
from character.models import MemoryItem
from character.temporal_projection import project_temporal_record
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository
from tests.test_deferred_evidence_binding import NOW, SCOPE

HEAD = "测试紫茶"
CONDITION = "只有表格已提交和预约确认通过才饮用测试紫茶"
EVIDENCE = "从2027年5月15日起，\n我喜欢\t\u3000测试紫茶且" + CONDITION + "。这不是已经实际饮用。"
SOURCE = "请记住我本人的完整未来条件。\n" + EVIDENCE + "该日期前现有条件继续有效。"


def candidate(operation="ADD", **changes):
    raw = dict(
        kind="like",
        value="测试紫茶（仅表格已提交且预约确认通过时）",
        content="用户未来喜欢测试紫茶且" + CONDITION + "，不表示已经实际饮用。",
        evidence=EVIDENCE,
        confidence=0.95,
        operation=operation,
        attributed_to="user",
        qualifiers={"condition": CONDITION, "time": "2027年5月15日", "certainty": "这不是已经实际饮用"},
    )
    return raw | changes


def parse(raw, source=SOURCE, existing=()):
    return parse_llm_proposals(
        json.dumps(dict(memories=[raw]), ensure_ascii=False), source_message=source, existing_memories=existing
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["ADD", "SUPERSEDE", "COEXIST", "MERGE"])
async def test_complete_conditional_label_preserves_short_index_full_condition_and_current_target(tmp_path, operation):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "conditional-index.sqlite"))
    old = await repo.append_claim(
        "unit-character",
        SCOPE,
        MemoryItem(memory_id="", memory_type="user_fact", content="用户喜欢测试紫茶", importance=0.6),
        memory_key="preference_测试紫茶",
        observed_at="2026-10-02T12:00:00+00:00",
        evidence=("我喜欢测试紫茶。",),
    )
    old.pop("persisted", None)
    await repo.capture_source(
        "unit-character", SCOPE, source_message_id="conditional-source", body=SOURCE, observed_at=NOW
    )
    raw = candidate(operation)
    if operation != "ADD":
        raw.update(target_memory_id=str(old["id"]), target_memory_key=old["memory_key"])
    proposals = parse(raw, existing=(old,))
    assert len(proposals) == 1
    proposal = proposals[0]
    assert dict(proposal.qualifiers)["condition"] == CONDITION
    assert proposal.memory.memory_key == "preference_测试紫茶"
    assert "仅表格已提交且预约确认通过时" not in proposal.memory.content
    assert CONDITION in proposal.memory.content
    job = SimpleNamespace(
        repository=repo,
        character_id="unit-character",
        user_scope=SCOPE,
        message=SOURCE,
        observed_at=NOW,
        source_message_id="conditional-source",
        write_mode="hot",
        feedback_target_ids=(),
    )
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unit-only", "unit-only"), completion=None)
    assert await scheduler._persist_proposal(job, proposal) == "saved"
    rows = await repo.list_memory_records("unit-character", SCOPE, include_inactive=True, limit=None)
    assert next(r for r in rows if r["id"] == old["id"]) == old
    new = next(r for r in rows if r["id"] != old["id"])
    assert new["metadata"]["qualifiers"]["condition"] == CONDITION
    linked = await repo.linked_source_receipts(
        "unit-character", SCOPE, claim_sources=((new["id"], job.source_message_id),)
    )
    assert linked[0]["body"] == SOURCE
    view = project_temporal_record(new, {r["source_message_id"]: r for r in linked})
    assert view["temporal_mode"] == "observation"
    assert SOURCE in json.loads(view["content"].split("：", 1)[1])
    if operation == "SUPERSEDE":
        assert new["relation_type"] == "COEXIST" and new["parent_memory_id"] == old["id"]


@pytest.mark.parametrize(
    "changes,source",
    [
        ({"value": "测试紫茶（仅表格已提交或预约确认通过时）"}, SOURCE),
        ({"value": "测试紫茶（仅表格已提交时）"}, SOURCE),
        ({"value": "测试紫茶（仅表格已提交且预约确认通过且身份核验通过时）"}, SOURCE),
        ({"value": "测试紫茶（仅表格未提交且预约确认通过时）"}, SOURCE),
        ({"qualifiers": {}}, SOURCE),
        ({"qualifiers": {"condition": "表格已提交和预约确认通过"}}, SOURCE),
        ({"qualifiers": {"condition": "只有表格已提交和身份核验通过才饮用测试紫茶"}}, SOURCE),
        ({"evidence": EVIDENCE.replace("我喜欢", "朋友喜欢")}, SOURCE.replace("我喜欢", "朋友喜欢")),
        (
            {"evidence": "我喜欢测试红茶。测试紫茶且" + CONDITION + "。"},
            "我喜欢测试红茶。测试紫茶且" + CONDITION + "。",
        ),
        ({"evidence": "“" + EVIDENCE + "”"}, "“" + EVIDENCE + "”"),
        ({"kind": "dislike"}, SOURCE),
        ({"value": "测试紫茶(仅表格已提交且预约确认通过时）"}, SOURCE),
    ],
)
def test_condition_or_actor_mismatch_does_not_shorten_an_unproven_label(changes, source):
    assert parse(candidate(**changes), source=source) == []


def test_already_literal_short_index_retains_ordinary_admission():
    proposals = parse(candidate(value=HEAD))
    assert len(proposals) == 1 and dict(proposals[0].qualifiers)["condition"] == CONDITION


def test_unqualified_literal_parenthesized_object_is_not_a_condition_label():
    source = "我喜欢紫茶（无咖啡因）。"
    proposals = parse(candidate(value="紫茶（无咖啡因）", content="", evidence=source, qualifiers={}), source=source)
    assert len(proposals) == 1 and "紫茶（无咖啡因）" in proposals[0].memory.content


def test_wrong_preference_target_cannot_be_recovered_by_a_literal_other_head():
    old = dict(
        id=1, memory_key="preference_测试红茶", memory_type="user_fact", content="用户喜欢测试红茶", status="active"
    )
    assert (
        parse(candidate("SUPERSEDE", target_memory_id="1", target_memory_key=old["memory_key"]), existing=(old,)) == []
    )


@pytest.mark.parametrize(
    "kind,value,condition",
    [
        ("like", "紫茶(仅表格已提交并且预约确认通过时)", "只有表格已提交和预约确认通过才饮用紫茶"),
        ("dislike", "紫茶（仅表格已提交且预约确认通过时）", "只有表格已提交和预约确认通过才饮用紫茶"),
        ("like", "紫茶（仅表格已提交或预约确认通过时）", "只有表格已提交或预约确认通过才饮用紫茶"),
    ],
)
def test_additional_literal_kind_bracket_and_disjunction_variants(kind, value, condition):
    predicate = "我不喜欢" if kind == "dislike" else "我喜欢"
    source = "从2027年6月1日起，" + predicate + "紫茶且" + condition + "。这不是实际饮用。"
    raw = candidate(
        kind=kind,
        value=value,
        content="",
        evidence=source,
        qualifiers={"condition": condition, "time": "2027年6月1日", "certainty": "这不是实际饮用"},
    )
    proposals = parse(raw, source=source)
    assert len(proposals) == 1 and proposals[0].memory.memory_key == "preference_紫茶"
    assert dict(proposals[0].qualifiers)["condition"] == condition
    assert ("不喜欢" in proposals[0].memory.content) == (kind == "dislike")
    assert ("或" in proposals[0].memory.content) == ("或" in condition)


def test_sufficient_condition_does_not_authorize_a_necessary_condition_label():
    source = "从2027年6月1日起，我喜欢紫茶且只要表格已提交和预约确认通过就饮用紫茶。"
    raw = candidate(
        value="紫茶（仅表格已提交且预约确认通过时）",
        content="",
        evidence=source,
        qualifiers={"condition": "只要表格已提交和预约确认通过就饮用紫茶"},
    )
    assert parse(raw, source=source) == []
