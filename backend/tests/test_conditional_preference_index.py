"""Conditional preferences use explicit literal objects and source-bound qualifiers."""

import json
from types import SimpleNamespace

import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, MemoryResponseError, parse_llm_proposals
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
        value=HEAD,
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
async def test_literal_object_preserves_full_condition_and_current_target(tmp_path, operation):
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
    assert SOURCE in view["evidence"] and SOURCE not in view["content"]
    if operation == "SUPERSEDE":
        assert new["relation_type"] == "COEXIST" and new["parent_memory_id"] == old["id"]


@pytest.mark.parametrize("value", [
    "测试紫茶（仅表格已提交且预约确认通过时）",
    "测试紫茶（仅表格已提交或预约确认通过时）",
])
def test_nonliteral_condition_labels_are_errors_instead_of_repaired_objects(value):
    with pytest.raises(MemoryResponseError) as caught:
        parse(candidate(value=value))
    assert caught.value.field == "value"


@pytest.mark.parametrize("source", [SOURCE.replace("我喜欢", "朋友喜欢"), "“" + EVIDENCE + "”"])
def test_other_speakers_and_quoted_preferences_are_not_promoted(source):
    raw = candidate(evidence=source)
    assert parse(raw, source=source) == []


def test_condition_must_still_be_present_in_the_original_evidence():
    raw = candidate(qualifiers={"condition": "只有身份核验通过才饮用测试紫茶"})
    with pytest.raises(MemoryResponseError) as caught:
        parse(raw)
    assert caught.value.field == "qualifiers"


def test_unqualified_literal_parenthesized_object_is_not_a_condition_label():
    source = "我喜欢紫茶（无咖啡因）。"
    proposals = parse(candidate(value="紫茶（无咖啡因）", content="", evidence=source, qualifiers={}), source=source)
    assert len(proposals) == 1 and "紫茶（无咖啡因）" in proposals[0].memory.content




@pytest.mark.parametrize(
    "kind,value,condition",
    [
        ("like", "紫茶(仅表格已提交并且预约确认通过时)", "只有表格已提交和预约确认通过才饮用紫茶"),
        ("dislike", "紫茶（仅表格已提交且预约确认通过时）", "只有表格已提交和预约确认通过才饮用紫茶"),
        ("like", "紫茶（仅表格已提交或预约确认通过时）", "只有表格已提交或预约确认通过才饮用紫茶"),
    ],
)
def test_literal_polarity_and_disjunction_preserve_the_full_source(kind, value, condition):
    predicate = "我不喜欢" if kind == "dislike" else "我喜欢"
    source = "从2027年6月1日起，" + predicate + "紫茶且" + condition + "。这不是实际饮用。"
    raw = candidate(
        kind=kind,
        value=value,
        content="",
        evidence=source,
        qualifiers={"condition": condition, "time": "2027年6月1日", "certainty": "这不是实际饮用"},
    )
    with pytest.raises(MemoryResponseError) as caught:
        parse(raw, source=source)
    assert caught.value.field == "value"
    proposals = parse(raw | {"value": "紫茶"}, source=source)
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
    with pytest.raises(MemoryResponseError) as caught:
        parse(raw, source=source)
    assert caught.value.field == "value"
    (proposal,) = parse(raw | {"value": "紫茶"}, source=source)
    assert dict(proposal.qualifiers)["condition"] == "只要表格已提交和预约确认通过就饮用紫茶"


@pytest.mark.parametrize("kind,value,source", [
    ("name", "冬青", "“我叫冬青。”"),
    ("other_user_fact", "自行车", "「我的自行车是蓝色的。」"),
])
def test_complete_quotation_is_not_promoted_to_another_personal_field(kind, value, source):
    assert parse(candidate(kind=kind, value=value, content="", evidence=source, qualifiers={}), source=source) == []


def test_quoted_object_name_inside_a_personal_statement_remains_supported():
    source = "我喜欢“测试紫茶”。"
    (proposal,) = parse(candidate(value=HEAD, content="", evidence=source, qualifiers={}), source=source)
    assert proposal.memory.memory_key == "preference_测试紫茶" and proposal.evidence == source


def source_reference_candidate(**changes):
    from character.memory_llm import build_memory_llm_messages

    messages = build_memory_llm_messages(SOURCE, (), (), (), 2000, .85)
    payload = json.loads(messages[-1]["content"])
    assert payload["current_user_message"] == SOURCE
    references = {body: key for key, body in payload["source_units"].items()}
    raw = candidate(qualifiers={"condition": {"source_unit": references[next(body for body in references if CONDITION in body)]},
        "certainty": {"source_unit": references[next(body for body in references if "这不是已经实际饮用" in body)]}})
    return raw | changes


def test_declared_source_references_preserve_condition_and_negation_verbatim():
    (proposal,) = parse(source_reference_candidate())
    assert dict(proposal.qualifiers) == {"condition": "我喜欢 测试紫茶且" + CONDITION, "certainty": "这不是已经实际饮用"}
    assert CONDITION in proposal.memory.content and proposal.evidence == " ".join(EVIDENCE.split())


@pytest.mark.parametrize("reference", [{"source_unit": "q999"}, {"source_unit": True},
    {"source_unit": "q0", "text": "虚构限定"}])
def test_unknown_or_ambiguous_source_references_are_not_guessed(reference):
    with pytest.raises(MemoryResponseError) as caught:
        parse(candidate(qualifiers={"condition": reference}))
    assert caught.value.field == "qualifiers"


def test_source_reference_cannot_extend_the_declared_evidence():
    raw = source_reference_candidate(evidence="我喜欢测试紫茶")
    with pytest.raises(MemoryResponseError) as caught:
        parse(raw)
    assert caught.value.field == "qualifiers"


def test_paraphrase_still_fails_instead_of_guessing_a_source_reference():
    with pytest.raises(MemoryResponseError) as caught:
        parse(candidate(qualifiers={"certainty": "尚未实际饮用"}))
    assert caught.value.field == "qualifiers"


def test_source_units_preserve_complete_clock_times():
    from character.memory_llm import build_memory_llm_messages

    source = "每天09:30以后，我喜欢紫茶，但只在课程10:00开始前饮用。"
    payload = json.loads(build_memory_llm_messages(source, (), (), (), 2000, .85)[-1]["content"])
    references = {body: key for key, body in payload["source_units"].items()}
    raw = candidate(value="紫茶", content="", evidence=source, qualifiers={
        "time": {"source_unit": references[next(body for body in references if "每天09:30以后" in body)]},
        "condition": {"source_unit": references[next(body for body in references if "但只在课程10:00开始前饮用" in body)]}})
    (proposal,) = parse(raw, source=source)
    assert dict(proposal.qualifiers) == {"time": "每天09:30以后", "condition": "但只在课程10:00开始前饮用"}
