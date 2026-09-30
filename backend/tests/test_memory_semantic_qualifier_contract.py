"""Admission regressions from real DeepSeek proposals with complete evidence."""

import json

import pytest

from character.memory_llm import parse_llm_proposals


def parse(source, value, evidence=None, **changes):
    raw = dict(
        kind="location", value=value, evidence=evidence or source, confidence=0.98, operation="ADD", qualifiers={}
    )
    raw.update(changes)
    return parse_llm_proposals(json.dumps({"memories": [raw]}, ensure_ascii=False), source_message=source)


@pytest.mark.parametrize(
    "source,evidence,value,label,key",
    [
        ("我叫叶澄，老家在宣城，目前住在丽水。", "老家在宣城", "宣城", "hometown", "user_origin"),
        ("我叫叶澄，老家在宣城，目前住在丽水。", "目前住在丽水", "丽水", "current_residence", "user_residence"),
        ("我的故乡是崇左。", "我的故乡是崇左", "崇左", "hometown", "user_origin"),
        ("我目前住在北海。", "我目前住在北海", "北海", "current_residence", "user_residence"),
    ],
)
def test_location_label_is_redundant_only_when_evidence_proves_field(source, evidence, value, label, key):
    (proposal,) = parse(source, value, evidence, qualifiers={"type": label})
    assert proposal.memory.memory_key == key
    assert proposal.qualifiers == ()


@pytest.mark.parametrize(
    "source,value,key",
    [
        ("我的老家在宣城。", "宣城", "user_origin"),
        ("我目前住在丽水。", "丽水", "user_residence"),
    ],
)
def test_field_does_not_depend_on_model_metadata(source, value, key):
    (proposal,) = parse(source, value)
    assert proposal.memory.memory_key == key


@pytest.mark.parametrize(
    "source,value,evidence,qualifiers",
    [
        ("我目前住在丽水。", "丽水", None, {"type": "hometown"}),
        ("我喜欢宣城。", "宣城", None, {"type": "hometown"}),
        ("我目前住在丽水。", "丽水", None, {"type": "unknown"}),
        ("我目前住在丽水。", "丽水", None, {"type": "current_residence", "secret": "x"}),
        ("我姐姐目前住在丽水。", "丽水", "目前住在丽水", {"type": "current_residence"}),
        ("假设我目前住在丽水。", "丽水", "我目前住在丽水", {"type": "current_residence"}),
        ("不要记住我目前住在丽水。", "丽水", "我目前住在丽水", {"type": "current_residence"}),
        ("我的老家在宣城吗？", "宣城", None, {"type": "hometown"}),
        ("我姐姐的老家在宣城。", "宣城", "老家在宣城", {"type": "hometown"}),
        ("她的故乡是崇左。", "崇左", "故乡是崇左", {"type": "hometown"}),
        ("小陈的老家在宣城。", "宣城", "老家在宣城", {"type": "hometown"}),
        ("假设我叫叶澄，老家在宣城。", "宣城", "老家在宣城", {"type": "hometown"}),
    ],
)
def test_model_label_cannot_supply_fact_or_bypass_admission(source, value, evidence, qualifiers):
    assert not parse(source, value, evidence, qualifiers=qualifiers)


def test_planned_certainty_is_grounded_and_remains_pending():
    source = "我计划2026年11月搬到安庆，搬家尚未发生。"
    (proposal,) = parse(
        source,
        "计划2026年11月搬到安庆",
        kind="other_user_fact",
        operation="PENDING",
        qualifiers={"certainty": "planned"},
    )
    assert proposal.operation == "PENDING"
    assert dict(proposal.qualifiers) == {"certainty": "计划"}
    assert proposal.memory.content.startswith("待确认：")


@pytest.mark.parametrize(
    "source,operation,label",
    [
        ("我住在安庆。", "PENDING", "planned"),
        ("我计划搬到安庆。", "ADD", "planned"),
        ("我计划搬到安庆。", "PENDING", "unknown"),
    ],
)
def test_certainty_label_cannot_invent_pending_evidence(source, operation, label):
    assert not parse(source, "安庆", kind="other_user_fact", operation=operation, qualifiers={"certainty": label})


@pytest.mark.parametrize("case_id", ["phase1_complete_profile", "phase1_future_not_present"])
def test_frozen_real_deepseek_output_preserves_complete_structured_facts(case_id):
    from pathlib import Path

    fixtures = json.loads(
        (Path(__file__).parent / "fixtures/deepseek_qualifier_admission.json").read_text(encoding="utf-8")
    )
    fixture = next(case for case in fixtures if case["case"] == case_id)
    proposals = parse_llm_proposals(
        json.dumps(fixture["writer_output"], ensure_ascii=False), source_message=fixture["source_message"]
    )
    assert len(proposals) == fixture["after_expected"]
    assert [p.memory.memory_key for p in proposals] == fixture["after_keys"]
    assert [p.operation for p in proposals] == fixture["after_operations"]
