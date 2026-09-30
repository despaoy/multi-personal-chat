"""Repair a known type/field schema collision only from independently supported evidence."""

import json

import pytest

from character.memory_llm import parse_llm_proposals


def parse(source, value, **changes):
    raw = dict(kind="user_fact", value=value, evidence=source, confidence=.96, operation="ADD")
    raw.update(changes)
    return parse_llm_proposals(json.dumps({"memories": [raw]}), source_message=source)


@pytest.mark.parametrize("source,value,key,content", [
    ("我叫周岚。", "周岚", "user_name", "用户说自己叫周岚"),
    ("我的专业是气象学。", "气象学", "user_major", "用户说自己的专业是气象学"),
    ("我是大二学生。", "大二", "user_study_stage", "用户说自己是大二"),
    ("我在出版社工作。", "出版社", "user_workplace", "用户说自己在出版社工作"),
    ("我来自淮安。", "淮安", "user_origin", "用户说自己来自淮安"),
    ("我住在漳州。", "漳州", "user_residence", "用户说自己居住在漳州"),
])
def test_generic_storage_type_uses_unique_supported_field(source, value, key, content):
    proposals = parse(source, value)
    assert len(proposals) == 1
    assert proposals[0].memory.memory_key == key
    assert proposals[0].memory.content == content


@pytest.mark.parametrize("source,value,changes", [
    ("我来自淮安。", "南京", {}),
    ("我的朋友住在漳州。", "漳州", {}),
    ("假设我住在漳州。", "漳州", {}),
    ("我住在漳州吗？", "漳州", {}),
    ("不要记住我住在漳州。", "漳州", {}),
    ("我住在漳州。", "漳州", {"confidence": .1}),
    ("我住在漳州。", "漳州", {"kind": "arbitrary_schema"}),
    ("我来自淮安，我住在淮安。", "淮安", {}),
    # This wording is not covered by the existing independent extractor;
    # a generic type alone is not enough to recover the specific field.
    ("我现在读大二。", "大二", {}),
])
def test_schema_repair_does_not_guess_or_weaken_other_checks(source, value, changes):
    assert parse(source, value, **changes) == []


def test_generic_content_is_reconstructed_from_evidence_not_model_paraphrase():
    proposals = parse("我的专业是气象学。", "气象学", content="用户在气象学机构工作")
    assert len(proposals) == 1
    assert proposals[0].memory.content == "用户说自己的专业是气象学"
