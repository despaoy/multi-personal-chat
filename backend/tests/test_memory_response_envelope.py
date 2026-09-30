"""Equivalent JSON containers must preserve candidate validation and atomic parsing."""

import json

import pytest

from character.memory_llm import parse_llm_proposals


def candidate(value="物理学"):
    return dict(kind="major", value=value, evidence="我的专业是物理学", confidence=.95, operation="ADD")


@pytest.mark.parametrize("prefix,suffix", [("", ""), ("```json\n", "\n```"), ("结果如下：\n", "")])
def test_bare_candidate_array_matches_envelope(prefix, suffix):
    source = "我的专业是物理学。"
    expected = parse_llm_proposals(json.dumps({"memories": [candidate()]}), source_message=source)
    actual = parse_llm_proposals(prefix + json.dumps([candidate()]) + suffix, source_message=source)
    assert actual == expected
    assert len(actual) == 1


def test_bare_array_does_not_bypass_evidence_validation():
    assert parse_llm_proposals(json.dumps([candidate("考古学")]), source_message="我的专业是物理学。") == []


@pytest.mark.parametrize("raw", ["{}", '{"result": []}', '{"kind":"major"}',
                                '[' + json.dumps(candidate()) + ','])
def test_invalid_or_truncated_container_is_failure_not_no_change(raw):
    with pytest.raises(ValueError):
        parse_llm_proposals(raw, source_message="我的专业是物理学。")


@pytest.mark.parametrize("raw", ['{"memories":[]}', '[]'])
def test_explicit_empty_collection_is_no_change(raw):
    assert parse_llm_proposals(raw, source_message="我的专业是物理学。") == []
