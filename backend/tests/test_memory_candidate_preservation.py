"""A coarse field key is not a unique statement or evidence identity."""

import json

import pytest

from character.memory_llm import parse_llm_proposals


@pytest.mark.parametrize("operation", ["ADD", "COEXIST", "PENDING"])
def test_distinct_statements_in_one_coarse_slot_survive(operation):
    # Explicit self-ownership isolates dedup from the separate legacy nickname
    # heuristic that currently misreads a bare "老家" as a person's name.
    source = "我老家是汕头，现在定居在中山。"
    candidates = [dict(kind="location", value=value, evidence=evidence,
                       operation=operation, confidence=.95)
                  for value, evidence in (("汕头", "我老家是汕头"), ("中山", "现在定居在中山"))]
    if operation == "COEXIST":
        for item in candidates:
            item.update(target_memory_id="7", target_memory_key="user_location")
    parsed = parse_llm_proposals(json.dumps({"memories": candidates}), source_message=source,
                                existing_memories=({"id": "7", "memory_key": "user_location",
                                                    "content": "用户的旧位置描述"},))
    assert len(parsed) == 2
    assert {item.evidence for item in parsed} == {item["evidence"] for item in candidates}


def test_identical_candidates_still_deduplicate():
    candidate = dict(kind="major", value="音乐学", evidence="我的专业是音乐学", operation="ADD", confidence=.95)
    parsed = parse_llm_proposals(json.dumps({"memories": [candidate, candidate]}),
                                source_message="我的专业是音乐学。")
    assert len(parsed) == 1


def test_different_evidence_is_not_discarded_as_same_fact():
    candidates = [dict(kind="major", value="音乐学", evidence=evidence, operation="ADD", confidence=.95)
                  for evidence in ("我的专业是音乐学", "我学的是音乐学")]
    parsed = parse_llm_proposals(json.dumps({"memories": candidates}),
                                source_message="我的专业是音乐学，我学的是音乐学。")
    assert len(parsed) == 2
