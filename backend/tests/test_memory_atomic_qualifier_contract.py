"""Lossless, atomic qualifier admission using independent complete sources."""

import json

import pytest

from character.memory_llm import (
    _extract_json,
    _sanitize_qualifiers,
    build_memory_llm_messages,
    parse_llm_proposals,
)

SOURCE = (
    "我喜欢蓝莓茶。条件：申请已提交；场景：仅在静夜；频率：每周一次；"
    "程度：比较喜欢；确定性：确实如此；例外：未通过不使用；地点：甲室；时间：每年冬季。"
)
QUALIFIERS = dict(
    condition="申请已提交",
    context="仅在静夜",
    frequency="每周一次",
    preference_strength="比较喜欢",
    certainty="确实如此",
    exception="未通过不使用",
    location="甲室",
    time="每年冬季",
)


def positive():
    assert len(SOURCE) < 120 and all(value in SOURCE for value in QUALIFIERS.values())
    assert dict(_sanitize_qualifiers(QUALIFIERS, evidence=SOURCE)) == QUALIFIERS


def proposal(qualifiers, *, source=SOURCE, operation="ADD"):
    return parse_llm_proposals(
        json.dumps(
            dict(
                memories=[
                    dict(
                        kind="like",
                        value="蓝莓茶",
                        evidence=source,
                        confidence=0.98,
                        operation=operation,
                        qualifiers=qualifiers,
                    )
                ]
            ),
            ensure_ascii=False,
        ),
        source_message=source,
    )


def test_eight_fields_survive_proposal_and_persistence_dictionary():
    positive()
    (accepted,) = proposal(QUALIFIERS)
    assert dict(accepted.qualifiers) == QUALIFIERS
    assert accepted.operation == "ADD"


def test_field_order_cannot_remove_exception_or_time():
    positive()
    assert dict(_sanitize_qualifiers(dict(reversed(list(QUALIFIERS.items()))), evidence=SOURCE)) == QUALIFIERS


@pytest.mark.parametrize(
    "tail",
    [
        {"unsupported": "申请已提交"},
        {"location": "乙室"},
        {"location": {"nested": "甲室"}},
        {"location": ""},
        {"location": "甲" * 49},
    ],
)
def test_invalid_suffix_rejects_whole_proposal(tail):
    positive()
    baseline = dict(list(QUALIFIERS.items())[:6])
    assert len(proposal(baseline)) == 1
    assert _sanitize_qualifiers({**baseline, **tail}, evidence=SOURCE) is None
    assert not proposal({**baseline, **tail})


@pytest.mark.parametrize("collision", [" Condition ", "CONDITION", "condition "])
def test_normalized_key_collision_is_not_last_value_wins(collision):
    positive()
    assert (
        _sanitize_qualifiers(
            {"condition": QUALIFIERS["condition"], collision: QUALIFIERS["exception"]}, evidence=SOURCE
        )
        is None
    )


def test_list_compatibility_never_discards_a_second_context():
    positive()
    assert _sanitize_qualifiers([QUALIFIERS["context"]], evidence=SOURCE) == (("context", QUALIFIERS["context"]),)
    assert _sanitize_qualifiers([QUALIFIERS["context"], QUALIFIERS["condition"]], evidence=SOURCE) is None


@pytest.mark.parametrize(
    "key,value,pending",
    [
        ("condition", True, False),
        ("condition", False, True),
        ("exception", "true", True),
        ("certainty", True, True),
        ("certainty", False, False),
    ],
)
def test_boolean_cannot_certify_an_unobserved_condition(key, value, pending):
    positive()
    assert _sanitize_qualifiers({key: value}, evidence=SOURCE, pending=pending) is None


def test_pending_false_certainty_keeps_uncertain_lifecycle():
    positive()
    source = "我可能喜欢蓝莓茶，还不确定。"
    (accepted,) = proposal({"certainty": False}, source=source, operation="PENDING")
    assert accepted.operation == "PENDING"
    assert dict(accepted.qualifiers) == {"certainty": "false"}
    assert accepted.memory.content.startswith("待确认：")


def test_literal_boolean_word_is_not_a_structural_bypass():
    positive()
    assert _sanitize_qualifiers({"context": "true"}, evidence="原文标签是true。") == (("context", "true"),)


@pytest.mark.parametrize(
    "text",
    [
        '{"memories":[],"memories":[{}]}',
        '{"memories":[{"operation":"ADD","operation":"PENDING"}]}',
        '{"memories":[{"qualifiers":{"condition":"甲","condition":"乙"}}]}',
    ],
)
def test_duplicate_json_is_a_format_error_before_any_candidate_admission(text):
    positive()
    assert _extract_json('{"memories":[]}') == {"memories": []}
    with pytest.raises(ValueError, match="重复"):
        _extract_json(text)


def test_prompt_advertises_the_actual_lossless_contract():
    positive()
    messages = build_memory_llm_messages(SOURCE, (), (), (), 2000, 0.8, context_window_tokens=65536)
    payload = json.loads(messages[-1]["content"])
    assert payload["current_user_message"] == SOURCE
    contract = payload["proposal_constraints"]
    assert contract["max_qualifiers"] == len(contract["qualifier_keys"]) == 8
    assert contract["qualifier_keys_unique"] is True
    assert contract["qualifier_truncation_allowed"] is False
