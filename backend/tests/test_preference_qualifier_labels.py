"""Fictional component cases for redundant labels, never a model evaluation."""

import json

import pytest

from character.memory_llm import parse_llm_proposals


def proposal(source, evidence, *, kind="dislike", qualifiers=None, operation="ADD"):
    response = json.dumps(
        {
            "memories": [
                {
                    "kind": kind,
                    "value": "紫樱茶",
                    "evidence": evidence,
                    "confidence": 0.95,
                    "operation": operation,
                    "qualifiers": qualifiers,
                }
            ]
        },
        ensure_ascii=False,
    )
    return parse_llm_proposals(response, source_message=source)


@pytest.mark.parametrize("kind,verb,label", [("dislike", "不喜欢", "否定"), ("like", "喜欢", "肯定")])
def test_complete_self_preference_ignores_only_redundant_labels(kind, verb, label):
    evidence = f"我{verb}紫樱茶。"
    result = proposal(
        evidence + "我喜欢菱谷饼，但只有周五在家才选择。",
        evidence,
        kind=kind,
        qualifiers={"certainty": "稳定", "preference_strength": label},
    )
    assert len(result) == 1
    assert result[0].memory.content == f"用户说{verb}紫樱茶"
    assert not result[0].qualifiers


@pytest.mark.parametrize(
    "qualifiers",
    [
        {"certainty": "稳定", "preference_strength": "肯定"},
        {"certainty": "稳定", "condition": "周五"},
        {"certainty": "可能"},
        {"preference_strength": "很强"},
    ],
)
def test_other_labels_and_unrelated_conditions_still_require_quote(qualifiers):
    assert not proposal("我不喜欢紫樱茶。我周五喜欢菱谷饼。", "我不喜欢紫樱茶。", qualifiers=qualifiers)


@pytest.mark.parametrize(
    "source,evidence",
    [
        ("只有周五我不喜欢紫樱茶。", "我不喜欢紫樱茶。"),
        ("我的朋友说我不喜欢紫樱茶。", "我不喜欢紫樱茶。"),
        ("我不喜欢紫樱茶，但周末除外。", "我不喜欢紫樱茶"),
    ],
)
def test_clipped_conditional_or_other_subject_cannot_gain_self_authority(source, evidence):
    assert not proposal(source, evidence, qualifiers={"certainty": "稳定", "preference_strength": "否定"})


def test_actual_literal_preference_qualification_remains():
    result = proposal(
        "我不喜欢紫樱茶。",
        "我不喜欢紫樱茶。",
        qualifiers={"certainty": "稳定", "preference_strength": "否定", "context": "紫樱茶"},
    )
    assert len(result) == 1
    assert dict(result[0].qualifiers) == {"context": "紫樱茶"}
