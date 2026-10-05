"""Independent complete fictional alias qualifiers; no model approval claimed."""

import json

import pytest

from character.memory_llm import parse_llm_proposals

QUOTE = "我的常用别名是澄羽。这个本人常用别名是没有角色专用别名记录时使用的全角色默认记录，所有角色记住这条默认别名。"
CONDITION = "没有角色专用别名记录时使用的全角色默认记录"
SOURCE = (
    QUOTE
    + "已有角色专用别名保持，不替换、撤回或删除。澄羽只是别名，不是规范姓名。这是本人明确立即生效的默认记录新增，不是待确认、假设或未来计划。"
)


def parse(*, certainty="明确立即生效", context=CONDITION):
    return parse_llm_proposals(
        json.dumps(
            {
                "memories": [
                    dict(
                        kind="name",
                        value="澄羽",
                        evidence=QUOTE,
                        operation="ADD",
                        attributed_to="user",
                        confidence=0.97,
                        qualifiers={"context": context, "certainty": certainty},
                        scope_level="user_global",
                    )
                ]
            },
            ensure_ascii=False,
        ),
        source_message=SOURCE,
        confidence_threshold=0.85,
    )


def test_asserted_alias_lifecycle_label_preserves_full_global_default_condition():
    (proposal,) = parse()
    assert proposal.memory.memory_key == "user_alias"
    assert proposal.scope_level == "user_global" and proposal.operation == "ADD"
    assert proposal.qualifiers == (("context", CONDITION),)
    assert proposal.evidence == QUOTE
    assert proposal.target_memory_id == "" and proposal.target_memory_key == ""


@pytest.mark.parametrize("certainty", ["planned", "hypothetical"])
def test_other_certainty_labels_do_not_gain_asserted_alias_admission(certainty):
    assert not parse(certainty=certainty)


def test_asserted_label_does_not_remove_unsupported_alias_use_condition():
    assert not parse(context="订单已付款")
