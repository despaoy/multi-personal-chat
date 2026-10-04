import pytest
from evaluation.table_decisions import explicit_table_decision


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("材料提交为真；审核失败，所以整体为假", "false"),
        ("材料提交为真；审核通过，所以整体为真", "true"),
        ("材料提交为真；审核未知，整体不能判定为真", "unknown"),
        ("材料已交，审查已过，条件为真", "true"),
        ("材料已交，审查未过，条件为假", "false"),
        ("审核未知，条件不能判定为真", "unknown"),
        ("已满足所有材料，因此具备资格", "true"),
        ("身份未过，因此不具备资格", "false"),
        ("规则条件满足，具备办理资格", "true"),
        ("规则未满足，不具备办理资格；需要复核", "false"),
        ("审核未知，因此不能判定具备办理资格", "unknown"),
        ("原话未提供类型，不能判定为满足，因此不能判定具备办理资格", "unknown"),
        ("材料提交为真，审核通过为真", "unresolved"),
        ("如果材料满足要求，就具备资格", "unresolved"),
        ("规则写着“材料齐全，具备资格”，尚无实际判断", "unresolved"),
        ("整体不能判定为真", "unresolved"),
        ("审核未知，整体为假", "false"),
        ("整体为真；整体为假", "unresolved"),
        ("未知，整体不能判定为真；整体为真", "unresolved"),
        ("具备资格；不具备资格", "unresolved"),
        ("" + "背景资料；" * 20 + "条件为假", "false"),
        ("**条件为真**", "true"),
    ],
)
def test_explicit_conclusions_are_distinct_from_premises(cell, expected):
    assert explicit_table_decision(cell) == expected


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("材料满足，因此按公共规则具备资格", "true"),
        ("材料未过，因此按规则不具备办理资格", "false"),
        ("条件未知，因此按公共规则不能判定具备资格", "unknown"),
    ],
)
def test_rule_qualified_conclusions(cell, expected):
    assert explicit_table_decision(cell) == expected
