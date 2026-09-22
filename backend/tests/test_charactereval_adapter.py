import copy
import json

import pytest
from scripts.audit_charactereval import audit

from evaluation.charactereval_adapter import (
    adapt_character_case,
    dialogue_messages,
    load_pinned_corpus,
    stable_character_subset,
)


def row(key=1, role="角色甲"):
    return {
        "id": key,
        "role": role,
        "novel_name": "独立作品",
        "context": f"{role}：此前：我说过的话。\n角色乙：接着说吧。",
    }


def test_named_speakers_and_every_profile_field_are_preserved_without_evaluation_labels():
    profiles = {"角色甲": {"姓名": "角色甲", "人物性格": ["克制", "认真"], "人物经历": "完整原始经历"}}
    case = adapt_character_case(row(), profiles, {"1": [["secret_metric", "不应进入生成"]]})
    assert json.loads(case.profile) == profiles["角色甲"]
    assert case.history[0] == {"role": "assistant", "content": "角色甲：此前：我说过的话。"}
    assert case.query == "角色乙：接着说吧。"
    messages = dialogue_messages(case)
    assert "secret_metric" not in json.dumps(messages)
    assert "独立作品" not in json.dumps(messages, ensure_ascii=False)
    assert "\n".join(message["content"] for message in messages) == row()["context"]


@pytest.mark.parametrize("context", ["没说话人", "角色甲：末轮已经是目标角色", "角色乙：", "角色乙：有内容\n空白叙述"])
def test_malformed_dialogue_is_not_repaired_by_dropping_content(context):
    with pytest.raises(ValueError):
        adapt_character_case({**row(), "context": context}, {"角色甲": "人物资料"}, {})


def test_grouped_subset_is_order_and_annotation_independent():
    rows = [row(index, f"角色{index // 3}") for index in range(12)]
    first = stable_character_subset(rows, roles=2, per_role=2)
    changed = copy.deepcopy(rows[::-1])
    for value in changed:
        value["context"] = "different"
        value["untrusted_score"] = 100
    second = stable_character_subset(changed, roles=2, per_role=2)
    assert [value["id"] for value in first] == [value["id"] for value in second]
    assert len(first) == 4
    assert len({value["role"] for value in first}) == 2
    with pytest.raises(ValueError, match="duplicate"):
        stable_character_subset([row(), row()])


def test_audit_keeps_invalid_selected_case_and_does_not_copy_corpus_text():
    rows = [row(), {**row(2), "context": "坏格式"}]
    result = audit(rows, {"角色甲": {"姓名": "角色甲"}}, {}, roles=1, per_role=2)
    assert result["summary"]["selected_cases"] == 2
    assert result["summary"]["invalid_selected_cases"] == 1
    assert "接着说吧" not in json.dumps(result, ensure_ascii=False)
    assert {item["id"] for item in result["cases"]} == {"1", "2"}


def test_altered_public_cache_fails_before_use(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/test_data.jsonl").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        load_pinned_corpus(tmp_path)
