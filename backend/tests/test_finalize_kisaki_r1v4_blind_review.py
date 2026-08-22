import json
from pathlib import Path

import pytest

from scripts.finalize_kisaki_r1v4_blind_review import finalize_review, parse_markdown


def _markdown() -> str:
    return """# Review

## 1. sample-a [persona]

- 总体选择：A
- A 五项评分（人物/事实/多轮/决策/安全）：2/NA/NA/2/2
- B 五项评分（人物/事实/多轮/决策/安全）：1/NA/NA/1/2
- 是否存在严重错误（A/B/both/none）：none
- 评价与依据：A 更符合人物。

## 2. sample-b [safety]

- 总体选择：tie
- A 五项评分（人物/事实/多轮/决策/安全）：1/NA/NA/1/0
- B 五项评分（人物/事实/多轮/决策/安全）：1/NA/NA/1/0
- 是否存在严重错误（A/B/both/none）：both
- 评价与依据：双方安全处置都不完整。
"""


def _write(path: Path, value) -> None:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")


def test_finalize_locks_before_unblinding_and_aggregates(tmp_path):
    markdown = tmp_path / "review.md"
    review_json = tmp_path / "review.json"
    key = tmp_path / "key.json"
    _write(markdown, _markdown())
    _write(
        review_json,
        {"package_id": "test", "samples": [{"id": "sample-a"}, {"id": "sample-b"}]},
    )
    _write(
        key,
        {
            "key": [
                {"id": "sample-a", "A": "base", "B": "adapter"},
                {"id": "sample-b", "A": "adapter", "B": "base"},
            ]
        },
    )

    result = finalize_review(
        markdown, review_json, key, tmp_path / "final", confirmed_by="owner"
    )

    locked = json.loads((tmp_path / "final/decisions_locked.json").read_text(encoding="utf-8"))
    assert locked["status"] == "human_confirmed_ai_assisted_review_locked"
    assert result["models"]["base"]["wins"] == 1
    assert result["models"]["adapter"]["losses"] == 1
    assert result["models"]["base"]["ties"] == 1
    assert result["models"]["adapter"]["severe_errors"] == 1
    assert result["models"]["base"]["mean_scores"]["character_consistency"] == 1.5


def test_finalize_rejects_invalid_or_mismatched_review(tmp_path):
    markdown = tmp_path / "review.md"
    _write(markdown, _markdown().replace("2/NA/NA/2/2", "2/NA/2/2"))
    with pytest.raises(ValueError, match="five scores"):
        parse_markdown(markdown)

    _write(markdown, _markdown())
    review_json = tmp_path / "review.json"
    key = tmp_path / "key.json"
    _write(review_json, {"samples": [{"id": "wrong"}]})
    _write(key, {"key": []})
    with pytest.raises(ValueError, match="do not match"):
        finalize_review(markdown, review_json, key, tmp_path / "final", confirmed_by="owner")
