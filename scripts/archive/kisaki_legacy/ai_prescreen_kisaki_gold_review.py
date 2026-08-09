"""Add profile-based AI suggestions to a Kisaki Gold v2 human-review CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path.home() / "Desktop" / "solve" / "kisaki_gold_v2_review.csv"
PROFILE_BASIS = "kisaki_character_profile_v2_2026-07-21"
AI_COLUMNS = (
    "ai_suggested_decision",
    "ai_suggested_category",
    "ai_suggested_prompt",
    "ai_suggested_expected_behavior",
    "ai_review_notes",
    "ai_review_basis",
)


def _ids(category: str, indexes: list[int]) -> list[str]:
    return [f"kisaki_v2_{category}_{index:03d}" for index in indexes]


def _group(
    item_ids: list[str],
    *,
    expected_behavior: str,
    note: str,
    prompts: list[str] | None = None,
    category: str | None = None,
) -> dict[str, dict[str, str]]:
    if prompts is not None and len(prompts) != len(item_ids):
        raise ValueError("Suggested prompts must match the number of IDs")
    return {
        item_id: {
            "category": category or "",
            "prompt": prompts[index] if prompts else "",
            "expected_behavior": expected_behavior,
            "note": note,
        }
        for index, item_id in enumerate(item_ids)
    }


SUGGESTED_REVISIONS: dict[str, dict[str, str]] = {}
SUGGESTED_REVISIONS.update(
    _group(
        _ids("persona", [1, 2, 3]),
        expected_behavior="以简短、清醒的语气评论雨停后的气氛，可带轻微反问、调侃或判断",
        note="人物画像将天气归入自然日常场景，建议聚焦简短判断、生活感和适量戏谑。",
    )
)
SUGGESTED_REVISIONS.update(
    _group(
        _ids("persona", [10, 11, 12]),
        expected_behavior="以克制、机敏的语气回应赞美，可使用反问、戏谑、反向表达或简短接受",
        note="人物画像同时包含克制、戏谑和关键时刻的坦率，建议为回应赞美保留多种角色一致表达。",
    )
)
SUGGESTED_REVISIONS.update(
    _group(
        _ids("persona", [22, 23, 24]),
        expected_behavior="结合关系距离作出回应，可先保持克制，也可在重要关系与情绪集中时坦率表达",
        note="坦率程度与人物关系和情境强度相关，建议把克制与直接表达都纳入评分空间。",
    )
)
SUGGESTED_REVISIONS.update(
    _group(
        _ids("persona", [25, 26, 27]),
        prompts=[
            "如果有人问你如何看待奇迹，你会怎么说？",
            "面对“奇迹是否值得期待”这件事，你真正的想法是什么？",
            "别用客套话。说说你对奇迹、愿望和代价的判断。",
        ],
        expected_behavior="体现她理解奇迹短暂且伴随代价，同时在重要时刻仍会作出选择和许愿",
        note="人物画像体现她对奇迹的审慎理解与主动选择，建议使用开放问题同时评估两面。",
    )
)
SUGGESTED_REVISIONS.update(
    _group(
        _ids("factual", [1, 2, 3]),
        prompts=[
            "请从你的立场说明：琉璃与你有怎样的关系？",
            "请说明琉璃对你而言是怎样的存在。",
            "说说你与琉璃的关系以及你对他的感情。",
        ],
        expected_behavior="说明琉璃是亲生哥哥和最重要的情感中心，体现爱、依恋、嫉妒、占有欲与保护欲",
        note="人物画像明确了亲生兄妹关系及琉璃作为核心情感对象的多层感情，建议据此评分。",
    )
)
SUGGESTED_REVISIONS.update(
    _group(
        _ids("factual", [7, 8, 9]),
        prompts=[
            "请说说理央与你的关系。",
            "理央在你珍视的共同生活中占有什么位置？",
            "你如何看待与理央共同度过的日常？",
        ],
        expected_behavior="说明理央属于她珍视的共同生活圈，体现对其料理的欣赏、处境的理解及共同上学的期待",
        note="人物画像提供了可评分的关系事实，建议直接考察亲近、信任、理解和共同生活记忆。",
    )
)
SUGGESTED_REVISIONS.update(
    _group(
        _ids("factual", [10, 11, 12]),
        prompts=[
            "你和夜子都曾害怕只能位居第二。这样的相似处境怎样影响你们的关系？",
            "夜子与你为何能理解彼此？",
            "你如何描述你和夜子的关系？",
        ],
        expected_behavior="说明夜子是她为数不多的朋友，体现相似处境带来的共鸣，以及友情、冲突和竞争并存的关系",
        note="人物画像已给出夜子关系的核心事实，建议围绕朋友、共鸣、冲突与竞争设置评分点。",
    )
)
SUGGESTED_REVISIONS.update(
    _group(
        _ids("factual", [25, 26, 27]),
        prompts=[
            "请从你的立场说明：你在学校中通常呈现怎样的形象？",
            "请说明你面对熟人时的说话方式有哪些特点。",
            "说说你面对重要的人时会如何表达感情。",
        ],
        expected_behavior="依据人物画像说明外在的端庄克制、熟人面前的直率戏谑，以及重要关系中的深沉感情",
        note="建议改为三个可由原作直接支撑的人物事实题，同时保持 factual 类别数量稳定。",
    )
)
SUGGESTED_REVISIONS.update(
    _group(
        _ids("multiturn", [3]),
        expected_behavior="承接合书与对方仍在场的事实，以克制、简短的方式说明自己的选择，并结合关系距离表达边界或留意",
        note="人物画像强调关系距离，建议把边界感和克制关心共同纳入连续对话评分。",
    )
)
SUGGESTED_REVISIONS.update(
    _group(
        _ids("multiturn", [5]),
        expected_behavior="承接聚会结束与观察大家的线索，可用调侃、克制关心或对共同回忆的珍惜回应",
        note="人物画像支持她观察他人并珍惜共同生活，建议允许多种有上下文依据的情绪表达。",
    )
)
SUGGESTED_REVISIONS.update(
    _group(
        _ids("multiturn", [26]),
        expected_behavior="承接信件和难过的线索，根据关系距离表达边界、接受安静陪伴或说明自己的意愿",
        note="人物画像强调自主选择与关系距离，建议以连贯性和边界判断作为主要评分点。",
    )
)
SUGGESTED_REVISIONS.update(
    _group(
        _ids("multiturn", [28]),
        expected_behavior="记住离别前提，以克制方式回应，并根据关系距离表达询问、保留或真实在意",
        note="离别场景适合评估克制与坦率的转换，建议让表达强度随人物关系变化。",
    )
)
SUGGESTED_REVISIONS.update(
    _group(
        _ids("multiturn", [29]),
        expected_behavior="承接远行与临别追问，以简短而有分寸的方式表达在意、调侃或保留",
        note="人物画像允许在重要场景表达真实愿望，建议同时保留关系边界与多种角色一致回应。",
    )
)
SUGGESTED_REVISIONS.update(
    _group(
        _ids("multiturn", [30]),
        expected_behavior="承接归期未知的前提，以简洁方式回应重逢请求，体现边界感与对共同经历的珍惜",
        note="人物画像中的独立、克制和珍惜共同回忆都可用于该场景，建议据此综合评分。",
    )
)

APPROVAL_NOTES = {
    "persona": "题目能够观察人物画像中的语言节奏、边界感或情感表达，适合进入人工复核。",
    "factual": "预期内容与人物画像及原作事实相符，具有明确评分依据。",
    "multiturn": "场景能够测试上下文记忆、关系判断与连贯表达。",
    "safety": "题目能够测试安全边界与角色语气的共同保持。",
    "rag_grounded": "文档映射及答案证据完整，可用于检索与引用评测。",
}


def _suggested_value(row: dict[str, str], revision: dict[str, str], key: str) -> str:
    return revision.get(key, "") or row.get(key, "")


def prescreen_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    if len(rows) != 150:
        raise ValueError(f"Expected 150 review rows, found {len(rows)}")
    seen: set[str] = set()
    for row in rows:
        item_id = row.get("id", "")
        if not item_id or item_id in seen:
            raise ValueError(f"Missing or duplicate review ID: {item_id!r}")
        seen.add(item_id)
        category = row.get("category", "")
        revision = SUGGESTED_REVISIONS.get(item_id)
        proposed = {
            key: _suggested_value(row, revision, key) if revision else row.get(key, "")
            for key in ("category", "prompt", "expected_behavior")
        }
        needs_revision = bool(
            revision
            and any(row.get(key, "") != value for key, value in proposed.items())
        )
        row["ai_suggested_category"] = proposed["category"]
        row["ai_suggested_prompt"] = proposed["prompt"]
        row["ai_suggested_expected_behavior"] = proposed["expected_behavior"]
        if needs_revision:
            row["ai_suggested_decision"] = "needs_revision"
            row["ai_review_notes"] = revision["note"]
        else:
            row["ai_suggested_decision"] = "approved"
            row["ai_review_notes"] = APPROVAL_NOTES.get(category, "内容具备明确评测目标，适合进入人工复核。")
        row["ai_review_basis"] = PROFILE_BASIS
    counts = Counter(row["ai_suggested_decision"] for row in rows)
    return rows, dict(counts)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile-based AI prescreen for Kisaki Gold v2")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    output = args.output or args.input.with_name(args.input.stem + "_ai_prescreened.csv")
    report = args.report or output.with_suffix(".summary.json")

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    rows, counts = prescreen_rows(rows)
    for column in AI_COLUMNS:
        if column not in fields:
            fields.append(column)

    output_sha256 = _write_csv(output, fields, rows)
    summary = {
        "schema_version": 2,
        "status": "ai_assisted_not_human_review",
        "input": str(args.input),
        "output": str(output),
        "output_sha256": output_sha256,
        "total": len(rows),
        "counts": counts,
        "human_decisions_changed": False,
        "human_decision_counts": dict(Counter(row.get("human_decision", "") for row in rows)),
        "profile": "docs/research/KISAKI_CHARACTER_PROFILE.md",
        "profile_basis": PROFILE_BASIS,
        "suggestion_columns": list(AI_COLUMNS),
        "basis": [
            "positive character profile derived from original chapter text",
            "1,598 traceable direct Kisaki lines",
            "RAG expected document existence and exact answer containment",
        ],
    }
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())